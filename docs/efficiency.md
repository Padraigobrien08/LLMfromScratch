# Inference efficiency: the KV cache, quantization, speculative decoding

Three optimisations, all hand-implemented, all measured against the 124M reproduction.
The theme running through the results is that **the headline number and the useful number
are rarely the same one**: 4-bit quantization is a memory win that costs speed,
speculative decoding can hit its algorithmic ceiling while still losing on the clock, and
the KV cache — the one optimisation nobody thinks to question — was for a while making
decoding *slower*.

Throughput figures are measured on a rented **RTX 4090** (sm_89, torch 2.4.1+cu124,
measured 167.9 TFLOP/s dense bf16). Memory and quality figures are device-independent. See
[Where these numbers come from](#where-these-numbers-come-from).

---

## The KV cache, and the bug a benchmark found

The earlier version of this document reported that the cache gave no speedup, and
explained it as a real property rather than a defect: decoding a 124M model is bound by
streaming weights from memory, not by attention over the prefix, so recomputing a short
prefix costs almost nothing. It also noted the benchmark should sweep sequence length,
since that is the axis a cache exists for.

Both halves of that turned out to matter. The sweep got written, and it did not show what
it was supposed to show:

| Generated tokens | naive (recompute) | KV cache | cache advantage |
| --- | --- | --- | --- |
| 128 | 254 tok/s | 168 | **0.66×** |
| 256 | 252 | 169 | **0.67×** |
| 512 | 246 | 172 | **0.70×** |
| 1024 | 196 | 161 | **0.82×** |

Not merely "no speedup" — the cache was 34% *slower*, and the naive path beat it at every
length measured. Two features of that table say it is not a fact about hardware. Cache
throughput is flat, ~161–172 tok/s regardless of length, which is the signature of a
fixed per-step cost rather than anything to do with attention. And a cache that does
strictly less arithmetic than recomputation cannot lose to it on work; it can only lose
on overhead.

The cause was three lines in `attention.py`. A decode step has `q_len == 1` and
`kv_len > 1`, so it fell into the branch that builds an explicit bottom-right aligned
causal mask — and **passing `attn_mask` to `scaled_dot_product_attention` disqualifies it
from the fused flash and mem-efficient kernels**, dropping it onto the math backend. The
naive path, where `q_len == kv_len`, passed `is_causal=True` and stayed fused. So the
benchmark had been comparing a fused kernel against an unfused one and reporting the
difference as a property of caching.

For `q_len == 1` that mask is all-`True` anyway — every cached key precedes the single
query, so it encodes no constraint. It cost a mask allocation per layer per token *and*
the fused kernel, in exchange for nothing. After the fix:

| Generated tokens | naive | KV cache | cache advantage | cache gain from the fix |
| --- | --- | --- | --- | --- |
| 128 | 247 tok/s | 218 | 0.88× | **1.30×** |
| 256 | 247 | 221 | 0.89× | **1.31×** |
| 512 | 247 | 223 | 0.90× | **1.30×** |
| 1024 | 197 | 222 | **1.12×** | **1.38×** |

The naive column is unchanged (0.97–1.01× across the sweep), which is what makes this a
measurement rather than a coincidence: it is the untouched control, and it says the 30%
belongs to the cache path specifically and not to machine state.

**What the corrected numbers actually say.** The original explanation was directionally
right and quantitatively wrong. Decode at this scale really is dominated by per-step
overhead — 12 layers of kernel launches against a 124M model leave the 4090 mostly idle —
which is why cache throughput stays flat while naive throughput decays with length. The
crossover is real but late: the cache only overtakes recomputation at **1024 tokens**,
which is exactly `block_size` for this model. So for this model at its full context, the
cache wins by 12%, and below that it loses. Both of those are worth knowing, and neither
is the "caching is obviously faster" that the optimisation is usually sold with.

The honest correction is that I wrote "a real result rather than a bug" about a number
that was both. The reasoning was plausible enough that it stopped me looking, which is the
actual lesson: **a plausible explanation for a disappointing measurement is the most
expensive kind of mistake**, because it converts a bug into a finding and closes the
investigation. What broke it open was the flat-throughput column — an explanation that
fits the headline number but not the shape of the data is not yet an explanation.

Where the cache is unambiguously worth it is batching, because there the cache is what
makes a batch affordable at all:

| Variant | tokens/sec | time to first token | KV cache |
| --- | --- | --- | --- |
| naive (no cache), batch 1 | 256 | 4.4 ms | — |
| kv-cache, batch 1 | 219 | 6.4 ms | 13.5 MiB |
| kv-cache, batch 4 | 917 | 5.6 ms | 54 MiB |
| kv-cache, batch 16 | **3,622** | 8.5 ms | 216 MiB |

**16.5× throughput from batch 1 to 16** for 216 MiB of cache. The batch-1 `ttft`
includes allocating the cache, which is why it reads worse than the naive path.

Two tests pin the fix, both mutation-checked — reverting it fails the first, and
broadening the shortcut to every `q_len` fails the second:

- `test_decode_step_passes_no_attn_mask` records what actually reaches SDPA and asserts
  no mask arrives on a single-token step. This is the test the repository was missing:
  every existing test asserted the cache produced the *right answer*, and none asserted it
  took the *fast path*, so a 30% regression was invisible to a green suite.
- `test_multi_token_verify_step_still_masks` guards the other direction. Prefill against
  a partly-filled cache and speculative verification both have `q_len > 1` and genuinely
  need the alignment. The pre-existing prefill test compared only the *last* logit, which
  a wrong mask would not disturb; the new one compares every interior position, where it
  would.

---

## Quantization

Weight-only, asymmetric, per-group affine: `w ≈ (q − z) · s`, with a scale and zero point
per group of consecutive input features. Activations stay in bf16. 4-bit codes are packed
two per byte, because otherwise "4-bit" still occupies a byte and saves nothing.

| Scheme | Memory | vs fp32 | Perplexity | Δ ppl | HellaSwag | Decode |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 baseline | 475 MiB | 1.00× | **19.091** | — | 0.3480 | 191.8 tok/s |
| int8 per-tensor | 232 MiB | 2.04× | 19.106 | +0.015 | 0.3470 | 142.5 (0.74×) |
| **int8 g128** | 237 MiB | 2.00× | **19.103** | **+0.013** | 0.3460 | 139.8 (0.73×) |
| int4 per-tensor | 192 MiB | 2.47× | 22.664 | **+3.574** | 0.3350 | 94.4 (0.49×) |
| **int4 g128** | 196 MiB | 2.42× | **20.442** | **+1.351** | 0.3400 | 94.1 (0.49×) |
| int4 g32 | 212 MiB | 2.24× | 20.297 | +1.206 | 0.3360 | 91.0 (0.47×) |

Perplexity over 200,000 tokens of held-out English. The HellaSwag column is from the
earlier MPS run over 1,000 examples — it is device-independent, and the section below
explains why it was not worth re-measuring.

### 8-bit is free, 4-bit is not

int8 costs **+0.013 perplexity** — seven parts in ten thousand. At that magnitude it is
indistinguishable from no change, and per-tensor versus per-group scaling makes no
difference either, because 256 levels are enough to represent a weight distribution
without help.

4-bit is a real trade: **+1.351 perplexity** at group 128, about 7% worse. Whether that is
acceptable is a deployment question, not a technical one, but it is not free and should
not be presented as such.

### Grouping is worth 2.2 perplexity points at 4 bits

The per-tensor 4-bit row exists as a control, and it earns its place: **22.664 against
20.442**, a 2.2-point penalty from nothing more than sharing one scale across a whole
matrix instead of one per 128 features.

The mechanism is outliers. A single large weight sets the scale for everything it shares
with, and at 4 bits there are only 16 levels to begin with — so every ordinary weight in
that matrix collapses onto two or three of them. Grouping confines the damage to the 128
features that actually contain the outlier. `tests/test_quant.py` isolates this directly:
with one weight set to 10.0 among values of ~0.01, the error *outside* the outlier's group
is more than 5× smaller with grouping than without.

Going finer still (g32) buys only 0.15 more perplexity and costs 16 MiB in extra scales —
it is past the point of diminishing returns. **g128 is the right default**, and that is not
a guess, it is where the measured curve flattens.

### Why HellaSwag could not answer this

The HellaSwag column is nearly flat: 0.3480 down to 0.3350, and the ordering is not even
monotonic in bit-width. That is not evidence that quantization is harmless — it is
evidence that the metric is too blunt for the question.

A 4-way accuracy over 1,000 examples has a standard error of **1.5 points**. Every delta in
that column is inside one standard error, and resolving a 0.5-point difference at 95%
confidence would need roughly **69,000 examples** — HellaSwag only has 10,042. Perplexity
is continuous and computed over every token, so 200,000 tokens resolve differences two
orders of magnitude smaller for a fraction of the compute.

The lesson generalises: a benchmark good enough to *validate a model* is not automatically
good enough to *compare two versions of it*. It is also why the CUDA re-run skipped
HellaSwag entirely — spending GPU minutes to re-measure a column that cannot resolve the
effect would have been buying precision in the wrong place.

### The memory ceiling is the embedding, and it is architectural

4-bit reaches 2.42×, not the ~8× the bit-width implies. The reason is that **the token
embedding is 33% of this model** — 147 MiB of the 471 MiB of weights — and it is left in
fp32.

That is not a lazy default. With `tie_embeddings: true`, `lm_head.weight` *is*
`tok_emb.weight`. Replacing the head with a quantized layer stores a quantized copy while
`nn.Embedding` keeps the original tensor, so the model gets **larger**: measured at 196 MiB
with the head skipped versus 217 MiB with it "quantized". `quantize_model` now refuses that
configuration rather than reporting a compression ratio worse than doing nothing.

Getting past this ceiling needs a `QuantEmbedding` sharing one set of codes with the head —
not implemented. Two things worth noting about the scope of the problem: at 7B the
embedding is a few percent of the model rather than a third, so this ceiling is a
small-model artefact; and against bf16, which is what you would actually serve, 4-bit
blocks plus an fp16 embedding is only **2.02×**.

### Every quantized scheme is slower

−27% at int8, −51% at int4. This is expected and worth stating plainly rather than
omitting the column.

`QuantLinear` dequantizes into an fp32 weight and calls `F.linear`. So the bytes read from
memory go **up**, not down — the packed codes are read *and* a full-size dequantized copy is
materialised. The memory saving is in what is *stored*; the speed saving would be in what
is *moved*, and only a fused kernel that dequantizes inside the matmul's inner loop
achieves that. That is what Marlin, GPTQ's CUDA kernels and bitsandbytes provide, and it is
the natural home for the Triton kernel this repo does not yet have.

The dequantized weight is deliberately not cached, because caching it would make this fast
and pointless — an fp32 copy alongside the codes costs more than quantization saves.

CUDA is markedly kinder here than MPS was, and the gap is informative: the same code lost
74–85% of throughput on MPS versus 27–51% on the 4090. Dequantize-then-matmul is pure
extra work on any device, but how much it costs depends on how much spare memory bandwidth
the device has to absorb it.

---

## Speculative decoding

A cheap drafter proposes `k` tokens; the target scores all `k+1` positions in one forward
pass; proposals are accepted up to the first disagreement, and the target's own token is
taken at the mismatch. Two drafters are implemented: a smaller model of the same
vocabulary, and prompt-lookup, which copies from the context itself and costs nothing.

### It is lossless, and that is verified

**All 18 benchmark runs reproduced greedy decoding token-for-token** — three prompts, two
drafters, three values of `k`. The unit tests assert the same property across `k` ∈
{1, 2, 4, 8} and four drafters including deliberately adversarial ones.

This is the property that makes speculation an optimisation rather than an approximation.
An implementation that is merely *close* is not a faster decoder; it is a different, worse
model.

Two invariants hold it up. Accepting only exact argmax matches means the output cannot
drift. Appending the target's own token at the rejection point means even a drafter that is
wrong every single time still produces one real token per iteration — the worst case is
ordinary decoding plus the drafter's cost, never a stall.
`test_a_useless_drafter_still_makes_progress` pins exactly that.

### Results

Greedy baselines: prose 219, repetitive 226, code-ish 223 tok/s.

| Prompt | Drafter | k | Speedup | Acceptance | Tokens/target forward |
| --- | --- | --- | --- | --- | --- |
| code-ish | **prompt-lookup** | **8** | **5.35×** | 97.4% | 7.53 |
| prose | **prompt-lookup** | **8** | **2.73×** | 66.7% | 3.28 |
| repetitive | prompt-lookup | 4 | 2.39× | 100% | 2.98 |
| repetitive | prompt-lookup | 8 | 2.38× | 66.2% | 2.98 |
| prose | prompt-lookup | 4 | 2.32× | 75.9% | 2.72 |
| code-ish | prompt-lookup | 2 | 1.77× | 89.3% | 2.37 |
| repetitive | prompt-lookup | 2 | 1.54× | 75.6% | 2.00 |
| code-ish | model-draft | 8 | 1.08× | 95.8% | **8.53** |
| repetitive | model-draft | 2 | 0.94× | 98.8% | 2.98 |
| prose | model-draft | 8 | 0.46× | 36.1% | 3.76 |

### The most instructive row is still the failure

`model-draft` at `k=8` on code-like text achieved **95.8% acceptance and 8.53 tokens per
target forward pass** — essentially the algorithmic ideal, eight tokens out of one pass of
the big model — and returned **1.08×**. It spent the entire theoretical win on overhead and
came out level.

Two reasons, and both are the point:

1. **The draft model is the same size as the target.** It is the 10% training milestone of
   the same 124M architecture, chosen because it shares the tokenizer. Eight drafter
   forwards to save eight target forwards is not a trade, it is a wash plus overhead.
2. **The drafter runs uncached**, so proposing `k` tokens costs `k` full-prefix passes.

So acceptance rate and speedup are genuinely different questions, and this table is the
clearest way to see it. The `tokens/target forward` column is the ceiling the algorithm
reaches; the speedup column is what the hardware pays out. **A drafter has to be cheap
first and accurate second.**

### Prompt-lookup wins where the context repeats itself

5.35× on code-like text and 2.38–2.39× on repetitive text. Copying from the context only
works where the context predicts itself: lists, quotations, code, boilerplate. It is also
the best speedup-per-effort available, because the drafter is free — no second model to
train, ship or hold in memory.

The prose result is the one that changed. On MPS, prompt-lookup *lost* on prose (0.76× at
`k=4`, 0.47× at `k=8`); on CUDA it wins (2.32× and 2.73×). Same algorithm, same prompt,
opposite conclusion — so the earlier claim that lookup "loses on free-form writing" was a
statement about MPS, not about the method. The mechanism is that a rejected draft wastes
target compute, and whether that waste is affordable depends on how much of the target
forward pass is fixed overhead you were paying anyway. On a 4090 with a 124M model, a batch
of `k+1` positions costs barely more than one, so even 67% acceptance pays. Two devices
were enough to turn a confident conclusion into a conditional one, which is a reasonable
argument for not reporting single-device throughput as a property of an algorithm.

### `k` is not monotonic

`k=8` beats `k=4` on code-like text (5.35× vs 1.73×) and the two are a coin-flip on
repetitive text (2.38× vs 2.39×). Larger `k` multiplies both the win when acceptance is
high and the waste when it is not, so the right `k` depends on the text. An adaptive `k` —
grow it while proposals are accepted, shrink it after rejections — is the obvious next step
and is not implemented.

### The two cache fixes, and why the second lowered these ratios

The verify pass originally re-ran the whole prefix every iteration with no KV cache at all.
Since the cache is preallocated, rejecting a draft is only a move of the write offset —
`KVCache.rewind_to` — rather than a reallocation, so there was no good reason for the
omission. On MPS that was worth 25–30% (code-ish `k=8`: 2.37× → 3.00×).

The mask fix at the top of this document then *reduced* every speculative ratio here —
code-ish `k=8` fell from **7.14× to 5.35×**. That is not a regression, and the direction is
the interesting part: the fix sped up ordinary greedy
decoding by 27% (176 → 223 tok/s) because greedy decoding is single-token decode, which is
precisely the shape that was unmasked. Speculation gained almost nothing, because its
verify step has `q_len = k+1 > 1` and still needs a real mask, so it still runs unfused.

So the pre-fix speculative speedups were partly measuring against a handicapped baseline.
The absolute throughput barely moved (1,259 → 1,194 tok/s on the best row); the denominator
got faster. **A ratio improves when you break the thing you are dividing by**, which is an
argument for reporting absolute numbers next to every speedup.

It also identifies the remaining opportunity: speculative verification is stuck on the math
backend because it needs a bottom-right aligned mask, so it forfeits exactly the kernel the
decode path just got back. Recovering it needs a masking approach flash tolerates —
FlexAttention, or a kernel that takes the alignment as a parameter rather than a tensor.
Not implemented, and it is the largest single item left in this document.

---

## Where these numbers come from

**Memory and quality are device-independent.** Byte counts, perplexity and HellaSwag
accuracy are the same arithmetic on any hardware, so measuring them on MPS costs nothing in
validity, and the quality columns above are unchanged from the local run.

**Throughput is not**, and this document is now the evidence for that. Every `tok/s` and
speedup figure here was measured on one rented RTX 4090 in a single session, so contention
and thermal state are controlled, and the naive-decode column serves as a within-run
control. But the prose speculative result flipped sign between MPS and CUDA, so single-
device throughput should be read as a measurement of a device-algorithm pair, not of an
algorithm.

Provenance is recorded in every results file: GPU, arch, driver, torch build, CUDA version,
measured bf16 TFLOP/s, and the git commit. The figures here come from commit `42ed0a6` on
an RTX 4090 (sm_89, 23.5 GiB, driver 580.173.02, torch 2.4.1+cu124, 168.1 measured
TFLOP/s), except the HellaSwag column, which is from the earlier MPS run.

One caveat retained from the earlier version: the *first* MPS speculative benchmark ran
concurrently with the quantization sweep on the same device, so its absolute throughputs
were depressed by contention. Only same-run ratios were ever compared from that run, and it
no longer supplies any number in this document except by way of the MPS-versus-CUDA
contrasts, which are labelled as such.

Reproduce the whole thing on a rented GPU with:

```bash
./scripts/gpu.sh preflight && ./scripts/gpu.sh setup
./scripts/gpu.sh autostop 10
./scripts/gpu.sh bench && ./scripts/gpu.sh watch
```

Roughly 5 minutes of GPU time and about $0.06 at 4090 prices — the benchmark path
deliberately skips corpus preparation, which is what made the earlier full pipeline cost 16
minutes before it touched the GPU.

## What is not built

- **A fused dequantize-matmul kernel.** The single change that would turn quantization from
  a memory optimisation into a speed one. The obvious Triton exercise.
- **A flash-compatible mask for speculative verification.** Now the largest item: the
  verify step is the one remaining shape running on the math backend, and the decode-path
  fix showed what recovering the fused kernel is worth.
- **`QuantEmbedding`.** Needed to get past the 2.42× ceiling on a tied-embedding model.
- **Adaptive `k`.** The non-monotonic results above are the argument for it.
- **A cached drafter, and a genuinely small draft model.** The model-draft rows measure a
  handicapped configuration; a 6M-parameter draft with its own cache is the setup that would
  actually win.
- **Batched speculation.** Ragged accept lengths per row make it a different algorithm;
  `speculative_generate` refuses batches rather than silently decoding only row 0.
- **A cache sweep past 1024 tokens.** The crossover lands exactly at this model's
  `block_size`, so the most interesting part of that curve is out of reach without training
  a longer-context model.
