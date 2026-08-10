# Inference efficiency: quantization and speculative decoding

Two optimisations, both hand-implemented, both measured against the 124M reproduction.
The theme running through the results is that **the headline number and the useful
number are rarely the same one**: 4-bit quantization is a memory win that costs speed,
and speculative decoding can hit its algorithmic ceiling while still losing on the
clock.

Throughput figures below are on Apple MPS and are directional only — see
[Where these numbers come from](#where-these-numbers-come-from).

---

## Quantization

Weight-only, asymmetric, per-group affine: `w ≈ (q − z) · s`, with a scale and zero
point per group of consecutive input features. Activations stay in bf16. 4-bit codes
are packed two per byte, because otherwise "4-bit" still occupies a byte and saves
nothing.

| Scheme | Memory | vs fp32 | Perplexity | Δ ppl | HellaSwag | Decode |
| --- | --- | --- | --- | --- | --- | --- |
| fp32 baseline | 475 MiB | 1.00× | **19.083** | — | 0.3480 | 154.3 tok/s |
| int8 per-tensor | 232 MiB | 2.04× | 19.095 | +0.012 | 0.3470 | 28.7 |
| **int8 g128** | 237 MiB | 2.00× | **19.089** | **+0.007** | 0.3460 | 28.1 |
| int4 per-tensor | 192 MiB | 2.47× | 22.648 | **+3.566** | 0.3350 | 22.4 |
| **int4 g128** | 196 MiB | 2.42× | **20.431** | **+1.348** | 0.3400 | 22.3 |
| int4 g32 | 212 MiB | 2.24× | 20.286 | +1.204 | 0.3360 | 23.2 |

Perplexity over 200,000 tokens of held-out English; HellaSwag over 1,000 examples.

### 8-bit is free, 4-bit is not

int8 costs **+0.007 perplexity** — three parts in ten thousand. At that magnitude it is
indistinguishable from no change, and per-tensor versus per-group scaling makes no
difference either, because 256 levels are enough to represent a weight distribution
without help.

4-bit is a real trade: **+1.348 perplexity** at group 128, about 7% worse. Whether that
is acceptable is a deployment question, not a technical one, but it is not free and
should not be presented as such.

### Grouping is worth 2.2 perplexity points at 4 bits

The per-tensor 4-bit row exists as a control, and it earns its place: **22.648 against
20.431**, a 2.2-point penalty from nothing more than sharing one scale across a whole
matrix instead of one per 128 features.

The mechanism is outliers. A single large weight sets the scale for everything it
shares with, and at 4 bits there are only 16 levels to begin with — so every ordinary
weight in that matrix collapses onto two or three of them. Grouping confines the damage
to the 128 features that actually contain the outlier. `tests/test_quant.py` isolates
this directly: with one weight set to 10.0 among values of ~0.01, the error *outside*
the outlier's group is more than 5× smaller with grouping than without.

Going finer still (g32) buys only 0.14 more perplexity and costs 16 MiB in extra
scales — it is past the point of diminishing returns. **g128 is the right default**, and
that is not a guess, it is where the measured curve flattens.

### Why HellaSwag could not answer this

The HellaSwag column is nearly flat: 0.3480 down to 0.3350, and the ordering is not even
monotonic in bit-width. That is not evidence that quantization is harmless — it is
evidence that the metric is too blunt for the question.

A 4-way accuracy over 1,000 examples has a standard error of **1.5 points**. Every delta
in that column is inside one standard error, and resolving a 0.5-point difference at 95%
confidence would need roughly **69,000 examples** — HellaSwag only has 10,042. Perplexity
is continuous and computed over every token, so 200,000 tokens resolve differences two
orders of magnitude smaller for a fraction of the compute.

The lesson generalises: a benchmark good enough to *validate a model* is not
automatically good enough to *compare two versions of it*.

### The memory ceiling is the embedding, and it is architectural

4-bit reaches 2.42×, not the ~8× the bit-width implies. The reason is that **the token
embedding is 33% of this model** — 147 MiB of the 471 MiB of weights — and it is left in
fp32.

That is not a lazy default. With `tie_embeddings: true`, `lm_head.weight` *is*
`tok_emb.weight`. Replacing the head with a quantized layer stores a quantized copy
while `nn.Embedding` keeps the original tensor, so the model gets **larger**: measured at
196 MiB with the head skipped versus 217 MiB with it "quantized". `quantize_model` now
refuses that configuration rather than reporting a compression ratio worse than doing
nothing.

Getting past this ceiling needs a `QuantEmbedding` sharing one set of codes with the
head — not implemented. Two things worth noting about the scope of the problem: at 7B
the embedding is a few percent of the model rather than a third, so this ceiling is a
small-model artefact; and against bf16, which is what you would actually serve, 4-bit
blocks plus an fp16 embedding is only **2.02×**.

### Every quantized scheme is slower

−74% to −85% decode throughput. This is expected and worth stating plainly rather than
omitting the column.

`QuantLinear` dequantizes into an fp32 weight and calls `F.linear`. So the bytes read
from memory go **up**, not down — the packed codes are read *and* a full-size dequantized
copy is materialised. The memory saving is in what is *stored*; the speed saving would be
in what is *moved*, and only a fused kernel that dequantizes inside the matmul's inner
loop achieves that. That is what Marlin, GPTQ's CUDA kernels and bitsandbytes provide,
and it is the natural home for the Triton kernel this repo does not yet have.

The dequantized weight is deliberately not cached, because caching it would make this
fast and pointless — an fp32 copy alongside the codes costs more than quantization saves.

---

## Speculative decoding

A cheap drafter proposes `k` tokens; the target scores all `k+1` positions in one
forward pass; proposals are accepted up to the first disagreement, and the target's own
token is taken at the mismatch. Two drafters are implemented: a smaller model of the
same vocabulary, and prompt-lookup, which copies from the context itself and costs
nothing.

### It is lossless, and that is verified

**All 18 benchmark runs reproduced greedy decoding token-for-token** — three prompts,
two drafters, three values of `k`. The unit tests assert the same property across `k` ∈
{1, 2, 4, 8} and four drafters including deliberately adversarial ones.

This is the property that makes speculation an optimisation rather than an
approximation. An implementation that is merely *close* is not a faster decoder; it is a
different, worse model.

Two invariants hold it up. Accepting only exact argmax matches means the output cannot
drift. Appending the target's own token at the rejection point means even a drafter that
is wrong every single time still produces one real token per iteration — the worst case
is ordinary decoding plus the drafter's cost, never a stall. `test_a_useless_drafter_still_makes_progress`
pins exactly that.

### Results

| Prompt | Drafter | k | Speedup | Acceptance | Tokens/target forward |
| --- | --- | --- | --- | --- | --- |
| code-ish | **prompt-lookup** | **8** | **3.00×** | 96.5% | 6.40 |
| repetitive | **prompt-lookup** | **4** | **1.59×** | 100% | 3.05 |
| code-ish | prompt-lookup | 4 | 1.18× | 50.7% | 2.21 |
| code-ish | prompt-lookup | 2 | 1.15× | 83.7% | 2.21 |
| repetitive | prompt-lookup | 2 | 0.99× | 76.7% | 2.00 |
| prose | prompt-lookup | 4 | 0.76× | 53.6% | 1.88 |
| prose | prompt-lookup | 8 | 0.47× | 42.3% | 2.00 |
| code-ish | model-draft | 8 | 0.64× | **100%** | **8.00** |
| repetitive | model-draft | 4 | 0.44× | 59.2% | 3.37 |
| prose | model-draft | 2 | 0.23× | 55.7% | 2.06 |

### The most instructive row is a failure

`model-draft` at `k=8` on code-like text achieved **100% acceptance and 8.00 tokens per
target forward pass** — the algorithmic ideal, every proposal correct, eight tokens from
one pass of the big model — and still ran at **0.64×**.

Two reasons, and both are the point:

1. **The draft model is the same size as the target.** It is the 10% training milestone
   of the same 124M architecture, chosen because it shares the tokenizer. Eight drafter
   forwards to save eight target forwards is not a trade, it is a wash plus overhead.
2. **The drafter runs uncached**, so proposing `k` tokens costs `k` full-prefix passes.

So acceptance rate and speedup are genuinely different questions, and this table is the
clearest way to see it. The `tokens/target forward` column is the ceiling the algorithm
reaches; the speedup column is what the hardware pays out. **A drafter has to be cheap
first and accurate second.**

### Prompt-lookup wins exactly where it should

3.00× on code-like text, 1.59× on repetitive text, and **0.76× on prose** — it loses on
free-form writing. That is not a defect; it is the mechanism. Copying from the context
only works where the context predicts itself: lists, quotations, code, boilerplate. On
prose there is no earlier match, so the proposals are wrong, and the wasted target
compute shows up as a slowdown.

It is also the best speedup-per-effort available, because the drafter is free — no
second model to train, ship or hold in memory.

### `k` is not monotonic

`k=8` beats `k=4` on code-like text (3.00× vs 1.18×) and loses badly on prose (0.47× vs
0.76×). Larger `k` multiplies both the win when acceptance is high and the waste when it
is not, so the right `k` depends on the text. An adaptive `k` — grow it while proposals
are accepted, shrink it after rejections — is the obvious next step and is not
implemented.

### Fixing the cache was worth 25–30%

The verify pass originally re-ran the whole prefix every iteration with no KV cache.
Since the cache is preallocated, rejecting a draft is only a move of the write offset —
`KVCache.rewind_to` — rather than a reallocation, so there was no good reason for the
omission.

Measured as within-run speedup ratios, which are comparable even though the absolute
throughputs were taken under different GPU load:

| | uncached | cached |
| --- | --- | --- |
| code-ish, lookup k=8 | 2.37× | **3.00×** |
| repetitive, lookup k=4 | 1.23× | **1.59×** |
| prose, lookup k=4 | 0.66× | **0.76×** |

---

## Where these numbers come from

**Memory and quality are device-independent.** Byte counts, perplexity and HellaSwag
accuracy are the same arithmetic on any hardware, so measuring them locally on Apple
MPS costs nothing in validity.

**Throughput is not.** Every `tok/s` and speedup figure here was measured on MPS, and
should be read as directional. The *direction* of each finding is hardware-independent —
dequantize-then-matmul adds work on any device, and a same-sized drafter cannot pay for
itself anywhere — but the magnitudes will differ on a CUDA GPU with different memory
bandwidth and kernel behaviour. A short run on a rented 4090 is the next step.

One measurement caveat: the first speculative benchmark ran concurrently with the
quantization sweep on the same GPU, so its absolute throughputs were depressed by
contention. Only same-run ratios are compared above.

## What is not built

- **A fused dequantize-matmul kernel.** The single change that would turn quantization
  from a memory optimisation into a speed one. The obvious Triton exercise.
- **`QuantEmbedding`.** Needed to get past the 2.42× ceiling on a tied-embedding model.
- **Adaptive `k`.** The non-monotonic results above are the argument for it.
- **A cached drafter, and a genuinely small draft model.** The model-draft rows measure
  a handicapped configuration; a 6M-parameter draft with its own cache is the setup that
  would actually win.
- **Batched speculation.** Ragged accept lengths per row make it a different algorithm;
  `speculative_generate` refuses batches rather than silently decoding only row 0.
