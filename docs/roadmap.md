# Roadmap: what is not built, and what it would take

Everything in the seven-pillar study is measured and written up. This file consolidates the
work that was scoped, understood, and deliberately not done, previously scattered across
"what is not built" sections in three documents, where it was easy to read as a disclaimer
rather than as a plan.

Each item says what it is, why it is worth doing, what to learn first, and how you would know
it worked. Ordered by value per unit of effort, not by ambition.

Nothing here is required by the study. The reproduction, ablations, efficiency benchmarks and
scaling report stand on their own.

---

## 1. A fused dequantize-matmul kernel

**The problem it solves.** Quantization currently makes inference *slower*: −27% at int8,
−51% at int4 ([docs/efficiency.md](efficiency.md)). `QuantLinear` dequantizes into a
full-size fp32 weight and calls `F.linear`, so bytes moved *increase*; the packed codes are
read *and* a full-size copy is materialised. The memory saving is in what is stored; the
speed saving would be in what is moved, and only a kernel that dequantizes inside the
matmul's inner loop achieves that.

This is the single change that would turn quantization from a memory optimisation into a
speed one, and it is the most-cited gap in the efficiency write-up.

**What to learn first.** Triton, and specifically its matmul tutorial: the tiled
block-pointer pattern where each program instance computes one output tile. The addition here
is that the B operand arrives as packed 4-bit codes plus per-group scales and zero points, so
the inner loop unpacks a tile before accumulating. Useful prior art: Marlin, GPTQ's CUDA
kernels, and bitsandbytes, all of which solve exactly this and are worth reading before
writing anything.

Concepts that matter: memory-bandwidth-bound versus compute-bound kernels (this one is
bandwidth-bound, which is *why* it can win); tile sizes and occupancy; and why fp32
accumulation is non-negotiable for numerical agreement with the reference.

**How you would know it worked.** The existing tests already define correctness: the
quantized model must produce the same perplexity as `QuantLinear` does today, to within
floating-point tolerance; `tests/test_quant.py` has the dequantization round-trip and the
outlier-grouping assertions. So the kernel is a drop-in replacement whose *output* is already
pinned, and the only new claim is speed. Success is `llmfs-quant-eval` reporting int4
decode throughput at or above the fp32 baseline instead of 49% of it.

**Cost.** Needs a CUDA GPU to develop against, not just to benchmark; a single cheap card is
enough, and an RTX 4090 at $0.74/hr was ample for the inference benchmarks. Budget days of
learning, not hours.

---

## 2. A flash-compatible mask for speculative verification

**The problem it solves.** This is now the largest single item in the efficiency write-up, and
it exists *because* of a fix. Removing the unnecessary mask from the single-token decode path
recovered the fused flash kernel and gained 30% ([docs/efficiency.md](efficiency.md)).
Speculative verification did not benefit: its query block is `k+1` tokens against a filled
cache, so it needs a genuine bottom-right aligned mask, and passing `attn_mask` to SDPA
forfeits the fused kernels. The verify step is the one remaining shape running on the math
backend.

**What to learn first.** FlexAttention (torch ≥ 2.5), which takes a *score-modification
function* instead of a mask tensor and compiles it into the kernel, exactly the shape of this
problem. The alternative is a kernel that takes the alignment offset as a scalar parameter
rather than materialising a mask.

**How you would know it worked.** `test_multi_token_verify_step_still_masks` already pins
correctness: interior queries of a multi-token block against a filled cache must match a full
forward pass, and a wrong mask fails it. Losslessness is separately pinned across 18
benchmark runs. So again, correctness is already guarded and the new claim is speed:
`llmfs-spec-bench` should show the model-draft rows improving, since those pay the verify cost
on every iteration.

Worth calibrating expectations first: the decode-path fix was worth 30% because *every* token
paid it. Verification happens once per accepted block, so the ceiling is lower.

---

## 3. `QuantEmbedding`

**The problem it solves.** 4-bit compression caps at 2.42×, not the ~8× the bit-width
implies, because the token embedding is 31% of this model (147 MiB of 475) and is left in
fp32. It cannot simply be quantized: with `tie_embeddings: true`, `lm_head.weight` *is*
`tok_emb.weight`, so replacing the head with a quantized layer stores a quantized copy while
`nn.Embedding` keeps the original, and the model gets **larger** (196 MiB → 217 MiB measured).
`quantize_model` refuses that configuration rather than reporting a compression ratio worse
than doing nothing.

Getting past the ceiling needs an embedding and a head sharing one set of codes.

**Worth knowing before starting:** this is a small-model artefact. At 7B the embedding is a
few percent of the model rather than a third, so the ceiling this lifts is one that mostly
does not exist at the scales anyone deploys. It is a good exercise and a poor priority.

**How you would know it worked.** Compression above 2.42× at 4 bits with perplexity unchanged
from the current int4 g128 figure of 20.442, and `quantize_model` no longer needing its tied
weight veto.

---

## 4. Adaptive `k` for speculative decoding

**The problem it solves.** `k` is not monotonic: `k=8` beats `k=4` on code-like text (5.35×
vs 1.73×) and the two are a coin-flip on repetitive text. Larger `k` multiplies both the win
when acceptance is high and the waste when it is not, so the right `k` depends on the text;
which means it should not be a constant. Grow `k` while proposals are accepted, shrink it
after rejections.

**Why it is cheap.** No kernel work, no new hardware knowledge. It is a control loop in
`speculative_generate`, and the benchmark to evaluate it already exists.

**How you would know it worked.** An adaptive policy should beat the *best fixed* `k` averaged
across the three prompt types, not beat the worst. The honest comparison is against an oracle
that picks the best fixed `k` per prompt; if adaptation cannot beat that, it is not earning
its complexity.

---

## 5. A genuinely small draft model, with its own cache

**The problem it solves.** The `model-draft` rows measure a deliberately handicapped
configuration: the drafter is the 10% training milestone of the *same* 124M architecture,
chosen because it shares the tokenizer. Eight drafter forwards to save eight target forwards
is a wash plus overhead, which is why 95.8% acceptance and 8.53 tokens per target forward
still only returned 1.08×. The drafter also runs uncached, so proposing `k` tokens costs `k`
full-prefix passes.

A ~6M-parameter draft model trained on the same tokenizer, with its own KV cache, is the setup
that would actually win.

**Cost.** Training a 6M model is cheap: minutes on one GPU, and the config system already
supports it. The work is mostly plumbing a second cache through `speculative_generate`.

**How you would know it worked.** `model-draft` beating `prompt-lookup` on *prose*, which is
where lookup fails because the context does not predict itself. That is the case a real draft
model exists for.

---

## 6. Batched speculation

**Why it is last.** Ragged accept lengths per row make it a different algorithm, not an
extension: each sequence in the batch accepts a different number of tokens, so the cache
positions diverge and the next iteration's query block is no longer rectangular.
`speculative_generate` refuses batches rather than silently decoding only row 0.

Worth doing only if serving throughput is the goal, in which case it matters a lot: batching
was the largest single inference win measured here, 16.5× from batch 1 to 16.

---

## 7. Multi-node scaling

**The gap it fills.** Every scaling number is single-node (`--nnodes=1`). Crossing hosts
introduces a network an order of magnitude slower than PCIe, and **none of the measured data
predicts that.** The accumulation sweep gives the tool to reason about it (communication cost
decomposes as `a + b/accum`, and a slower interconnect raises `b`), but `b` for Ethernet or
InfiniBand is unmeasured.

**What to learn first.** `torchrun` with `--nnodes` and a real rendezvous backend; NCCL over
network transports; and why gradient compression and overlapping become interesting when `b`
gets large enough.

**Cost.** The expensive item here: two multi-GPU pods with a fast interconnect between them,
which is a different product tier from anything rented so far.

---

## Not on this list, deliberately

**An NVLink comparison.** Measured out of relevance rather than skipped. PCIe achieves 95.1%
at the reproduction's batch, and the accumulation fit attributes only ~2.8 points to the
all-reduce itself, so a perfect interconnect could recover about three points. Two different
machines would confound interconnect with architecture, memory bandwidth and NCCL version to
chase that. [docs/scaling.md](scaling.md) has the reasoning.

**MFU on `sm_120`.** `peak_flops()` deliberately has no RTX 5090 entry. The commonly quoted
209.5 TFLOP/s dense bf16 figure is contradicted by measurement: an 8192³ bf16 matmul reached
234.7 TFLOP/s on one, and nothing exceeds its own peak. Adding a number that produces 96% MFU
would be worse than reporting none. If a vendor figure can be confirmed, the entry is one line.
(That 234.7 was printed on a pod and never captured to `results/`; `bootstrap.sh` now writes
the probe to an artifact so the next one is not lost the same way.)
