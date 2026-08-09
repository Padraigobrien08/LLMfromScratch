# Reproduction protocol: GPT-2 124M on FineWeb-Edu

The purpose of this run is to produce a number that someone else can check. A model
that generates fluent-looking text proves very little; a model that lands on a
documented validation loss, under a protocol stated in advance, proves the training
pipeline is correct end to end.

**Status: not yet run.** This document fixes the protocol *before* the run, so the
tolerance cannot be chosen after seeing the result.

---

## Target

| | |
| --- | --- |
| Model | 124M-parameter decoder-only transformer, GPT-2 small architecture |
| Corpus | FineWeb-Edu `sample-10BT` |
| Tokenizer | GPT-2 BPE, 50,257 tokens (padded to 50,304) |
| Budget | 1 epoch ≈ 10B tokens = 19,073 steps × 524,288 tokens |
| Target metric | Validation cross-entropy on the held-out FineWeb-Edu shard |
| **Target value** | **≤ 3.29** |
| Tolerance | Within +0.02 of target counts as reproduced; above +0.05 is a failure to be investigated and reported, not quietly retried |

### Where the target comes from

The reference point is Karpathy's `build-nanogpt`, which trains this architecture on
this corpus for this budget and reports the OpenAI GPT-2 124M checkpoint scoring
≈3.2924 on the same held-out FineWeb-Edu split, with the from-scratch run reaching
slightly below it after one epoch.

Two caveats that matter for honesty:

1. **The number is corpus-specific.** Validation loss is per-token and depends on the
   tokenizer and the evaluation set. 3.29 on FineWeb-Edu is not comparable to a loss
   on OpenWebText, WikiText or anything else. Both the tokenizer and the split are
   pinned here for exactly that reason.
2. **This target should be re-confirmed against the source before the run is
   reported**, rather than taken from this document. It is recorded here as the
   pre-registered target; its provenance is a secondary source, not a measurement of
   ours.

A second, harder check is planned alongside it: zero-shot HellaSwag accuracy, where
GPT-2 124M scores ≈0.2955. Loss can be gamed by a tokenizer or evaluation-set
mismatch; a downstream task is much less forgiving.

---

## Protocol

Fixed in advance, and not to be adjusted after seeing the result:

- **One run, one seed** (1337). If the target is missed, the failure is reported
  along with the diagnosis. Re-rolling seeds until one clears the bar is how a
  reproduction becomes a lottery.
- **No validation-set tuning.** Hyperparameters are GPT-2's published values, not
  ones searched against the metric being reported.
- **Evaluation over the entire validation shard**, not a 50-batch sample. The
  in-training estimate uses 50 batches for a cheap curve; the reported number comes
  from `llmfs-eval`, which sweeps the whole split.
- **Everything published**: config, `metrics.jsonl`, loss curve, wall-clock, hardware,
  total cost, and the final checkpoint.

## Configuration

Complete and version-controlled in [`configs/gpt2-124m.yaml`](../configs/gpt2-124m.yaml).
The parameters that matter:

| Setting | Value | Why |
| --- | --- | --- |
| Optimiser | AdamW, β = (0.9, 0.95), ε = 1e-8 | GPT-2 |
| Peak LR | 6e-4 | GPT-2 at a 0.5M-token batch |
| Schedule | Cosine to 10% of peak, 700-step warmup | GPT-2 |
| Weight decay | 0.1, on matmul weights only | Decaying norm gains and biases penalises the parameters whose job is to set a scale |
| Grad clip | 1.0 | |
| Batch | 524,288 tokens/step | GPT-2's 0.5M, reached by gradient accumulation so the value is hardware-independent |
| Dropout | 0.0 | One epoch over 10B tokens — no repetition to regularise against |
| Precision | bf16 autocast | Matches fp32's exponent range, so no loss scaler and no silent overflow divergence |

`vocab_size` is padded from 50,257 to 50,304 (a multiple of 64) for tensor-core
alignment. The 47 unused rows receive no gradient signal from targets and train
towards never being predicted; this is standard and costs ~0.04% of parameters.

---

## Running it

```bash
llmfs-prepare-data --source fineweb-edu --out-dir data/fineweb-edu-10B
```

```bash
llmfs-train --config gpt2-124m
```

On 8 GPUs — same config, same optimisation, gradient accumulation drops from 32 to 4
to hold the effective batch fixed:

```bash
torchrun --nproc_per_node=8 -m llmfs.train.cli --config gpt2-124m
```

Report the final number over the full validation split:

```bash
llmfs-eval --checkpoint out/gpt2-124m-repro/best.pt --out results/reproduction.json
```

### Interruptions

The run is resumable and expected to be resumed — spot instances get reclaimed.

```bash
llmfs-train --config gpt2-124m --resume auto
```

Data-loader position is *derived* from the step counter rather than stored, so a
resumed run cannot disagree with the original about which tokens belong to which
step. Checkpoints are written to a temporary file and renamed, so a process killed
mid-write leaves the previous checkpoint intact rather than a truncated one. Both
properties are covered by tests.

---

## Cost estimate

To be replaced with measured figures. The estimate below is what the run is being
budgeted against, and the gap between it and reality is itself worth reporting.

| | |
| --- | --- |
| Compute | 6 × 124e6 × 10e9 ≈ 7.4e18 FLOPs (forward+backward, 6N rule) |
| At 40% MFU on one A100-80GB (312 TFLOP/s bf16) | ≈ 16 GPU-hours |
| At ~$1.50/hr spot | ≈ $25 |

MFU is logged every `log_interval` steps from the first step, so the assumption
behind this estimate is visible while the run is in progress rather than after it.

---

## What gets published

- `results/reproduction.json` — final loss, perplexity, step, tokens evaluated
- `out/gpt2-124m-repro/metrics.jsonl` — the full training trace
- Loss curve, and the target drawn on the same axes
- Hardware, wall-clock, measured MFU, and total cost
- The final checkpoint
