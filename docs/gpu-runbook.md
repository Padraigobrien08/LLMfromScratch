# GPU runbook

How to take this repo from "runs on a laptop" to "produced the numbers", on a rented
pod, without wasting money.

Everything here goes through [`scripts/gpu.sh`](../scripts/gpu.sh), which drives an
SSH-reachable box. It is written against RunPod but assumes nothing beyond SSH, a
GPU, and tmux.

---

## Which GPU

**There is no A100 available**, so the earlier A100 estimates elsewhere in these docs
do not apply. The table below is computed from the model's own
`Transformer.flops_per_token` — not the `6N` rule of thumb — across the pods on offer.

The MFU column is an **estimate, not a measurement**, and it is where the uncertainty
lives. A 51M-parameter model at 512 context does not come close to saturating an
H100: the GPU spends much of its time waiting on memory and kernel launches, so the
big cards look far worse per dollar than their peak FLOPs suggest. The reproduction,
being 124M at 1024 context with a 0.5M-token batch, uses them much better. MFU is
logged from step one of every run — replace these numbers with the real ones after
the first hour and re-decide.

### Sweep sizing: seeds versus tokens

The sweep runs every arm at the same seeds so comparisons are paired (see
[the ablation section of the README](../README.md#ablation-study)). Three seeds is
the default. That triples the compute, and how it is paid for is a real choice:

| | runs | H100 | RTX 5090 |
| --- | --- | --- | --- |
| 13 arms × 1 seed, 1B tokens | 13 | 10 h / $34 | 27 h / $26 |
| **13 arms × 3 seeds, 1B tokens** (default) | 39 | **31 h / $102** | 80 h / $79 |
| 13 arms × 3 seeds, 500M tokens | 39 | 16 h / $51 | 40 h / $39 |

The last row halves the per-run token budget to pay for the extra seeds:

```bash
SWEEP_EXTRA="--set train.max_steps=2000"
```

That takes each arm from 20 tokens/parameter (Chinchilla-optimal) to 10
(undertrained). For *relative* comparisons between architectures this is usually the
better trade — replication is what makes an effect visible, and run length mostly is
not — but it is a trade, and the write-up should say which was used.

### Ablation sweep — per-run cost, 7.6e18 FLOPs for 15 runs

| GPU | $/hr | est. MFU | est. hours | est. cost |
| --- | --- | --- | --- | --- |
| **RTX 4090** | 0.74 | 35% | 36.7 | **$27** |
| **RTX 5090** | 0.99 | 33% | 30.6 | $30 |
| L40S | 0.99 | 32% | 36.6 | $36 |
| **H100 SXM** | 3.29 | 18% | **11.9** | $39 |
| H100 PCIe | 2.89 | 20% | 14.0 | $41 |
| B200 | 6.79 | 12% | 7.9 | $53 |

### Reproduction — 124M, 10B tokens, 1.09e19 FLOPs

| GPU | $/hr | est. MFU | est. hours | est. cost |
| --- | --- | --- | --- | --- |
| **H100 SXM** | 3.29 | 35% | **8.7** | **$29** |
| B200 | 6.79 | 30% | 4.5 | $30 |
| H100 PCIe | 2.89 | 32% | 12.5 | $36 |
| RTX 4090 | 0.74 | 30% | 61.0 | $45 |
| RTX 5090 | 0.99 | 30% | 47.9 | $47 |

### Recommendation

**H100 SXM for both, ~21 hours and ~$68 total.** It is not the cheapest way to run
the sweep — a 4090 saves about $12 — but it is three times faster there and it is
clearly the right choice for the reproduction, so one pod does everything and there
is one environment to get right instead of two.

Take the 4090 route only if wall-clock genuinely does not matter: 37 hours of sweep
plus 61 hours of reproduction is four days of babysitting a spot instance to save
roughly $25.

### RTX 5090 vs H100: the argument is wall-clock, not cost

At $0.99/hr against $3.29 the 5090 looks obviously cheaper. It is not, quite — but
the gap is smaller than a first pass suggests.

The planning figure for a 5090 is **~210 TFLOP/s dense bf16**. Consumer NVIDIA quotes
Tensor FP16/BF16 throughput *already assuming FP32 accumulation*, at roughly twice the
FP32 shader rate: the RTX 4090's 165 TFLOP/s comes from 82.6 FP32, and the 5090's
104.8 FP32 implies ~210. There is no further halving to apply for PyTorch's default
accumulation mode.

| | sweep | reproduction | total time | total cost |
| --- | --- | --- | --- | --- |
| RTX 5090 @ $0.99 | ~31 h | ~48 h | **78 h (3.3 days)** | ~$78 |
| H100 SXM @ $3.29 | ~12 h | ~9 h | **21 h** | ~$68 |

**The costs are within the error bars of these estimates; the wall-clock is not.**
Depending on how well a 51M-parameter model fills an H100, its total lands anywhere
between $57 and $99 — so "the H100 is cheaper" is not a claim worth defending. "The
H100 finishes in a day instead of three and a half" is.

Pick the H100 if you want results tomorrow. Pick the 5090 if the runs can sit in the
background for a long weekend and you would rather not think about the hourly rate.
Either is defensible; they are not far apart on money.

**Either way, check the measurement.** `gpu.sh setup` prints
`MEASURED bf16 matmul: N TFLOP/s`. If a 5090 reports far below ~200, or an H100 far
below ~800, something is wrong with the build or the card before any paid work
starts.

VRAM is not the constraint on either: 32GB comfortably holds the 124M reproduction at
`micro_batch_size: 16` (the logits tensor is the largest single allocation, at
1.5 GiB). If it does OOM, halve it to 8 — gradient accumulation doubles automatically
and the optimisation is unchanged.

---

## Pod configuration checklist

| Setting | Value | Why |
| --- | --- | --- |
| Template | **PyTorch 2.8+ with cu128** | Blackwell cards (RTX 5090 = sm_120) have no kernels in older builds. The bootstrap keeps the image's torch rather than replacing it, precisely so this stays matched. |
| **Network volume** | **required, ≥100GB at `/workspace`** | See below — this is the one that costs real money to get wrong. |
| Container disk | 50GB | Only holds the image and the venv; nothing durable. |
| SSH terminal access | **on** | Everything here runs over SSH. |
| Jupyter notebook | off | Nothing uses it. |
| Instance pricing | On-Demand | Interruptible is cheaper, and both jobs resume cleanly — but on-demand removes preemption from the list of things to think about. |

### The network volume is not optional

RunPod offers two kinds of persistent storage at `/workspace`, and they are easy to
confuse:

- **Volume disk** — persistent, but *"will be deleted when Pod is terminated"*.
- **Network volume** — independent of the pod's lifecycle, survives termination,
  and can be mounted by a later pod.

Only the second one protects the corpus. Preparing FineWeb-Edu is ~20GB written after
hours of CPU tokenisation; on a volume disk that work is destroyed the moment the pod
is terminated, and has to be paid for again. If the storage panel is still offering
**"+ Create a network volume"**, one has not been attached yet.

`gpu.sh preflight` checks whether `/workspace` is a real mount and says so plainly.

**Size it for at least 100GB**, not the 50GB that a default volume disk gives:

| | |
| --- | --- |
| FineWeb-Edu 10B tokens, uint16 | 20 GB exactly |
| Ablation sweep, 39 runs (13 arms × 3 seeds) | 45 GB at `keep_last_n: 0` |
| Reproduction checkpoints | 1.4 GB each |

Three seeds means **39 run directories, not 13**. At the default `keep_last_n: 2`
that is ~109 GB of checkpoints — more than the corpus and more than the volume. The
pipeline therefore runs the sweep with `keep_last_n=0`, which keeps no rolling
checkpoints; `best.pt` and `final.pt` are never pruned, so every arm stays
recoverable and its best model intact. That brings the sweep to ~45 GB, and the
total to ~65 GB.

---

## The sequence

```bash
cp .gpu.env.example .gpu.env    # fill in host/port from the RunPod Connect tab
```

**1. Verify the pod is what you think it is.**

```bash
./scripts/gpu.sh preflight
```

Reports the GPU, disk, and whether `/workspace` persists.

**2. Install.**

```bash
./scripts/gpu.sh setup
```

Clones the repo at `main` and installs with **CUDA** torch. The bootstrap then runs a
real bf16 matmul and prints measured TFLOP/s — this catches a broken driver/toolkit
pairing now, rather than at step 1 of a paid run. It aborts if torch cannot see the
GPU, because the failure mode otherwise is a job that runs to completion on the CPU,
slowly, while `nvidia-smi` still shows a healthy card.

**3. Prove the whole path in two minutes.**

```bash
./scripts/gpu.sh data smoke && ./scripts/gpu.sh smoke
```

Cheap insurance. Discovering a broken environment at hour six of a paid job is the
expensive version of this.

**4. Prepare the corpus.** Hours, detached.

```bash
./scripts/gpu.sh data ablation
```

This is CPU-bound tokenisation — it does not need the GPU, and paying H100 rates for
it is pure waste. If you are cost-sensitive, do this step on a cheap CPU pod attached
to the same network volume, then start the GPU pod.

**5. Run everything.**

```bash
./scripts/gpu.sh all
```

One detached pipeline: corpus → ablation sweep → reproduction → final evaluation over
the full validation split → attention explorer rebuilt from the real model. Each
stage is marker-guarded on the pod, so re-running `all` after a crash, a preemption
or a pod restart resumes at the first unfinished stage rather than redoing hours of
work. Within a stage recovery is finer still: the sweep skips completed arms and
training resumes from its last checkpoint.

A failed stage stops the pipeline rather than pressing on — otherwise the
reproduction would run against a corpus that failed to prepare, and bill for it.

To run just one piece:

```bash
./scripts/gpu.sh sweep     # ablation sweep only
```

```bash
./scripts/gpu.sh repro     # 124M reproduction only
```

Or skip a stage of the full pipeline with `RUN_SWEEP=0` / `RUN_REPRO=0` in `.gpu.env`.
Everything launches inside tmux, so closing the laptop does not kill it.

**6. Watch, then collect.**

```bash
./scripts/gpu.sh status    # progress, GPU utilisation, spend so far
```

```bash
./scripts/gpu.sh watch     # polls, then fetches and post-processes automatically
```

`watch` runs `fetch` when the job ends: results and metrics come back, the ablation
report and plots are regenerated locally, and everything lands in `results/`.

**7. Stop paying.**

```bash
./scripts/gpu.sh done
```

Then terminate the pod in the console. A *stopped* pod still bills for storage; a
*terminated* one does not, but takes the container filesystem with it.

---

## What comes back, and what does not

`fetch` pulls **only** metrics, configs and results — a few hundred KB. Those are
small, they are the artifacts every claim rests on, and they belong in git:

```bash
git add results && git commit -m "Add ablation results"
```

Checkpoints stay on the volume. At 1.4GB each they do not belong in a repository;
pull one deliberately when you want it:

```bash
./scripts/gpu.sh fetch-checkpoints
```

---

## When it goes wrong

**SSH dropped.** Nothing happened to the job — it is in tmux. `gpu.sh status`, or
`gpu.sh attach` to watch it live.

**Pod was preempted.** Both jobs resume. The sweep skips arms that already have a
result; the reproduction is launched with `--resume auto` and picks up from the last
checkpoint. Because loader position is derived from the step counter rather than
stored, a resumed run consumes exactly the tokens it would have. See
[fault-tolerance.md](fault-tolerance.md).

**An arm diverged.** Expected for `lr-3e-3`. It is recorded as a result and the sweep
continues.

**Throughput is far below the table.** Check `perf/mfu` in the logs. Most often the
micro-batch is too small to fill the GPU — raise `data.micro_batch_size` via
`SWEEP_EXTRA`; gradient accumulation adjusts automatically and the optimisation is
unchanged, so this is a free knob.

**`nvidia-smi` shows 0% utilisation.** The job is starved on data, not compute. On a
network volume this usually means the loader is reading over the network for every
batch; copy the corpus to the pod's local disk first.

---

## Checkpoint interval

The default is 1000 steps, which [fault-tolerance.md §3.2](fault-tolerance.md) shows
wastes ~16% of a single-GPU spot run. Until that is fixed properly, set it explicitly
for the hardware you are on:

```bash
REPRO_EXTRA="--set log.checkpoint_interval=100"
```

At ~4.6s/step on one A100-class card that is a checkpoint every ~8 minutes, close to
the Young/Daly optimum for a 4-hour MTBF.
