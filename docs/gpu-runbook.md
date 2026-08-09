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

### Ablation sweep — 15 runs, 7.6e18 FLOPs

| GPU | $/hr | est. MFU | est. hours | est. cost |
| --- | --- | --- | --- | --- |
| **RTX 4090** | 0.74 | 35% | 36.7 | **$27** |
| RTX 5090 | 0.99 | 33% | 30.6 | $30 |
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

---

## Before anything else: the network volume

**Attach a RunPod network volume mounted at `/workspace`.** The container filesystem
is destroyed when the pod is terminated, and the prepared corpus is ~20GB that takes
hours of CPU to tokenise. Losing it means paying for that twice.

`gpu.sh preflight` checks this explicitly and warns if `/workspace` is not a separate
mount. Do not skip it.

Size it for roughly 60GB: ~20GB corpus, plus checkpoints at 1.4GB each.

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

**5. Run the job.**

```bash
./scripts/gpu.sh sweep     # ablation sweep
```

```bash
./scripts/gpu.sh repro     # 124M reproduction
```

Both launch inside tmux, so closing the laptop does not kill them.

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
