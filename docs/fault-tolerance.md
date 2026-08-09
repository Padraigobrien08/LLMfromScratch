# Fault tolerance for long training runs

## Scope

A 124M reproduction on one rented GPU is roughly a 24-hour job. That is long enough
that failure is a routine event rather than an exception: spot instances get
reclaimed, hosts reboot, disks fill, and occasionally a GPU computes the wrong answer
without telling anyone. This document sets out what the training system currently
guarantees, what it does not, and what would have to change to run the same job on
three orders of magnitude more hardware.

The bar it is written against: **any failure should cost bounded, quantifiable work,
and no failure should silently corrupt a result.** The second half is the harder one.
A crash is cheap — you notice it. A run that quietly trains on the wrong data, or on
a GPU with a bad SM, produces a loss curve that looks fine and a number that is wrong,
and you find out weeks later when it fails to reproduce.

Every claim below is marked **[implemented]**, with the test that pins it, or
**[proposed]**, with an estimate of the work. Nothing here is aspirational unless it
says so.

---

## 1. Failure taxonomy

Ordered by how often they actually happen on rented hardware.

| # | Failure | Symptom | Detection | Cost if unhandled |
| --- | --- | --- | --- | --- |
| 1 | Spot preemption | SIGTERM, then the box vanishes | Process exit | All work since the last checkpoint |
| 2 | OOM | CUDA OOM traceback | Process exit | Same, plus a config change |
| 3 | Host/GPU crash, Xid error | Process dies, or hangs forever | Exit code, or a watchdog timeout | Same, or **unbounded** if it hangs |
| 4 | Disk full | Checkpoint write fails | Exception at write time | Potentially every checkpoint, if the failure is silent |
| 5 | Straggler rank | Throughput drops, no error | Per-rank step-time telemetry | Runs to completion, just slowly and expensively |
| 6 | Loss divergence | Loss spikes to NaN/Inf | Loss and grad-norm checks | Everything since the divergence started |
| 7 | **Silent data corruption** | Nothing at all | Very hard | **The entire run, discovered later** |

Rows 1–4 are loud: the process stops and something restarts it. Rows 5–7 are the
interesting ones, because the run keeps going and keeps producing plausible-looking
numbers.

The ranking matters for where to spend effort. Rows 1–3 are solved by checkpointing
and correct resumption, which is mostly done. Row 7 is where a training system earns
its trust, and is the least addressed here.

---

## 2. What is already guaranteed

### 2.1 Resumption is exact, because the data loader has no state

This is the design decision the rest of the document leans on.

The obvious way to make a data loader resumable is to checkpoint its position
alongside the model. That creates a second source of truth which can disagree with
the first — the classic version of this bug is a checkpoint written at step *N* while
the loader had already advanced to *N+1*, so the resumed run silently re-trains on one
batch and skips another.

Instead, the loader stores nothing. The corpus is one long token stream and position
is a pure function of the step number:

```
position = step × grad_accum × world_size × micro_batch × block_size
```

[`ShardDataLoader.set_step`](../src/llmfs/data/loader.py) computes it directly. There
is no loader state in the checkpoint, so there is nothing that can go stale, and
resumption is idempotent by construction: resuming at step *N* twice consumes exactly
the same tokens both times.

**[implemented]** — `test_set_step_is_the_only_state_needed_to_resume` asserts a
fast-forwarded loader yields exactly what an uninterrupted one would have.

Two properties fall out of this for free:

- **World-size invariance.** Ranks read disjoint interleaved slices whose union is the
  contiguous span a single-GPU run would have consumed, so a run can be resumed on a
  different number of GPUs and still see the same tokens in the same order.
  (`test_distributed_step_matches_single_gpu_token_span`)
- **No data-induced load imbalance.** Every rank gets exactly `micro_batch × block_size`
  tokens per micro-step. There is no variable-length batching, so a straggler is
  always a hardware or network problem, never a data problem. That makes straggler
  diagnosis much simpler than it is in systems with dynamic batching.

### 2.2 Checkpoint writes are atomic

[`save_checkpoint`](../src/llmfs/train/checkpoint.py) writes to a temporary file and
`os.replace`s it into position. A process killed mid-write leaves the previous
checkpoint intact rather than a truncated file that fails to load — which is the
difference between losing one checkpoint interval and losing the whole run.

**[implemented]** — `test_checkpoint_write_is_atomic`.

**Known gap:** `os.replace` is atomic with respect to *visibility*, not *durability*.
On a host that loses power, the rename can be visible in the directory while the file
contents are still in the page cache and never reached the disk. A correct
implementation `fsync`s the file before the rename and `fsync`s the containing
directory after it. Currently it does neither. See §6.1.

### 2.3 Checkpoints are self-describing

The model architecture is reconstructed from the config recorded inside the
checkpoint, not from whatever config the resuming process happens to have
([`model_from_checkpoint`](../src/llmfs/train/checkpoint.py)). Loading a checkpoint
into a mismatched architecture fails loudly at `load_state_dict` rather than
half-succeeding.

Wrappers (`DistributedDataParallel`, `torch.compile`) are stripped before saving, so a
checkpoint is not tied to the topology or the torch version that produced it — an
8-GPU run's checkpoint loads into a 1-GPU process for evaluation without special
handling.

**[implemented]** — `test_checkpoint_round_trips_through_its_own_config`.

### 2.4 Tokenizer and vocabulary mismatches fail at startup

Training on data prepared with a different tokenizer does not crash — it converges
badly, and looks like a hyperparameter problem for a day. The trainer compares the
data directory's `meta.json` against the config and refuses to start on a mismatch,
and likewise if the tokenizer's vocabulary exceeds the model's.

**[implemented]** — `test_data_tokenizer_mismatch_is_caught`,
`test_vocab_too_small_for_the_data_is_caught`.

---

## 3. Checkpointing strategy

### 3.1 What goes in

| Contents | Size (124M model) |
| --- | --- |
| Model weights, fp32 | 475 MiB |
| AdamW first moment | 475 MiB |
| AdamW second moment | 475 MiB |
| Step counter, config, metrics | negligible |
| **Total** | **1.39 GiB** |

Optimiser state is two-thirds of it and is not optional. Resuming from weights alone
discards Adam's moment estimates, which restarts the optimiser cold: the effective
learning rate is wrong for several hundred steps and the loss visibly bumps. A
"resume" that produces a bump is not a resume, it is a warm restart, and it makes the
loss curve unusable as evidence.

### 3.2 How often — and why the current default is wrong

Checkpointing trades a known cost against an unknown one. Writing costs `C` seconds
every interval; a failure costs, on average, half an interval of lost work. With
interval `τ` and mean time between failures `M`, the wasted fraction of the run is:

```
waste(τ) ≈ C/τ  +  τ/(2M)
```

which is minimised at the Young/Daly optimum `τ* = sqrt(2·C·M)`, giving
`waste(τ*) = sqrt(2C/M)`.

Applying this to the actual reproduction. Step times are computed from the model's own
FLOP accounting at 40% MFU; `C` is estimated at 5 s, dominated by the GPU→host copy
rather than the disk write, and **should be replaced with a measurement** on the first
real run.

| Scenario | Step time | Assumed MTBF | `τ*` | Optimal waste | Waste at the configured 1000 steps |
| --- | --- | --- | --- | --- | --- |
| 1× A100, spot | 4.6 s | 4 h | ~83 steps | 2.6% | **16.0%** |
| 8× A100, one node, spot | 0.57 s | 4 h | ~666 steps | 2.6% | 2.9% |
| 8 nodes, spot, object storage (`C`≈7 s) | 0.57 s | 30 min | ~279 steps | 8.8% | **17.1%** |

The configured default of 1000 steps is well-tuned for exactly one of these cases and
badly wrong for the other two. On a single spot A100 it throws away about **16% of a
24-hour run** — roughly 3.9 hours of GPU time, for no benefit.

**The underlying problem is that the interval is expressed in steps.** A step is not a
fixed amount of time; it varies by an order of magnitude across these scenarios, so a
step-denominated interval cannot be right for more than one of them. The failure rate
it is protecting against is denominated in *wall-clock*.

**[proposed] Express the checkpoint interval in seconds** (`checkpoint_interval_seconds`,
default ~300 s), keeping the step-based setting as an optional override. This makes
the default correct across hardware without retuning, and matches how the risk
actually behaves. Small change: one config field and one condition in the training
loop.

### 3.3 Retention

`keep_last_n` rolling checkpoints (default 2) plus `best.pt` and `final.pt`, which are
never pruned. Two rolling checkpoints rather than one is deliberate: if the newest
turns out to be corrupt, there is something to fall back to. With `fsync` in place
(§6.1) that margin matters less, but it costs 1.4 GiB to keep.

---

## 4. Detecting silent corruption

This is the failure mode that actually invalidates results, and the one with the least
coverage today.

### 4.1 What it looks like

Silent data corruption — a GPU that computes a wrong result without raising an error —
is rare per device-hour and unavoidable at fleet scale; both Meta and Google have
published on encountering it in production. At a few hundred GPU-hours it is a tail
risk worth a cheap check. At a million it is a scheduled event.

The insidious property is that a corrupted gradient does not usually produce a NaN. It
produces a slightly wrong update, the loss curve stays smooth, and the final number is
quietly worse than it should be — indistinguishable from a hyperparameter being
slightly off.

### 4.2 Detection, cheapest first

**[implemented] Grad-norm logging.** The gradient norm is computed every step for
clipping and logged every `log_interval`. It is the cheapest available anomaly signal:
a healthy run's grad norm is smooth and slowly decreasing, so a step change or a spike
is a leading indicator of divergence, bad data, or bad hardware. Currently it is
*recorded* but nothing acts on it.

**[proposed] Non-finite guards, ~10 lines.** Assert the loss and grad norm are finite
every step and abort with the step number and the offending value if not. Today a NaN
propagates into the weights and every subsequent checkpoint is poisoned; the last good
checkpoint might be thousands of steps back. Fail on the first bad step instead. This
is the single highest value-per-line item in the document.

**[proposed] Grad-norm anomaly detection, ~30 lines.** Track a running median and MAD
of the grad norm; warn above ~8 MAD, and optionally skip the update. Skipping outlier
updates is a well-established stabilisation trick, and the same statistic doubles as a
hardware-fault signal.

**[proposed] Canary batch, ~50 lines.** Every *N* steps, run one fixed batch through
the model in eval mode and record the loss. It is deterministic given the weights, so
a mismatch on re-execution — same weights, same batch, different answer — is direct
evidence of nondeterministic hardware. Cost is one extra forward pass per *N* steps.
This is the only proposal here that detects SDC rather than inferring it.

**[proposed] Cross-rank agreement.** Under DDP every rank should compute an identical
loss for the same global batch after all-reduce. Periodically all-gathering per-rank
losses and comparing them isolates a single misbehaving rank, which the aggregate
metric hides entirely. Near-zero cost at a low frequency.

**[proposed] Checkpoint integrity.** Store a hash of the state dict in the payload and
verify on load. Catches bit rot and truncated writes at the point of use rather than
at the point of confusion.

---

## 5. Stragglers

A straggler does not fail the run, it taxes it. Under synchronous data parallelism
every rank waits for the slowest at each all-reduce, so one rank at 70% speed makes the
whole job 30% more expensive, indefinitely, with no error anywhere.

Common causes on rented hardware: thermal throttling, a degraded NIC, a noisy
neighbour on a shared host, or a GPU stuck in a lower clock state.

As noted in §2.1, data-induced imbalance is designed out — every rank processes
exactly the same number of tokens per step — so any imbalance that does appear is a
hardware or network problem. That is a useful diagnostic property: it removes the most
common false explanation before you start looking.

**[proposed] Per-rank step-time telemetry, ~40 lines.** Each rank records its own
step time; all-gather periodically, log the median and the spread, and warn when any
rank exceeds ~1.2× the median consistently. Currently only rank 0 reports throughput,
so a slow rank 5 is invisible — it shows up as "the job is slower than expected" with
nothing to point at.

**[proposed] Watchdog on the collective.** A hung rank is worse than a slow one: NCCL
blocks forever by default, so the job burns money making no progress and no process
has exited for a supervisor to restart. Setting `TORCH_NCCL_BLOCKING_WAIT` with a
timeout converts an infinite hang into a crash, which the restart path already
handles. This is a configuration change plus documentation, and it converts failure
mode #3's unbounded cost into a bounded one.

---

## 6. Prioritised gaps

Ordered by expected value — probability of occurring, times cost when it does, divided
by effort.

| Priority | Gap | Why it matters | Effort |
| --- | --- | --- | --- |
| 1 | No non-finite guard on loss/grad norm | A NaN poisons every subsequent checkpoint; recovery point could be thousands of steps back | ~10 lines |
| 2 | No `fsync` before rename | Power loss can leave a visible but empty checkpoint, defeating the atomic write | ~5 lines |
| 3 | No SIGTERM handler | Spot preemption gives ~30 s notice; that is enough to checkpoint and lose nothing instead of an interval | ~20 lines |
| 4 | Checkpoint interval in steps, not seconds | Wastes ~16% of a single-GPU spot run (§3.2) | ~10 lines |
| 5 | No NCCL timeout | A hung collective costs unbounded money with no crash to trigger a restart | config |
| 6 | No per-rank step-time telemetry | A straggler is invisible in rank-0-only logging | ~40 lines |
| 7 | No canary batch | Only direct evidence of silent hardware corruption | ~50 lines |
| 8 | No checkpoint integrity hash | Bit rot surfaces as confusion rather than an error | ~15 lines |

Items 1–5 are together under a hundred lines and cover every loud failure mode plus
the largest efficiency loss. They are the right next increment; 6–8 are worth doing
before any run long enough that a wasted week matters.

### 6.1 Note on the `fsync` gap

Worth stating precisely because "atomic write" is easy to over-claim. `os.replace` is
atomic against *interruption* — no reader ever sees a half-renamed file, so a killed
process is handled correctly, which is the common case and the one the test covers. It
is not atomic against *power loss*, because POSIX does not guarantee the file's data
reached stable storage before the rename became visible. The fix is `f.flush()`,
`os.fsync(f.fileno())`, then rename, then `fsync` the directory.

---

## 7. What changes at 1,000+ GPUs

None of this has been run at that scale. It is included because the design decisions
above have different consequences there, and knowing which ones break is part of
knowing whether they are the right decisions.

**Failure stops being an event and becomes a background rate.** If a node fails
independently every 4 hours, 1,000 nodes see a failure roughly every 15 seconds.
Nothing in this document's model survives that: checkpoint-and-restart-the-world is no
longer a recovery strategy, because the world will fail again before it finishes
restarting. Training has to continue *through* failures.

**Checkpoint I/O becomes the bottleneck.** A single rank writing 1.39 GiB is fine; a
trillion-parameter model's optimiser state is terabytes, and every rank writing to
shared storage at once will saturate it. The responses:

- *Sharded checkpointing* (`torch.distributed.checkpoint`): each rank writes only its
  own shard, in parallel. Requires resharding logic on resume, since the parallelism
  layout may differ.
- *Asynchronous checkpointing*: copy state to pinned host memory, let training
  continue, flush to storage in the background. Turns `C` from seconds into
  milliseconds of stall, which changes the §3.2 arithmetic completely.
- *In-memory redundancy*: keep a copy of each rank's state in a peer's memory so a
  single-node failure recovers over the network rather than from storage. Orders of
  magnitude faster, and the only approach that keeps up at a 15-second failure rate.

**3D parallelism changes what a checkpoint *is*.** With data, tensor and pipeline
parallelism combined, model state is partitioned across ranks and a checkpoint is
inherently topology-dependent. The §2.3 property — that a checkpoint reloads into a
bare model independent of topology — stops holding for free and has to be engineered,
via a canonical serialisation format plus resharding on load.

**Elasticity replaces restart.** Waiting for a replacement node wastes the other 999.
Elastic training (`torchrun --max_restarts`, with a rendezvous backend) continues at
reduced world size and reincorporates capacity when it returns. This interacts
directly with §2.1: because the token stream position is derived from the step number
and the world size, a world-size change mid-run alters which rank reads what. The
current derivation keeps the *global* token span per step correct across world sizes,
which is the property that matters — but it is worth stating that this is the
assumption elasticity would be built on, and it deserves an explicit test before
anyone relies on it.

**Network topology stops being an abstraction.** All-reduce cost depends on the
physical layout; rank-to-node assignment, hierarchical reduction (intra-node NVLink
before inter-node InfiniBand), and overlapping communication with computation become
first-order throughput concerns rather than details.

**Detection has to be automatic.** At 1,000 nodes nobody reads logs. Everything in §4
and §5 has to feed a health check that can evict a bad node and continue, rather than
warn a human who is asleep.

---

## 8. Principles carried over from production systems

Reliability engineering for training runs is not a separate discipline from
reliability engineering for services; the constants differ, the reasoning does not.
Four things transfer directly:

1. **Design for the failure you cannot see.** A crash is self-reporting and therefore
   cheap. The expensive failures are the ones where the system keeps running and keeps
   producing output that looks fine — a stale cache serving correct-looking data, a
   training run on a GPU with a bad SM. Effort belongs where the feedback loop is
   broken, which is why §4 is the longest section here despite covering the rarest
   failure.

2. **Make the recovery path the normal path.** A restore procedure exercised only
   during incidents is a procedure that does not work. Resumption here is covered by
   a test on every commit (`test_resume_continues_from_the_recorded_step`), and the
   CI end-to-end job trains, stops, and resumes from the checkpoint — so the recovery
   path is exercised more often than the failure it exists for. It is a clean shutdown
   rather than a kill, so it does not yet cover recovery from a *mid-write*
   interruption; that is what gap #2 in §6 is about.

3. **Bound the blast radius before optimising the common case.** Two rolling
   checkpoints, atomic renames, and a hard stop on NaN are all about capping the worst
   case rather than improving the average. A 16% throughput loss is an annoyance; an
   undetected corrupt run is a lost week.

4. **Instrument before you need it.** Throughput, MFU and grad norm are logged from
   step one, at zero cost, because the moment you want them is the moment something
   has already gone wrong and re-running a 24-hour job to collect them is not an
   option.

---

## References

- Daly, "A higher order estimate of the optimum checkpoint interval for restart
  dumps" (2006) — the `sqrt(2CM)` result used in §3.2.
- Dixit et al., "Silent Data Corruptions at Scale" (Meta, 2021).
- Hochschild et al., "Cores that don't count" (Google, 2021).
- Google, "Site Reliability Engineering" — the failure-domain and blast-radius
  framing in §8.
