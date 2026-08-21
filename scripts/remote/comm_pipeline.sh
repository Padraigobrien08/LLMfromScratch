#!/usr/bin/env bash
#
# Runs ON the pod, inside tmux, launched by `gpu.sh comm-sweep`. Measures how scaling
# efficiency depends on how much compute each gradient all-reduce is amortised over.
#
# WHY THIS EXISTS. The 8x RTX 5090 scaling run reached 95.1% efficiency over PCIe with no
# NVLink (docs/scaling.md). The explanation offered there is `no_sync`: at gradient
# accumulation 4, only the last micro-step syncs, so one all-reduce covers four
# micro-batches and communication is amortised over 4x the compute. That explanation
# predicts something testable — shrink the amortisation and efficiency should fall.
#
# So this holds the world size at 8 and varies tokens_per_step, which is the only thing
# that changes the accumulation:
#
#   accum@8 = tokens_per_step / (micro_batch x block_size x 8) = tokens_per_step / 131,072
#
# 1,048,576 -> accum 8      262,144 -> accum 2
#   524,288 -> accum 4      131,072 -> accum 1   (one all-reduce per micro-batch)
#
# Each batch size is run at world size 1 *and* 8, because efficiency is relative to
# single-GPU throughput at the *same* batch — reusing one baseline across batch sizes would
# quietly compare against the wrong denominator.
#
# This isolates communication on a single machine. Comparing two different machines
# (NVLink vs PCIe) would confound the interconnect with architecture, memory bandwidth and
# NCCL version all at once; varying one parameter on one box does not.
#
# Marker-guarded per batch size, so an interrupted run resumes where it stopped.

set -uo pipefail

WORKDIR="${GPU_WORKDIR:-/workspace}"
REPO="$WORKDIR/nanogpt-from-scratch"
RESULTS="${RESULTS_DIR:-$WORKDIR/results}"
MARKERS="$WORKDIR/.comm-stages"
DATA_DIR="${SCALING_DATA_DIR:-$WORKDIR/data/fineweb-edu-scaling}"
LIMIT_DOCS="${SCALING_LIMIT_DOCS:-40000}"
# 30/10 rather than the scaling sweep's 50/15. The largest batch here is 8x the smallest,
# so 30 steps at accum 8 is already 31M tokens — more than 30 would wrap the corpus, and
# per-run wall-clock is dominated by compile anyway.
STEPS="${COMM_STEPS:-30}"
WARMUP="${COMM_WARMUP:-10}"
WORLD_SIZES="${COMM_WORLD_SIZES:-1,8}"
BATCH_SIZES="${COMM_BATCH_SIZES:-1048576,524288,262144,131072}"
CONFIG="${SCALING_CONFIG:-gpt2-124m}"
# Extra override applied to every point, so the sweep stays internally consistent.
# Its intended use is runtime.compile=false. On the 8x5090 box, torch.compile under DDP
# cost ~690s per run and then hung outright — one rank pegged at 100% in what looked like
# inductor autotuning while the other seven blocked waiting for it, for 100 minutes. This
# sweep measures the *shape* of efficiency against accumulation, which does not need the
# production compile path; it only needs every point to share whatever setting is chosen.
COMM_SET="${COMM_SET:-}"

mkdir -p "$MARKERS" "$RESULTS"
cd "$REPO" || exit 1
if [[ ! -f "$WORKDIR/env.sh" ]]; then
  echo "FATAL: $WORKDIR/env.sh is missing, so llmfs is not installed on this pod." >&2
  echo "  Run setup first:" >&2
  echo "    cd $REPO && GPU_WORKDIR=$WORKDIR bash scripts/remote/bootstrap.sh" >&2
  exit 1
fi
# shellcheck disable=SC1091
source "$WORKDIR/env.sh"

say() { echo "[$(date -u +%FT%TZ)] $*"; }

do_data() {
  if compgen -G "$DATA_DIR/train_*.bin" > /dev/null; then
    say "corpus already present at $DATA_DIR"
    return 0
  fi
  llmfs-prepare-data --source fineweb-edu --out-dir "$DATA_DIR" --limit-docs "$LIMIT_DOCS" || true

  # Success is the artifacts existing, not the exit code. `tokenizers` can abort during
  # interpreter shutdown with a PyGILState_Release fatal error *after* every shard is
  # written and flushed, so a non-zero status here does not mean the corpus is bad. That
  # cost ten minutes of re-tokenising twice on rented hardware. prepare.py now exits before
  # that teardown, and this check means a future variant of the same crash cannot block a
  # run either. _assert_trainable has already vetoed a corpus with no training tokens, so
  # shards existing is a sufficient test.
  if compgen -G "$DATA_DIR/train_*.bin" > /dev/null; then
    say "corpus verified on disk at $DATA_DIR"
    return 0
  fi
  say "no train shards in $DATA_DIR after preparation"
  return 1
}

# ------------------------------------------------------------------------- driver

say "communication sweep starting"
say "  world sizes: $WORLD_SIZES   batch sizes: $BATCH_SIZES"
[[ -n "$COMM_SET" ]] && say "  extra override on every point: $COMM_SET"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
say "visible GPUs: $gpu_count"
if [[ "$gpu_count" -lt 8 ]]; then
  say "WARNING: this sweep is designed for 8 GPUs; comparability with"
  say "         results/scaling-5090x8.json needs the same 8x RTX 5090 configuration."
fi

if ! do_data; then
  say "corpus preparation failed — cannot measure"
  exit 1
fi

rc=0
IFS=',' read -r -a batches <<< "$BATCH_SIZES"
for tps in "${batches[@]}"; do
  accum=$(( tps / 131072 ))
  label="accum${accum}"

  if [[ -f "$MARKERS/$label.done" ]]; then
    say "batch $tps (accum@8=$accum): already complete, skipping"
    continue
  fi

  say "=== tokens_per_step=$tps  (accum@8=$accum) ==="
  started=$(date -u +%s)
  if llmfs-scaling \
      --config "$CONFIG" \
      --world-sizes "$WORLD_SIZES" \
      --steps "$STEPS" \
      --warmup "$WARMUP" \
      --out-dir "$WORKDIR/out/comm-$label" \
      --out "$RESULTS/comm-$label.json" \
      --label "$label" \
      --set "data.data_dir=$DATA_DIR" \
      --set "train.tokens_per_step=$tps" \
      ${COMM_SET:+--set "$COMM_SET"}; then
    date -u +%FT%TZ > "$MARKERS/$label.done"
    say "batch $tps: done in $(( $(date -u +%s) - started ))s"
  else
    say "batch $tps: FAILED (continuing; the other points are still worth having)"
    rc=1
  fi
done

if compgen -G "$RESULTS/comm-accum*.json" > /dev/null; then
  say "rendering the communication table"
  # The one consumer of these artifacts with a console entry point; it was installed
  # and documented nowhere, so the table only ever existed for people who knew.
  llmfs-comm-report "$RESULTS"/comm-accum*.json --plot "$RESULTS/comm-sweep.png" \
    || say "comm-report failed; the artifacts above are still good"
fi

say "communication sweep complete"
echo "--- artifacts ---"
ls -la "$RESULTS" 2>/dev/null || true
exit "$rc"
