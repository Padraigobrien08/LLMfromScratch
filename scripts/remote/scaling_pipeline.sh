#!/usr/bin/env bash
#
# Runs ON the pod, inside tmux, launched by `gpu.sh scaling`. Measures multi-GPU scaling
# of the real trainer at world sizes 1, 2, 4, 8 (whatever the box actually has).
#
# Needs a corpus, unlike the benchmark path — the point is to measure the real training
# step, and that reads real batches. But it needs a *small* corpus: 30 optimiser steps at
# 524,288 tokens is under 16M tokens, so this prepares a few tens of thousands of
# documents rather than the full 10B-token sample. That is ~2 minutes of tokenisation
# instead of ~16, and the content is irrelevant to a throughput measurement.
#
# Marker-guarded, so an interrupted run resumes at the first unfinished stage.

set -uo pipefail

WORKDIR="${GPU_WORKDIR:-/root/llmfs}"
REPO="$WORKDIR/LLMfromScratch"
RESULTS="${RESULTS_DIR:-$WORKDIR/results}"
MARKERS="$WORKDIR/.scaling-stages"
DATA_DIR="${SCALING_DATA_DIR:-$WORKDIR/data/fineweb-edu-scaling}"
LIMIT_DOCS="${SCALING_LIMIT_DOCS:-40000}"
# 50/15 rather than 30/10. The extra 20 steps cost ~35s at world size 1 and essentially
# nothing at world size 8 (each step there is ~230ms), and they take the steady-state
# sample from 20 points to 35 — worth it when a single transient can otherwise invent a
# scaling cliff. Warmup is generous because compile is on, and it lands on step 1.
STEPS="${SCALING_STEPS:-50}"
WARMUP="${SCALING_WARMUP:-15}"
WORLD_SIZES="${SCALING_WORLD_SIZES:-1,2,4,8}"
CONFIG="${SCALING_CONFIG:-gpt2-124m}"
# Names the output file. Two pods (NVLink vs PCIe) are measured in this study, and an
# unlabelled scaling.json from the second would silently overwrite the first.
LABEL="${SCALING_LABEL:-unlabelled}"

mkdir -p "$MARKERS" "$RESULTS"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source "$WORKDIR/env.sh"

say() { echo "[$(date -u +%FT%TZ)] $*"; }

stage() {
  local name="$1"; shift
  if [[ -f "$MARKERS/$name.done" ]]; then
    say "stage '$name': already complete, skipping"
    return 0
  fi
  say "stage '$name': starting"
  local started; started=$(date -u +%s)
  if "$@"; then
    date -u +%FT%TZ > "$MARKERS/$name.done"
    say "stage '$name': done in $(( $(date -u +%s) - started ))s"
    return 0
  fi
  say "stage '$name': FAILED"
  return 1
}

do_data() {
  # Idempotent: if shards are already there, tokenising again buys nothing.
  if compgen -G "$DATA_DIR/train_*.bin" > /dev/null; then
    say "corpus already present at $DATA_DIR"
    return 0
  fi
  llmfs-prepare-data --source fineweb-edu --out-dir "$DATA_DIR" --limit-docs "$LIMIT_DOCS"
}

do_scaling() {
  llmfs-scaling \
    --config "$CONFIG" \
    --world-sizes "$WORLD_SIZES" \
    --steps "$STEPS" \
    --warmup "$WARMUP" \
    --out-dir "$WORKDIR/out/scaling" \
    --out "$RESULTS/scaling-$LABEL.json" \
    --label "$LABEL" \
    --set "data.data_dir=$DATA_DIR"
}

# ------------------------------------------------------------------------- driver

say "scaling pipeline starting (label: $LABEL)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

gpu_count=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
say "visible GPUs: $gpu_count (world sizes above this are skipped automatically)"
if [[ "$gpu_count" -lt 2 ]]; then
  say "WARNING: a scaling report needs at least 2 GPUs; this box has $gpu_count"
fi

stage data    do_data    || { say "corpus preparation failed — cannot measure scaling"; exit 1; }
stage scaling do_scaling || say "scaling measurement failed"

say "scaling pipeline complete"
echo "--- artifacts ---"
ls -la "$RESULTS" 2>/dev/null || true
