#!/usr/bin/env bash
#
# Runs ON the pod, inside tmux, launched by `gpu.sh bench`. Measures everything that
# needs real CUDA hardware and nothing that does not.
#
# No corpus, no training. Every benchmark here works from an uploaded checkpoint and
# synthetic or self-supplied inputs, so the pod does useful work from the first minute
# instead of spending twenty of them tokenising. That is the whole point of this path:
# the earlier full pipeline spent 16 minutes on data prep before touching the GPU, and
# for a benchmarking session that would be most of the bill.
#
# Marker-guarded like the training pipeline, so an interrupted run resumes at the first
# unfinished stage.

set -uo pipefail

WORKDIR="${GPU_WORKDIR:-/root/llmfs}"
REPO="$WORKDIR/nanogpt-from-scratch"
RESULTS="${RESULTS_DIR:-$WORKDIR/results}"
MARKERS="$WORKDIR/.bench-stages"
CKPT="${CKPT:-$WORKDIR/checkpoints/best.pt}"
DRAFT="${DRAFT:-$WORKDIR/checkpoints/draft.pt}"

mkdir -p "$MARKERS" "$RESULTS"
cd "$REPO" || exit 1
# Fail here rather than three stages later. env.sh is written by bootstrap.sh and puts the
# llmfs entry points on PATH; without it every stage dies with "command not found", which
# reads like a packaging bug rather than "setup was never run". These scripts deliberately
# do not use `set -e` — a failing stage must not abort the ones after it — so a bare
# `source` of a missing file only warns and carries on. That cost a confusing round-trip
# on a pod billing by the minute.
if [[ ! -f "$WORKDIR/env.sh" ]]; then
  echo "FATAL: $WORKDIR/env.sh is missing, so llmfs is not installed on this pod." >&2
  echo "  Run setup first:" >&2
  echo "    cd $REPO && GPU_WORKDIR=$WORKDIR bash scripts/remote/bootstrap.sh" >&2
  echo "  (or ./scripts/gpu.sh setup from your own machine)" >&2
  exit 1
fi
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

# ------------------------------------------------------------------------ stages

do_throughput() {
  # Training knobs, inference batching, and the cache-vs-length sweep in one pass.
  llmfs-bench --suite all --config gpt2-124m \
    --checkpoint "$CKPT" \
    --out "$RESULTS/benchmarks-cuda.json"
}

do_quant() {
  # Memory and quality are device-independent and already measured; what CUDA adds is
  # the decode column, which is the one that was MPS-specific and therefore unreportable.
  llmfs-quant-eval --checkpoint "$CKPT" \
    --hellaswag-limit 0 \
    --out "$RESULTS/quantization-cuda.json"
}

do_spec() {
  local args=(--checkpoint "$CKPT" --max-new-tokens 128 --out "$RESULTS/speculative-cuda.json")
  # The model-draft rows only exist to show that a same-sized drafter cannot win, so
  # they are worth having but not worth failing over if the file was not uploaded.
  [[ -f "$DRAFT" ]] && args+=(--draft-checkpoint "$DRAFT")
  llmfs-spec-bench "${args[@]}"
}

# ------------------------------------------------------------------------- driver

say "benchmark pipeline starting"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

if [[ ! -f "$CKPT" ]]; then
  say "FATAL: no checkpoint at $CKPT — upload it with 'gpu.sh bench' rather than by hand"
  exit 1
fi
say "target checkpoint: $(du -h "$CKPT" | cut -f1)"
[[ -f "$DRAFT" ]] && say "draft checkpoint:  $(du -h "$DRAFT" | cut -f1)" \
                  || say "no draft checkpoint — model-draft rows will be skipped"

# Independent stages: a failure in one must not cost the others, since each is a
# separate measurement and the pod is billing either way.
stage throughput do_throughput || say "throughput benchmarks failed; continuing"
stage quantization do_quant   || say "quantization benchmarks failed; continuing"
stage speculative do_spec     || say "speculative benchmarks failed; continuing"

say "benchmark pipeline complete"
echo "--- artifacts ---"
ls -la "$RESULTS" 2>/dev/null || true
