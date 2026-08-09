#!/usr/bin/env bash
#
# Runs ON the pod, inside tmux, launched by `gpu.sh all`. Carries the whole job
# through unattended: corpus -> ablation sweep -> reproduction -> final evaluation ->
# attention explorer.
#
# Every stage is guarded by a marker file, so re-running the pipeline after a crash,
# a preemption or a pod restart resumes at the first unfinished stage instead of
# redoing hours of work. Within a stage, the sweep skips completed arms and training
# resumes from its last checkpoint, so recovery is fine-grained rather than
# stage-granular.
#
# A stage that fails stops the pipeline. The alternative — pressing on — would run the
# reproduction against a corpus that failed to prepare, and bill for it.

set -uo pipefail

WORKDIR="${GPU_WORKDIR:-/workspace}"
REPO="$WORKDIR/LLMfromScratch"
DATA_DIR="$WORKDIR/data/fineweb-edu-10B"
OUT_DIR="$WORKDIR/out"
RESULTS="$WORKDIR/results"
MARKERS="$WORKDIR/.stages"

SEEDS="${SEEDS:-3}"
RUN_SWEEP="${RUN_SWEEP:-1}"
RUN_REPRO="${RUN_REPRO:-1}"
SWEEP_EXTRA="${SWEEP_EXTRA:-}"
REPRO_EXTRA="${REPRO_EXTRA:-}"

mkdir -p "$MARKERS" "$RESULTS"
cd "$REPO" || exit 1
# shellcheck disable=SC1091
source "$WORKDIR/env.sh"

stamp() { date -u +%FT%TZ; }
say()   { echo "[$(stamp)] $*"; }

# Run a stage unless its marker exists. The marker is written only on success.
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
  say "stage '$name': FAILED — stopping the pipeline"
  return 1
}

# ------------------------------------------------------------------------ stages

do_data() {
  # The corpus is the expensive prerequisite; never rebuild it if it is already there.
  if [[ -f "$DATA_DIR/meta.json" ]]; then
    say "corpus already present at $DATA_DIR"
    return 0
  fi
  llmfs-prepare-data --source fineweb-edu --out-dir "$DATA_DIR"
}

do_sweep() {
  # keep_last_n=0 keeps no rolling checkpoints — best.pt and final.pt are never
  # pruned, so every arm stays recoverable. At 3 seeds this is 39 run directories;
  # at the default of 2 rolling checkpoints each that is ~109 GiB, more than the
  # volume and more than the corpus. The sweep needs loss curves, not checkpoints.
  # shellcheck disable=SC2086
  llmfs-ablate \
    --out-dir "$OUT_DIR/ablations" \
    --results "$RESULTS/ablations.json" \
    --seeds "$SEEDS" \
    --set data.data_dir="$DATA_DIR" \
    --set log.keep_last_n=0 \
    $SWEEP_EXTRA
}

do_sweep_report() {
  llmfs-ablate-report --results "$RESULTS/ablations.json" --out-dir "$RESULTS"
}

do_repro() {
  # --resume auto so a restarted pod continues rather than starting over.
  # shellcheck disable=SC2086
  llmfs-train --config gpt2-124m --resume auto \
    --set data.data_dir="$DATA_DIR" \
    --set log.out_dir="$OUT_DIR" \
    $REPRO_EXTRA
}

do_repro_eval() {
  # The number that gets reported comes from the whole validation split, not the
  # 50-batch estimate the training loop uses for its curve.
  llmfs-eval \
    --checkpoint "$OUT_DIR/gpt2-124m-repro/best.pt" \
    --data-dir "$DATA_DIR" \
    --out "$RESULTS/reproduction.json"
}

do_viz() {
  # Rebuild the attention explorer from the real model. This is the point at which
  # the hosted page stops showing a toy trained on one book.
  llmfs-viz \
    --checkpoint "$OUT_DIR/gpt2-124m-repro/best.pt" \
    --out "$RESULTS/attention.html"
}

# ------------------------------------------------------------------------- driver

say "pipeline starting (seeds=$SEEDS sweep=$RUN_SWEEP repro=$RUN_REPRO)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

stage data do_data || exit 1

if [[ "$RUN_SWEEP" == "1" ]]; then
  stage sweep do_sweep || exit 1
  # Reporting is cheap and must not be able to fail the pipeline before the
  # reproduction runs — the raw results are already safely on disk.
  stage sweep_report do_sweep_report || say "report failed; results JSON is still intact"
fi

if [[ "$RUN_REPRO" == "1" ]]; then
  stage repro do_repro || exit 1
  stage repro_eval do_repro_eval || exit 1
  stage viz do_viz || say "explorer build failed; not fatal"
fi

say "pipeline complete"
echo "--- artifacts ---"
ls -la "$RESULTS" 2>/dev/null || true
