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
MARKERS="$WORKDIR/.stages"

# Results live apart from the bulk. On RunPod the container disk is large and fast
# but is erased when the pod stops, while the network volume persists and is small.
# So the regenerable bulk — corpus, 39 sweeps' worth of checkpoints — goes on the
# fast disk, and the artifacts that would actually hurt to lose go on the volume.
RESULTS="${RESULTS_DIR:-$WORKDIR/results}"
KEEP_DIR="${KEEP_DIR:-$RESULTS/../checkpoints}"

SEEDS="${SEEDS:-3}"
RUN_SWEEP="${RUN_SWEEP:-1}"
RUN_REPRO="${RUN_REPRO:-1}"
SWEEP_EXTRA="${SWEEP_EXTRA:-}"
REPRO_EXTRA="${REPRO_EXTRA:-}"

mkdir -p "$MARKERS" "$RESULTS" "$KEEP_DIR"
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

do_repro_hellaswag() {
  # The downstream check. Validation loss can look right while a tokenizer or split
  # mismatch quietly invalidates the comparison to GPT-2; accuracy near the 25% floor
  # would expose that, and loss alone never would.
  llmfs-eval-hellaswag \
    --checkpoint "$OUT_DIR/gpt2-124m-repro/best.pt" \
    --data-dir "$WORKDIR/data/hellaswag" \
    --out "$RESULTS/hellaswag.json"
}

do_bench() {
  # Minutes of GPU, and an entire pillar of the project. Skipping it here means
  # renting a second pod later purely to measure what this one could have.
  llmfs-bench --suite both \
    --config gpt2-124m \
    --checkpoint "$OUT_DIR/gpt2-124m-repro/best.pt" \
    --out "$RESULTS/benchmarks.json"
}

do_samples() {
  # Free, and the README needs them. Fixed seeds so they are reproducible.
  for prompt in "The capital of France is" "In a distant galaxy," "def fibonacci(n):"; do
    echo "=== prompt: $prompt ==="
    llmfs-generate \
      --checkpoint "$OUT_DIR/gpt2-124m-repro/best.pt" \
      --prompt "$prompt" --max-new-tokens 128 --seed 0 --num-samples 2
  done | tee "$RESULTS/samples.txt"
}

do_preserve() {
  # The reproduction checkpoint is the deliverable, and the container disk it was
  # written to does not survive the pod being stopped. Copy it somewhere that does.
  cp -f "$OUT_DIR/gpt2-124m-repro/best.pt" "$KEEP_DIR/gpt2-124m-best.pt"
  cp -f "$OUT_DIR/gpt2-124m-repro/config.yaml" "$KEEP_DIR/" 2>/dev/null || true
  say "preserved $(du -h "$KEEP_DIR/gpt2-124m-best.pt" | cut -f1) to $KEEP_DIR"
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
  # Everything past this point is cheap, and none of it can fail the run that
  # matters — the checkpoint and its loss are already on disk.
  stage preserve do_preserve || say "could not preserve the checkpoint; not fatal"
  stage repro_hellaswag do_repro_hellaswag || say "hellaswag failed; not fatal"
  stage samples do_samples || say "sample generation failed; not fatal"
  stage viz do_viz || say "explorer build failed; not fatal"
fi

# Benchmarks last: they need the GPU but not the corpus, so if anything above went
# wrong there is still a reason to have paid for the pod.
stage bench do_bench || say "benchmarks failed; not fatal"

say "pipeline complete"
echo "--- artifacts ---"
ls -la "$RESULTS" 2>/dev/null || true
