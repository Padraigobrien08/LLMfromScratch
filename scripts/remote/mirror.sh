#!/usr/bin/env bash
#
# Mirrors the newest training checkpoint to persistent storage on an interval.
#
# Training writes to the container disk, which is fast but erased when the pod stops.
# The pipeline copies the final checkpoint to the network volume once training
# finishes — which protects nothing if the pod stops at hour five of six. This closes
# that window: worst case you lose one interval, not the whole run.
#
# Runs in its own tmux session, so it can be started against a job already in flight.
#
# Usage: mirror.sh <interval_minutes> [run_name]

set -uo pipefail

INTERVAL_MIN="${1:-30}"
RUN_NAME="${2:-gpt2-124m-repro}"
WORKDIR="${GPU_WORKDIR:-/root/llmfs}"
KEEP="${KEEP_DIR:-/workspace/checkpoints}"

SRC_DIR="$WORKDIR/out/$RUN_NAME"
mkdir -p "$KEEP"

say() { echo "[mirror $(date -u +%FT%TZ)] $*"; }
say "mirroring $SRC_DIR -> $KEEP every ${INTERVAL_MIN} min"

while true; do
  sleep $(( INTERVAL_MIN * 60 ))

  # Prefer best.pt; fall back to the newest rolling checkpoint early in a run, before
  # the first evaluation has produced a best.
  src=""
  if [[ -f "$SRC_DIR/best.pt" ]]; then
    src="$SRC_DIR/best.pt"
  else
    src=$(ls -t "$SRC_DIR"/ckpt_step*.pt 2>/dev/null | head -1)
  fi
  [[ -n "$src" && -f "$src" ]] || { say "nothing to mirror yet"; continue; }

  # The volume is small, so keep exactly one checkpoint plus the metrics. Written to
  # a temporary name and renamed, because a half-copied 1.4GB file that replaced a
  # good one would be worse than not mirroring at all.
  need=$(( $(stat -c%s "$src") / 1024 / 1024 + 200 ))
  avail=$(df -BM --output=avail "$KEEP" | tail -1 | tr -dc '0-9')
  if [[ "$avail" -lt "$need" ]]; then
    say "only ${avail}MB free on the volume, need ~${need}MB — skipping"
    continue
  fi

  if cp -f "$src" "$KEEP/latest.pt.tmp" 2>/dev/null; then
    mv -f "$KEEP/latest.pt.tmp" "$KEEP/latest.pt"
    cp -f "$SRC_DIR/metrics.jsonl" "$KEEP/metrics.jsonl" 2>/dev/null || true
    cp -f "$SRC_DIR/config.yaml" "$KEEP/config.yaml" 2>/dev/null || true
    say "mirrored $(basename "$src") ($(du -h "$KEEP/latest.pt" | cut -f1))"
  else
    rm -f "$KEEP/latest.pt.tmp"
    say "copy failed; keeping the previous mirror"
  fi
done
