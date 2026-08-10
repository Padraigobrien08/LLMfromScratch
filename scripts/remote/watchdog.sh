#!/usr/bin/env bash
#
# Waits for the pipeline to finish, then stops the pod so it does not sit idle
# burning GPU-hours. Runs in its own tmux session, so it can be started or killed
# without touching a job already in flight.
#
# Why not runpodctl: stopping a pod through the API needs a RunPod key, and that key
# can also create pods. Putting one on a rented box to save a few dollars of idle time
# is a bad trade. Instead this exits the container's init process, which RunPod treats
# as the pod finishing — no credential involved.
#
# What survives: /workspace is a network volume and is untouched. Everything the run
# produced — results, plots, the reproduction checkpoint — is written there by the
# pipeline. The container disk is wiped, taking the corpus (regenerable in ~16 min)
# and the sweep's per-arm checkpoints (not needed; their loss curves are already
# inside ablations.json).
#
# Usage: watchdog.sh <grace_minutes>

set -uo pipefail

GRACE_MIN="${1:-10}"
SESSION="${SESSION:-llmfs}"
WORKDIR="${GPU_WORKDIR:-/root/llmfs}"
KEEP="${RESULTS_DIR:-/workspace/results}"

say() { echo "[watchdog $(date -u +%FT%TZ)] $*"; }

say "watching session '$SESSION'; will stop the pod ${GRACE_MIN} min after it ends"

while true; do
  # Wait for the pipeline to finish.
  while tmux has-session -t "$SESSION" 2>/dev/null; do sleep 60; done
  say "pipeline session ended"

  # The log lives on the container disk, which is about to be wiped. Without it a
  # failed run becomes undiagnosable, which is the one thing worse than idling.
  mkdir -p "$KEEP"
  cp -f "$WORKDIR/run.log" "$KEEP/run.log" 2>/dev/null && say "saved run.log to $KEEP"

  say "grace period: ${GRACE_MIN} min — relaunch the pipeline to cancel the stop"
  sleep $(( GRACE_MIN * 60 ))

  # Someone restarted the job during the grace period; go back to watching rather
  # than stopping a run that is now in progress.
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    say "pipeline restarted — stop cancelled, resuming watch"
    continue
  fi

  say "stopping the pod now"
  sync

  # Best effort, most-preferred first. runpodctl only works if a key happens to be
  # configured; it is never required.
  if runpodctl stop pod "${RUNPOD_POD_ID:-}" >/dev/null 2>&1; then
    say "stopped via runpodctl"
    exit 0
  fi
  # Exiting init ends the container, which RunPod reports as the pod stopping.
  kill -TERM 1 2>/dev/null || true
  sleep 20
  kill -KILL 1 2>/dev/null || true
  say "sent shutdown to init; if the pod is still running, stop it in the console"
  exit 0
done
