#!/usr/bin/env bash
#
# Waits for the pipeline to finish, then stops the pod so it does not sit idle
# burning GPU-hours. Runs in its own tmux session, so it can be started or killed
# without touching a job already in flight.
#
# HOW IT STOPS THE POD — and the version of this that did not work.
#
# The first version avoided credentials on principle: rather than call the RunPod API it
# sent SIGTERM to the container's init, on the belief that exiting init is what RunPod
# treats as the pod finishing. That belief was wrong, and it cost real money before the
# 4090 benchmark session caught it. `kill -TERM 1` **restarts the container**: init comes
# back with a fresh PID, the tmux server and every process under it are destroyed, and the
# pod keeps right on billing. Worse, it destroys the evidence — the watchdog kills its own
# logging pipe, so the log ends before the line that would have explained what happened.
# Verified directly: PID 1's start time moves to the moment the signal is sent, and SSH
# keeps working afterwards.
#
# So the credential objection had to be revisited, and it turns out not to apply. RunPod
# already injects RUNPOD_POD_ID and RUNPOD_API_KEY into the container itself — they are in
# /proc/1/environ, put there by the platform, not by us. Nothing new is placed on the box.
# The key is read on the pod, used on the pod to stop that same pod, and never printed or
# copied off it. (It *can* also create pods, which is why it is never echoed, never
# written to a file, and never passed through an argument list where `ps` would show it.)
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
LOG="${WATCHDOG_LOG:-$WORKDIR/watchdog.log}"

# Appends straight to the file rather than relying on a `tee` further up the pipe. tee
# block-buffers when its output is a file, so the earlier version lost every line it wrote
# in its final minutes — precisely the lines needed to debug a failed stop.
say() {
  local line="[watchdog $(date -u +%FT%TZ)] $*"
  echo "$line"
  printf '%s\n' "$line" >> "$LOG" 2>/dev/null || true
}

# Exact match. `tmux has-session -t llmfs` also succeeds on a *prefix* match, and this
# watchdog's own session is called "$SESSION-watchdog" — so the loose form can report the
# job still running when only the watchdog is left, and wait forever. The "=" forces exact.
job_running() { tmux has-session -t "=$SESSION" 2>/dev/null; }

# Stops the pod through the platform API, which is the only thing that actually stops
# billing. Credentials come from init's environ; they never reach a log, a file, or argv.
stop_pod() {
  local pod_id api_key
  pod_id=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_POD_ID=//p' | head -1)
  api_key=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^RUNPOD_API_KEY=//p' | head -1)

  if [[ -z "$pod_id" || -z "$api_key" ]]; then
    say "FAILED to stop: RunPod credentials are not in /proc/1/environ on this image"
    say "  stop the pod from the RunPod console — results are already on the network volume"
    return 1
  fi

  # python3 rather than curl, deliberately. curl's --url-query would keep the key out of
  # argv, but it needs curl 7.87+ and these images ship 7.81, where the flag is simply
  # unrecognised — curl then writes nothing to stdout and the failure looks like an empty
  # API response. python3 is present on every image used here and takes the key from the
  # environment, which keeps it out of argv just as well.
  local response
  response=$(RUNPOD_API_KEY="$api_key" POD_ID="$pod_id" python3 - <<'PY' 2>&1
import json, os, urllib.request

key, pod = os.environ["RUNPOD_API_KEY"], os.environ["POD_ID"]
body = json.dumps(
    {"query": "mutation { podStop(input: {podId: \"%s\"}) { id desiredStatus } }" % pod}
).encode()
req = urllib.request.Request(
    "https://api.runpod.io/graphql?api_key=" + key,
    data=body,
    headers={"Content-Type": "application/json"},
)
try:
    print(json.dumps(json.load(urllib.request.urlopen(req, timeout=30))))
except Exception as exc:  # noqa: BLE001 - anything here means "not stopped"
    print(f"REQUEST_FAILED: {type(exc).__name__}: {exc}")
PY
  )

  if [[ "$response" == *'"desiredStatus"'* && "$response" != *'"errors"'* ]]; then
    say "stop requested for pod $pod_id — API reports: $response"
    return 0
  fi
  say "FAILED to stop pod $pod_id. API said: ${response:-<none>}"
  say "  This path only works if the pod can reach api.runpod.io. A container restart can"
  say "  leave /etc/resolv.conf empty, in which case nothing here can stop the pod."
  say "  Stop it from the RunPod console, or run 'gpu.sh stop' from your own machine."
  say "  Results are already on the network volume and survive the stop."
  return 1
}

say "watching session '$SESSION'; will stop the pod ${GRACE_MIN} min after it ends"

while true; do
  # Wait for the pipeline to finish.
  while job_running; do sleep 60; done
  say "pipeline session ended"

  # The log lives on the container disk, which is about to be wiped. Without it a
  # failed run becomes undiagnosable, which is the one thing worse than idling.
  mkdir -p "$KEEP"
  cp -f "$WORKDIR/run.log" "$KEEP/run.log" 2>/dev/null && say "saved run.log to $KEEP"

  say "grace period: ${GRACE_MIN} min — relaunch the pipeline to cancel the stop"
  sleep $(( GRACE_MIN * 60 ))

  # Someone restarted the job during the grace period; go back to watching rather
  # than stopping a run that is now in progress.
  if job_running; then
    say "pipeline restarted — stop cancelled, resuming watch"
    continue
  fi

  say "stopping the pod now"
  sync

  stop_pod || true

  # Deliberately no fallback to signalling init. That is what the old version did, and it
  # restarted the container while leaving the pod billing — a fallback that looks like it
  # worked is worse than none, because the log said "sent shutdown" and the meter kept
  # running. If the API call failed, the lines above say so and say what to do about it.
  #
  # Stay alive either way: if the stop did land the pod is going down regardless, and if it
  # did not, a live watchdog keeps reporting rather than exiting silently.
  sleep 120
  if job_running; then
    say "pipeline restarted after a failed stop — resuming watch"
    continue
  fi
  say "still running $(date -u +%FT%TZ) — the stop did not take effect; stop the pod manually"
  exit 1
done
