#!/usr/bin/env bash
#
# Drive a rented GPU pod (RunPod or any SSH-reachable box) through the full
# pipeline: provision the environment, prepare data, run a job, and pull the
# results back.
#
# Three properties this is built around, all of them about the fact that these are
# multi-hour jobs on hardware you are paying for by the minute:
#
#   1. Jobs are detached. Everything runs inside tmux on the pod, so losing the SSH
#      connection — laptop sleeping, wifi dropping, a flight — does not kill a job
#      you are paying for. Reattaching is `gpu.sh attach`.
#   2. What runs is what is committed. The pod clones the repo from GitHub at a
#      specific commit rather than rsyncing your working tree, so a result can
#      always be traced to code someone else can check out. Use `--rsync` to
#      override while iterating.
#   3. Nothing is left only on the pod. `fetch` pulls results back and runs the
#      post-processing locally, so the artifacts survive the pod being terminated —
#      which, on a spot instance, can happen without warning.
#
# Configuration lives in .gpu.env (gitignored). Copy .gpu.env.example and fill it in.
# No secret is ever passed on a command line or written to the repo.
#
# Usage:
#   ./scripts/gpu.sh preflight          # verify the pod: GPU, disk, CUDA, volume
#   ./scripts/gpu.sh setup              # clone repo, install deps
#   ./scripts/gpu.sh all                # EVERYTHING: data -> sweep -> repro -> eval
#   ./scripts/gpu.sh bench              # benchmarks only: upload a checkpoint, measure
#   ./scripts/gpu.sh data ablation      # just the corpus (hours)
#   ./scripts/gpu.sh sweep              # launch the ablation sweep, detached
#   ./scripts/gpu.sh repro              # launch the 124M reproduction, detached
#   ./scripts/gpu.sh scaling LABEL      # multi-GPU scaling report (needs 2+ GPUs)
#   ./scripts/gpu.sh comm-sweep         # accumulation sweep: why scaling holds up
#   ./scripts/gpu.sh autostop [min]     # best-effort: stop the pod N min after the job ends
#   ./scripts/gpu.sh stop               # stop the pod now, from this machine (reliable)
#   ./scripts/gpu.sh mirror [min]       # mirror checkpoints to persistent storage
#   ./scripts/gpu.sh status             # progress + spend so far
#   ./scripts/gpu.sh watch              # poll until done, then fetch automatically
#   ./scripts/gpu.sh fetch              # pull results and post-process locally
#   ./scripts/gpu.sh attach             # attach to the running tmux session
#   ./scripts/gpu.sh shell              # interactive shell on the pod

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${GPU_ENV_FILE:-$REPO_ROOT/.gpu.env}"
STATE_DIR="$REPO_ROOT/.gpu-state"
SESSION="llmfs"

# ----------------------------------------------------------------- configuration

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
fi

GPU_HOST="${GPU_HOST:-}"
GPU_PORT="${GPU_PORT:-22}"
GPU_USER="${GPU_USER:-root}"
GPU_KEY="${GPU_KEY:-$HOME/.ssh/id_ed25519}"
GPU_RATE="${GPU_RATE:-0}"                      # $/hour, for the spend estimate
# Only used by `gpu.sh stop`, and only from this machine. Never copied to the pod: a key
# that can stop pods can also create them, so it has no business on a rented box.
RUNPOD_API_KEY="${RUNPOD_API_KEY:-}"
RUNPOD_POD_ID="${RUNPOD_POD_ID:-}"             # optional; otherwise read from the pod
GPU_WORKDIR="${GPU_WORKDIR:-/workspace}"       # a network volume, ideally
GPU_REPO="${GPU_REPO:-https://github.com/Padraigobrien08/LLMfromScratch.git}"
GPU_BRANCH="${GPU_BRANCH:-main}"
# Where results and the kept checkpoint land. Defaults inside GPU_WORKDIR, but on a
# pod whose bulk disk is ephemeral this should point at persistent storage.
GPU_RESULTS="${GPU_RESULTS:-$GPU_WORKDIR/results}"

REMOTE_REPO="$GPU_WORKDIR/LLMfromScratch"
RUN_LOG="$GPU_WORKDIR/run.log"
# bootstrap.sh decides whether to use the pod image's python or a fresh venv and
# records the answer here, so the driver never has to guess.
REMOTE_ENV="source $GPU_WORKDIR/env.sh"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

require_host() {
  [[ -n "$GPU_HOST" ]] || die "GPU_HOST is not set. Copy .gpu.env.example to .gpu.env and fill it in."
  [[ -f "$GPU_KEY" ]]  || die "SSH key not found: $GPU_KEY"
}

ssh_run() {
  require_host
  ssh -p "$GPU_PORT" -i "$GPU_KEY" \
      -o StrictHostKeyChecking=accept-new \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \
      "$GPU_USER@$GPU_HOST" "$@"
}

# Launch a command inside tmux so it outlives this SSH connection.
#
# The job is written to a file on the pod and tmux is asked to run *that*, rather
# than having the command embedded in tmux's own quoted argument. Embedding cannot
# work in general: the command carries config overrides like
# `SWEEP_EXTRA='--set data.micro_batch_size=64'`, and those single quotes terminate
# the outer single-quoted string, so tmux receives a fragment and the session dies
# instantly leaving no log. A file has no quoting to nest.
ssh_detached() {
  local name="$1" cmd="$2"
  # The "=" is load-bearing, and its absence cost real money. `tmux -t llmfs` matches by
  # *prefix* when no session is named exactly that (verified on tmux 3.2a), so with the
  # autostop watchdog running in "llmfs-watchdog" and no job session yet, this line
  # resolved to the watchdog and killed it. Net effect: arming the autostop and then
  # launching a job silently disarmed the autostop, and the pod idled until noticed by
  # hand. Every session target in this file is exact for that reason.
  ssh_run "tmux has-session -t =$SESSION 2>/dev/null && tmux kill-session -t =$SESSION || true"

  # Unquoted heredoc: $name/$cmd/$REMOTE_REPO expand here, \$(date) and \$? do not.
  ssh_run "cat > $GPU_WORKDIR/job.sh && chmod +x $GPU_WORKDIR/job.sh" <<EOF
#!/usr/bin/env bash
cd $REMOTE_REPO || exit 1
echo "[$name] started \$(date -u +%FT%TZ)"
$cmd
rc=\$?
echo "[$name] exited \$(date -u +%FT%TZ) rc=\$rc"
EOF

  ssh_run "tmux new-session -d -s $SESSION 'bash $GPU_WORKDIR/job.sh 2>&1 | tee -a $RUN_LOG'"

  # Confirm it is actually alive: a session that dies on launch is the failure this
  # function exists to make impossible, and silence looks identical to success.
  sleep 3
  if [[ "$(ssh_run "tmux has-session -t =$SESSION 2>/dev/null && echo yes || echo no")" != "yes" ]]; then
    ssh_run "tail -n 20 $RUN_LOG 2>/dev/null" || true
    die "the '$name' session exited immediately — see the log above"
  fi

  mkdir -p "$STATE_DIR"
  date -u +%s > "$STATE_DIR/started_at"
  echo "$name" > "$STATE_DIR/job"
  bold "launched '$name' in tmux session '$SESSION'"
  echo "  ./scripts/gpu.sh status   progress and spend"
  echo "  ./scripts/gpu.sh watch    poll until done, then fetch"
  echo "  ./scripts/gpu.sh attach   attach to the live session"
}

spend_so_far() {
  [[ -f "$STATE_DIR/started_at" ]] || return 0
  local start now elapsed
  start=$(cat "$STATE_DIR/started_at"); now=$(date -u +%s)
  elapsed=$(( now - start ))
  awk -v e="$elapsed" -v r="$GPU_RATE" 'BEGIN{
    h=e/3600.0; printf "elapsed %.2f h", h;
    if (r>0) printf "  |  ~$%.2f at $%.2f/hr", h*r, r;
    printf "\n"}'
}

# ---------------------------------------------------------------------- commands

cmd_preflight() {
  bold "pod preflight"
  ssh_run 'bash -s' <<'REMOTE'
set -u
echo "--- gpu ---"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "NO GPU VISIBLE (nvidia-smi failed)"
echo "--- disk ---"
df -h /workspace 2>/dev/null || df -h /
echo "--- workspace persistence ---"
if mountpoint -q /workspace 2>/dev/null; then
  echo "/workspace is a mount (network volume) — data survives pod termination"
else
  echo "WARNING: /workspace is NOT a separate mount."
  echo "  Anything written there dies with the pod, including hours of tokenised data."
  echo "  Attach a RunPod network volume before preparing the corpus."
fi
echo "--- cpu / ram ---"
nproc | sed 's/^/vcpu: /'; free -g 2>/dev/null | awk '/Mem:/{print "ram: "$2" GB"}'
REMOTE
}

cmd_setup() {
  bold "setting up $REMOTE_REPO on the pod"
  local rsync_mode="${1:-}"

  ssh_run "mkdir -p $GPU_WORKDIR"

  if [[ "$rsync_mode" == "--rsync" ]]; then
    warn "rsyncing the working tree — results will not be traceable to a commit"
    rsync -az --delete \
      --exclude '.git' --exclude '.venv' --exclude 'out' --exclude 'data/wizard' \
      --exclude 'data/fineweb*' --exclude '__pycache__' --exclude '.gpu-state' \
      -e "ssh -p $GPU_PORT -i $GPU_KEY" \
      "$REPO_ROOT/" "$GPU_USER@$GPU_HOST:$REMOTE_REPO/"
  else
    ssh_run "if [ -d $REMOTE_REPO/.git ]; then \
               cd $REMOTE_REPO && git fetch --all -q && git checkout -q $GPU_BRANCH && git reset --hard -q origin/$GPU_BRANCH; \
             else \
               git clone -q --branch $GPU_BRANCH $GPU_REPO $REMOTE_REPO; \
             fi"
  fi

  # GPU_WORKDIR must cross the SSH boundary explicitly — the remote shell does not
  # inherit the local environment, and bootstrap.sh would silently fall back to
  # /workspace and install into the wrong place. Same for the torch-pinning knobs, which
  # matter when two pods must run identical builds to be comparable.
  ssh_run "GPU_WORKDIR='$GPU_WORKDIR' \
           LLMFS_FORCE_TORCH_INSTALL='${LLMFS_FORCE_TORCH_INSTALL:-0}' \
           CUDA_INDEX='${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}' \
           bash -s" < "$REPO_ROOT/scripts/remote/bootstrap.sh"
  bold "setup complete"
  ssh_run "cd $REMOTE_REPO && git log --oneline -1 2>/dev/null || echo '(rsynced working tree)'"
}

cmd_data() {
  local which="${1:-ablation}"
  case "$which" in
    ablation|repro|fineweb)
      # Both the sweep and the reproduction read the same corpus.
      bold "preparing FineWeb-Edu (this takes hours — it runs detached)"
      ssh_detached "prepare-data" \
        "$REMOTE_ENV && llmfs-prepare-data --source fineweb-edu --out-dir $GPU_WORKDIR/data/fineweb-edu-10B"
      ;;
    smoke)
      bold "preparing the small local corpus (seconds)"
      ssh_run "cd $REMOTE_REPO && $REMOTE_ENV && llmfs-prepare-data --source text \
        --input data/wizard_of_oz.txt --out-dir $GPU_WORKDIR/data/wizard --shard-tokens 40000"
      ;;
    *) die "unknown data target: $which (expected ablation|repro|smoke)" ;;
  esac
}

cmd_all() {
  # The whole job unattended: corpus, sweep, reproduction, final eval, explorer.
  # Stages are marker-guarded on the pod, so re-running this after any interruption
  # resumes rather than restarting.
  bold "launching the full pipeline (data -> sweep -> reproduction -> eval -> explorer)"
  ssh_run "mkdir -p $GPU_WORKDIR/scripts"
  scp -q -P "$GPU_PORT" -i "$GPU_KEY" \
    "$REPO_ROOT/scripts/remote/pipeline.sh" \
    "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/pipeline.sh"
  ssh_run "chmod +x $GPU_WORKDIR/pipeline.sh"
  ssh_detached "pipeline" \
    "GPU_WORKDIR=$GPU_WORKDIR SEEDS=${SEEDS:-3} \
     RUN_SWEEP=${RUN_SWEEP:-1} RUN_REPRO=${RUN_REPRO:-1} \
     RESULTS_DIR=$GPU_RESULTS \
     SWEEP_EXTRA='${SWEEP_EXTRA:-}' REPRO_EXTRA='${REPRO_EXTRA:-}' \
     bash $GPU_WORKDIR/pipeline.sh"
}

cmd_bench() {
  # The benchmark-only path: no corpus, no training. Uploads the checkpoint and runs
  # every measurement that genuinely needs CUDA. A benchmarking session should not
  # spend its first 16 minutes tokenising a corpus it will never train on.
  local target="${1:-out/gpt2-124m-repro/best.pt}"
  local draft="${2:-out/gpt2-124m-repro/milestone_010pct_step0001907.pt}"

  [[ -f "$REPO_ROOT/$target" || -f "$target" ]] || die "target checkpoint not found: $target"
  local target_path="${target}"
  [[ -f "$REPO_ROOT/$target" ]] && target_path="$REPO_ROOT/$target"

  bold "uploading checkpoints (this is the only slow part)"
  ssh_run "mkdir -p $GPU_WORKDIR/checkpoints"
  rsync -az --progress -e "ssh -p $GPU_PORT -i $GPU_KEY" \
    "$target_path" "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/checkpoints/best.pt"

  local draft_path=""
  [[ -f "$REPO_ROOT/$draft" ]] && draft_path="$REPO_ROOT/$draft"
  [[ -z "$draft_path" && -f "$draft" ]] && draft_path="$draft"
  if [[ -n "$draft_path" ]]; then
    rsync -az --progress -e "ssh -p $GPU_PORT -i $GPU_KEY" \
      "$draft_path" "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/checkpoints/draft.pt"
  else
    warn "no draft checkpoint found; model-draft rows will be skipped"
  fi

  bold "launching benchmarks"
  scp -q -P "$GPU_PORT" -i "$GPU_KEY" \
    "$REPO_ROOT/scripts/remote/bench_pipeline.sh" "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/bench_pipeline.sh"
  ssh_run "chmod +x $GPU_WORKDIR/bench_pipeline.sh"
  ssh_detached "benchmarks" \
    "GPU_WORKDIR=$GPU_WORKDIR RESULTS_DIR=$GPU_RESULTS bash $GPU_WORKDIR/bench_pipeline.sh"
}

cmd_sweep() {
  bold "launching the ablation sweep"
  ssh_detached "ablation-sweep" \
    "$REMOTE_ENV && llmfs-ablate \
       --out-dir $GPU_WORKDIR/out/ablations \
       --results $GPU_WORKDIR/results/ablations.json \
       --seeds ${SEEDS:-3} \
       --set data.data_dir=$GPU_WORKDIR/data/fineweb-edu-10B \
       ${SWEEP_EXTRA:-}"
}

cmd_repro() {
  bold "launching the 124M reproduction"
  ssh_detached "reproduction" \
    "$REMOTE_ENV && llmfs-train --config gpt2-124m --resume auto \
       --set data.data_dir=$GPU_WORKDIR/data/fineweb-edu-10B \
       --set log.out_dir=$GPU_WORKDIR/out \
       ${REPRO_EXTRA:-}"
}

cmd_smoke() {
  # Proves the whole path on the pod in a couple of minutes before committing to a
  # multi-hour job. Cheap insurance against discovering a broken environment at hour six.
  bold "running the end-to-end smoke test on the pod"
  ssh_run "cd $REMOTE_REPO && $REMOTE_ENV && llmfs-train --config debug \
    --set data.data_dir=$GPU_WORKDIR/data/wizard \
    --set train.max_steps=30 --set log.eval_interval=15 \
    --set log.tensorboard=false --set log.out_dir=$GPU_WORKDIR/out/smoke"
}

cmd_status() {
  bold "job status"
  spend_so_far
  local running
  running=$(ssh_run "tmux has-session -t =$SESSION 2>/dev/null && echo yes || echo no")
  echo "tmux session '$SESSION': $running"
  echo
  # metrics.jsonl rather than the log: Python block-buffers stdout through tee, so a
  # healthy run can look frozen for many minutes. This reads what training actually
  # committed to disk.
  echo "--- training progress ---"
  ssh_run "GPU_RATE=$GPU_RATE python3 - $GPU_WORKDIR/out 19073" \
    < "$REPO_ROOT/scripts/remote/progress.py" 2>/dev/null || echo "  (unavailable)"
  echo
  echo "--- recent log ---"
  # tqdm redraws with carriage returns, so a progress bar is one enormous "line".
  # Translate CR to LF before tailing, or `tail` returns tens of KB of redraws.
  ssh_run "tail -c 200000 $RUN_LOG 2>/dev/null | tr '\\r' '\\n' | grep -v '^\\s*$' | tail -n 20 || echo '(no log yet)'"
  echo
  echo "--- gpu ---"
  ssh_run "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo '(nvidia-smi unavailable)'"
}

cmd_watch() {
  local interval="${1:-120}"
  bold "watching (poll every ${interval}s); Ctrl-C to stop watching — the job keeps running"
  while true; do
    if [[ "$(ssh_run "tmux has-session -t =$SESSION 2>/dev/null && echo yes || echo no")" == "no" ]]; then
      bold "job finished"
      spend_so_far
      cmd_fetch
      return 0
    fi
    printf '\033[2K\r%s  %s' "$(date +%H:%M:%S)" \
      "$(ssh_run "tail -c 4000 $RUN_LOG 2>/dev/null | tr '\\r' '\\n' | grep -v '^\\s*\$' | tail -n 1" || true)"
    sleep "$interval"
  done
}

cmd_fetch() {
  bold "fetching results"
  mkdir -p "$REPO_ROOT/results" "$REPO_ROOT/out"

  # Metrics and results only — small, and the artifacts every claim rests on.
  # Checkpoints stay on the volume; pull them explicitly with `fetch-checkpoints`.
  rsync -az --prune-empty-dirs \
    --include '*/' --include '*.json' --include '*.jsonl' --include 'config.yaml' \
    --exclude '*' \
    -e "ssh -p $GPU_PORT -i $GPU_KEY" \
    "$GPU_USER@$GPU_HOST:$GPU_RESULTS/" "$REPO_ROOT/results/" 2>/dev/null || true

  rsync -az --prune-empty-dirs \
    --include '*/' --include 'metrics.jsonl' --include 'config.yaml' --exclude '*' \
    -e "ssh -p $GPU_PORT -i $GPU_KEY" \
    "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/out/" "$REPO_ROOT/out/" 2>/dev/null || true

  bold "post-processing locally"
  if [[ -f "$REPO_ROOT/results/ablations.json" ]]; then
    "$REPO_ROOT/.venv/bin/llmfs-ablate-report" \
      --results "$REPO_ROOT/results/ablations.json" --out-dir "$REPO_ROOT/results"
  else
    warn "no results/ablations.json — nothing to report on (was this a reproduction run?)"
  fi

  echo
  bold "fetched"
  find "$REPO_ROOT/results" -maxdepth 1 -type f 2>/dev/null | sed 's|.*/|  |'
  echo
  echo "Results are small and belong in git; checkpoints do not:"
  echo "  git add results && git commit -m 'Add ablation results'"
}

cmd_fetch_checkpoints() {
  bold "fetching best checkpoints (these are large)"
  mkdir -p "$REPO_ROOT/out"
  rsync -az --progress --prune-empty-dirs \
    --include '*/' --include 'best.pt' --exclude '*' \
    -e "ssh -p $GPU_PORT -i $GPU_KEY" \
    "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/out/" "$REPO_ROOT/out/"
}

cmd_scaling() {
  # Multi-GPU scaling. Unlike `bench` this needs a corpus, because it measures the real
  # training step — but only a small one, since 30 steps is under 16M tokens.
  #
  # The label names the output file. It matters because this study measures two pods —
  # NVLink and PCIe — and an unlabelled result from the second would overwrite the first.
  local label="${1:-}"
  [[ -n "$label" ]] || die "usage: gpu.sh scaling <label>   e.g. 'a100x8' or '4090x8'
  The label names results/scaling-<label>.json. Two pods are compared in this study,
  so an unlabelled run would overwrite the other one's results."

  bold "launching scaling measurement (label: $label)"
  SCALING_EXTRA="SCALING_LABEL=$label ${SCALING_EXTRA:-}"
  scp -q -P "$GPU_PORT" -i "$GPU_KEY" \
    "$REPO_ROOT/scripts/remote/scaling_pipeline.sh" \
    "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/scaling_pipeline.sh"
  ssh_run "chmod +x $GPU_WORKDIR/scaling_pipeline.sh"
  ssh_detached "scaling" \
    "GPU_WORKDIR=$GPU_WORKDIR RESULTS_DIR=$GPU_RESULTS ${SCALING_EXTRA:-} \
     bash $GPU_WORKDIR/scaling_pipeline.sh"
}

cmd_comm_sweep() {
  # The communication-sensitivity sweep: fixed world size, varying gradient accumulation.
  # Tests the explanation the scaling report gives for 95.1% on PCIe, rather than restating
  # it — see scripts/remote/comm_pipeline.sh.
  bold "launching communication sweep"
  scp -q -P "$GPU_PORT" -i "$GPU_KEY" \
    "$REPO_ROOT/scripts/remote/comm_pipeline.sh" \
    "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/comm_pipeline.sh"
  ssh_run "chmod +x $GPU_WORKDIR/comm_pipeline.sh"
  ssh_detached "comm-sweep" \
    "GPU_WORKDIR=$GPU_WORKDIR RESULTS_DIR=$GPU_RESULTS ${COMM_EXTRA:-} \
     bash $GPU_WORKDIR/comm_pipeline.sh"
}

cmd_autostop() {
  # Runs in its own tmux session so it never disturbs a job already in flight.
  local grace="${1:-10}"
  bold "arming auto-stop: pod stops ${grace} min after the pipeline ends"
  scp -q -P "$GPU_PORT" -i "$GPU_KEY" \
    "$REPO_ROOT/scripts/remote/watchdog.sh" "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/watchdog.sh"
  ssh_run "chmod +x $GPU_WORKDIR/watchdog.sh"
  ssh_run "tmux kill-session -t =${SESSION}-watchdog 2>/dev/null || true"
  # No `| tee` here: the watchdog appends to its own log. Routing it through tee as well
  # both duplicated every line and, because tee block-buffers to a file, lost the last
  # few lines whenever the container went down — which was exactly when they mattered.
  ssh_run "tmux new-session -d -s ${SESSION}-watchdog \
    'GPU_WORKDIR=$GPU_WORKDIR RESULTS_DIR=$GPU_RESULTS SESSION=$SESSION \
     bash $GPU_WORKDIR/watchdog.sh $grace'"
  sleep 2
  if [[ "$(ssh_run "tmux has-session -t =${SESSION}-watchdog 2>/dev/null && echo yes || echo no")" == "yes" ]]; then
    bold "armed"
    echo "  results and the checkpoint are on the network volume and survive the stop"
    echo "  the corpus does not — regenerating it costs ~16 min"
    echo "  disarm with: ./scripts/gpu.sh autostop-off"
    echo
    echo "  NOTE: this is best-effort. It stops the pod through the RunPod API from"
    echo "  inside the container, so it fails if the pod's own networking is degraded."
    echo "  For an unattended overnight run, also set an idle timeout in the RunPod"
    echo "  console — that is enforced platform-side and cannot be defeated from here."
    echo "  'gpu.sh stop' stops the pod from this machine, which is the reliable path."
  else
    die "watchdog failed to start"
  fi
}

# Stops the pod from *this* machine rather than from inside the container. This is the
# dependable path: it does not care whether the pod's DNS works, whether the container
# was restarted, or whether a job wedged. It needs RUNPOD_API_KEY in .gpu.env, which is
# gitignored and stays local — the key is never copied to the rented box.
cmd_stop() {
  require_host
  if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
    die "set RUNPOD_API_KEY in .gpu.env (it stays on this machine; never goes to the pod).
  Create a read/write key at https://www.runpod.io/console/user/settings
  Or just press Stop in the console — that is equally fine and needs no key."
  fi
  local pod_id="${RUNPOD_POD_ID:-}"
  if [[ -z "$pod_id" ]]; then
    # The pod knows its own id, and the id is not a secret.
    pod_id=$(ssh_run "tr '\\0' '\\n' < /proc/1/environ | sed -n 's/^RUNPOD_POD_ID=//p' | head -1" 2>/dev/null | tr -d '\r')
  fi
  [[ -n "$pod_id" ]] || die "could not determine the pod id; set RUNPOD_POD_ID in .gpu.env"

  bold "stopping pod $pod_id"
  # The key goes in via the environment, so it never appears in argv or in this file.
  RUNPOD_API_KEY="$RUNPOD_API_KEY" POD_ID="$pod_id" python3 - <<'PY'
import json, os, sys, urllib.error, urllib.request

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
    payload = json.load(urllib.request.urlopen(req, timeout=30))
except urllib.error.HTTPError as exc:
    sys.exit(f"HTTP {exc.code}: {exc.read().decode()[:300]}")
except Exception as exc:  # noqa: BLE001 - any failure here means "not stopped"
    sys.exit(f"request failed: {exc}")

if payload.get("errors"):
    sys.exit(f"API error: {json.dumps(payload['errors'])[:300]}")
status = (payload.get("data") or {}).get("podStop") or {}
print(f"  desiredStatus = {status.get('desiredStatus', '?')}")
PY
  bold "stop requested — confirm in the console that it is no longer billing"
}

cmd_mirror() {
  # Own tmux session, so it can be armed against a job already running.
  local interval="${1:-30}" run="${2:-gpt2-124m-repro}"
  bold "mirroring checkpoints to persistent storage every ${interval} min"
  scp -q -P "$GPU_PORT" -i "$GPU_KEY" \
    "$REPO_ROOT/scripts/remote/mirror.sh" "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/mirror.sh"
  ssh_run "chmod +x $GPU_WORKDIR/mirror.sh"
  ssh_run "tmux kill-session -t =${SESSION}-mirror 2>/dev/null || true"
  ssh_run "tmux new-session -d -s ${SESSION}-mirror \
    'GPU_WORKDIR=$GPU_WORKDIR KEEP_DIR=$(dirname "$GPU_RESULTS")/checkpoints \
     bash $GPU_WORKDIR/mirror.sh $interval $run 2>&1 | tee -a $GPU_WORKDIR/mirror.log'"
  sleep 2
  if [[ "$(ssh_run "tmux has-session -t =${SESSION}-mirror 2>/dev/null && echo yes || echo no")" == "yes" ]]; then
    bold "armed — worst case you lose ${interval} min of training, not the run"
  else
    die "mirror failed to start"
  fi
}

cmd_autostop_off() {
  ssh_run "tmux kill-session -t =${SESSION}-watchdog 2>/dev/null || true"
  bold "auto-stop disarmed"
}

cmd_attach() { require_host; ssh -t -p "$GPU_PORT" -i "$GPU_KEY" "$GPU_USER@$GPU_HOST" "tmux attach -t =$SESSION"; }
cmd_shell()  { require_host; ssh -t -p "$GPU_PORT" -i "$GPU_KEY" "$GPU_USER@$GPU_HOST" "cd $REMOTE_REPO 2>/dev/null; exec bash -l"; }
cmd_kill()   { ssh_run "tmux kill-session -t =$SESSION 2>/dev/null || true"; bold "killed session '$SESSION'"; }

cmd_done() {
  bold "before you terminate the pod"
  spend_so_far
  cat <<'EOF'

  1. ./scripts/gpu.sh fetch              results + post-processing
  2. ./scripts/gpu.sh fetch-checkpoints  if you want the weights locally
  3. Terminate the pod in the RunPod console.

  A stopped pod still bills for its disk. A terminated pod does not — but anything
  outside a network volume is gone with it.
EOF
}

usage() {
  sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

main() {
  local cmd="${1:-}"; shift || true
  case "$cmd" in
    all)               cmd_all "$@" ;;
    bench)             cmd_bench "$@" ;;
    scaling)           cmd_scaling "$@" ;;
    comm-sweep)        cmd_comm_sweep "$@" ;;
    autostop)          cmd_autostop "$@" ;;
    autostop-off)      cmd_autostop_off "$@" ;;
    stop)              cmd_stop "$@" ;;
    mirror)            cmd_mirror "$@" ;;
    preflight)         cmd_preflight "$@" ;;
    setup)             cmd_setup "$@" ;;
    data)              cmd_data "$@" ;;
    smoke)             cmd_smoke "$@" ;;
    sweep)             cmd_sweep "$@" ;;
    repro)             cmd_repro "$@" ;;
    status)            cmd_status "$@" ;;
    watch)             cmd_watch "$@" ;;
    fetch)             cmd_fetch "$@" ;;
    fetch-checkpoints) cmd_fetch_checkpoints "$@" ;;
    attach)            cmd_attach "$@" ;;
    shell)             cmd_shell "$@" ;;
    kill)              cmd_kill "$@" ;;
    done)              cmd_done "$@" ;;
    ""|-h|--help|help) usage ;;
    *) die "unknown command: $cmd (try --help)" ;;
  esac
}

main "$@"
