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
#   ./scripts/gpu.sh data ablation      # just the corpus (hours)
#   ./scripts/gpu.sh sweep              # launch the ablation sweep, detached
#   ./scripts/gpu.sh repro              # launch the 124M reproduction, detached
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
GPU_WORKDIR="${GPU_WORKDIR:-/workspace}"       # a network volume, ideally
GPU_REPO="${GPU_REPO:-https://github.com/Padraigobrien08/LLMfromScratch.git}"
GPU_BRANCH="${GPU_BRANCH:-main}"

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
ssh_detached() {
  local name="$1" cmd="$2"
  ssh_run "tmux has-session -t $SESSION 2>/dev/null && tmux kill-session -t $SESSION || true"
  ssh_run "cd $REMOTE_REPO && tmux new-session -d -s $SESSION \
    'echo \"[$name] started \$(date -u +%FT%TZ)\" | tee -a $RUN_LOG; \
     { $cmd ; } 2>&1 | tee -a $RUN_LOG; \
     echo \"[$name] exited \$(date -u +%FT%TZ) rc=\$?\" | tee -a $RUN_LOG'"
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
  # /workspace and install into the wrong place.
  ssh_run "GPU_WORKDIR='$GPU_WORKDIR' bash -s" < "$REPO_ROOT/scripts/remote/bootstrap.sh"
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
     SWEEP_EXTRA='${SWEEP_EXTRA:-}' REPRO_EXTRA='${REPRO_EXTRA:-}' \
     bash $GPU_WORKDIR/pipeline.sh"
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
  running=$(ssh_run "tmux has-session -t $SESSION 2>/dev/null && echo yes || echo no")
  echo "tmux session '$SESSION': $running"
  echo
  echo "--- last 25 log lines ---"
  ssh_run "tail -n 25 $RUN_LOG 2>/dev/null || echo '(no log yet)'"
  echo
  echo "--- gpu ---"
  ssh_run "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo '(nvidia-smi unavailable)'"
}

cmd_watch() {
  local interval="${1:-120}"
  bold "watching (poll every ${interval}s); Ctrl-C to stop watching — the job keeps running"
  while true; do
    if [[ "$(ssh_run "tmux has-session -t $SESSION 2>/dev/null && echo yes || echo no")" == "no" ]]; then
      bold "job finished"
      spend_so_far
      cmd_fetch
      return 0
    fi
    printf '\033[2K\r%s  %s' "$(date +%H:%M:%S)" "$(ssh_run "tail -n 1 $RUN_LOG 2>/dev/null | tr -d '\n'" || true)"
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
    "$GPU_USER@$GPU_HOST:$GPU_WORKDIR/results/" "$REPO_ROOT/results/" 2>/dev/null || true

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

cmd_attach() { require_host; ssh -t -p "$GPU_PORT" -i "$GPU_KEY" "$GPU_USER@$GPU_HOST" "tmux attach -t $SESSION"; }
cmd_shell()  { require_host; ssh -t -p "$GPU_PORT" -i "$GPU_KEY" "$GPU_USER@$GPU_HOST" "cd $REMOTE_REPO 2>/dev/null; exec bash -l"; }
cmd_kill()   { ssh_run "tmux kill-session -t $SESSION 2>/dev/null || true"; bold "killed session '$SESSION'"; }

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
