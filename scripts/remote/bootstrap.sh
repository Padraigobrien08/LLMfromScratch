#!/usr/bin/env bash
#
# Runs ON the GPU pod. Installs uv, creates the venv, and installs the package with
# CUDA torch. Idempotent — safe to re-run after a pod restart.
#
# Invoked by scripts/gpu.sh setup; not usually run by hand.

set -euo pipefail

WORKDIR="${GPU_WORKDIR:-/workspace}"
REPO="$WORKDIR/LLMfromScratch"

cd "$REPO"

echo "--- system packages ---"
if ! command -v tmux >/dev/null || ! command -v rsync >/dev/null; then
  # Pod images vary; only reach for apt if something is genuinely missing.
  apt-get update -qq >/dev/null 2>&1 || true
  apt-get install -y -qq tmux rsync git >/dev/null 2>&1 || true
fi
command -v tmux >/dev/null || { echo "tmux missing and could not be installed" >&2; exit 1; }

echo "--- uv ---"
if ! command -v uv >/dev/null && [[ ! -x "$HOME/.local/bin/uv" ]]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "--- python environment ---"
# Note the absence of the CPU index used in CI: this box has a GPU, and the default
# PyPI wheel is the CUDA build. Installing the CPU wheel here would run the whole
# job on the CPU while still reporting a GPU in nvidia-smi — slow, expensive, and
# not obviously wrong from the logs.
uv venv --python 3.11 .venv >/dev/null
uv pip install -q -e ".[dev,train,bench]"

echo "--- verifying the GPU is actually usable from torch ---"
.venv/bin/python - <<'PY'
import sys, torch
print("torch:", torch.__version__)
if not torch.cuda.is_available():
    sys.exit("FATAL: torch cannot see a GPU. The job would silently run on CPU.")
name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
print(f"gpu: {name}  {props.total_memory/2**30:.0f} GiB  sm_{props.major}{props.minor}")
print("bf16 supported:", torch.cuda.is_bf16_supported())

# A real matmul, not just a capability flag — catches a broken driver/toolkit pairing
# now rather than at step 1 of a paid run.
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
import time
t = time.perf_counter()
for _ in range(20):
    a @ a
torch.cuda.synchronize()
tflops = 20 * 2 * 4096**3 / (time.perf_counter() - t) / 1e12
print(f"measured bf16 matmul: {tflops:,.0f} TFLOP/s")
PY

echo "--- smoke: import and build a model ---"
.venv/bin/python -c "
from llmfs.config import load_config
from llmfs.model import Transformer
c = load_config('gpt2-124m'); m = Transformer(c.model).cuda()
print(f'gpt2-124m builds on GPU: {m.num_params()/1e6:.1f}M params')
"

echo "bootstrap complete"
