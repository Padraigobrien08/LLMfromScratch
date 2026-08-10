"""Report training progress from metrics.jsonl.

Python block-buffers stdout when it is piped, so a long training run can appear
frozen in the log for many minutes while it is in fact fine. metrics.jsonl is
flushed on every write, which makes it the reliable source — and it carries the
numbers worth seeing anyway.
"""

import glob
import json
import os
import sys

RATE = float(os.environ.get("GPU_RATE", "0") or 0)
TARGET = 3.29  # the reproduction target, for context

paths = sorted(glob.glob(sys.argv[1] + "/*/metrics.jsonl"), key=os.path.getmtime)
if not paths:
    print("  no metrics yet")
    raise SystemExit

path = paths[-1]
run = os.path.basename(os.path.dirname(path))
with open(path) as fh:
    rows = [json.loads(line) for line in fh if line.strip()]
train = [r for r in rows if "train/loss" in r]
val = [r for r in rows if "val/loss" in r]

if not train:
    print(f"  {run}: started, no steps logged yet")
    raise SystemExit

last = train[-1]
step, tps = last["step"], last.get("perf/tokens_per_sec", 0)
total = int(sys.argv[2]) if len(sys.argv) > 2 else 0

print(f"  run          {run}")
print(f"  step         {step:,}" + (f" / {total:,}  ({step / total:.1%})" if total else ""))
print(f"  train loss   {last['train/loss']:.4f}")
if val:
    delta = val[-1]["val/loss"] - TARGET
    print(f"  val loss     {val[-1]['val/loss']:.4f}   (target {TARGET}, {delta:+.4f})")
print(f"  throughput   {tps:,.0f} tok/s   mfu {last.get('perf/mfu', 0) * 100:.1f}%")

if total and tps:
    hours = (total - step) * last.get("progress/tokens", 0) / max(step, 1) / tps / 3600
    if hours:
        cost = f"   ~${hours * RATE:.2f}" if RATE else ""
        print(f"  remaining    {hours:.2f} h{cost}")
