"""The distributed training path, run for real with two gloo processes on CPU.

Everything else in the suite runs the trainer single-process, which is why a rank-0-only
final evaluation survived: on one rank the `all_reduce_mean` at the end of `evaluate()` is
a no-op, and the collective only deadlocks when there is a second rank that has already
gone home. The failure needs two processes to exist at all, so these tests make two.

They are slow by test standards (seconds, not milliseconds) and they are worth it: the
documented multi-GPU command hung after its last step, wrote no `final.pt`, and every
green test agreed the trainer was fine.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TIMEOUT_S = 300

WORKER = """
import json, sys
from pathlib import Path
from llmfs.config import load_config
from llmfs.train.trainer import Trainer
from llmfs.train.distributed import cleanup_distributed

data_dir, out_dir = sys.argv[1], sys.argv[2]
max_steps, eval_interval = int(sys.argv[3]), int(sys.argv[4])

cfg = load_config("debug")
cfg.model.vocab_size = 64
cfg.model.n_layer = 2
cfg.model.n_head = 2
cfg.model.n_embd = 32
cfg.model.block_size = 32
cfg.data.data_dir = data_dir
cfg.data.block_size = 32
cfg.data.micro_batch_size = 4
cfg.train.tokens_per_step = 4 * 32 * 4  # two micro-batches per rank at world size 2
cfg.train.max_steps = max_steps
cfg.optim.warmup_steps = 2
cfg.runtime.compile = False
cfg.runtime.device = "cpu"
cfg.log.out_dir = out_dir
cfg.log.run_name = "ddp"
cfg.log.tensorboard = False
cfg.log.eval_interval = eval_interval
cfg.log.eval_steps = 2
cfg.log.log_interval = 1
cfg.log.milestone_fracs = []

# Count how often DDP's gradient sync is suppressed. `no_sync` is the whole reason a
# multi-GPU run holds 95% efficiency over PCIe, and nothing in the suite ran it.
from torch.nn.parallel import DistributedDataParallel

_no_sync_calls = 0
_real_no_sync = DistributedDataParallel.no_sync


def _counting_no_sync(self):
    global _no_sync_calls
    _no_sync_calls += 1
    return _real_no_sync(self)


DistributedDataParallel.no_sync = _counting_no_sync

trainer = Trainer(cfg)
state = trainer.train()
cleanup_distributed(trainer.dist)
print("VAL", repr(state.history[-1]["val_loss"]))
print("NOSYNC", _no_sync_calls, trainer.grad_accum_steps)
print("done")
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def make_corpus(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, size in [("train_000000.bin", 20_000), ("val_000000.bin", 4_000)]:
        rng.integers(0, 64, size=size, dtype=np.uint16).tofile(data_dir / name)
    (data_dir / "meta.json").write_text(json.dumps({"tokenizer": "gpt2", "vocab_size": 64}))


def run_two_ranks(tmp_path: Path, max_steps: int, eval_interval: int) -> Path:
    """Run the trainer under two gloo ranks; fail on a nonzero exit or a hang."""
    data_dir, out_dir = tmp_path / "data", tmp_path / "out"
    make_corpus(data_dir)
    script = tmp_path / "worker.py"
    script.write_text(WORKER)

    env = dict(os.environ)
    env.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(free_port()),
        WORLD_SIZE="2",
        OMP_NUM_THREADS="1",
        # gloo would otherwise pick an interface that may not exist on a CI runner.
        GLOO_SOCKET_IFNAME=env.get(
            "GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo"
        ),
    )
    args = [
        sys.executable,
        str(script),
        str(data_dir),
        str(out_dir),
        str(max_steps),
        str(eval_interval),
    ]

    procs = []
    for rank in range(2):
        procs.append(
            subprocess.Popen(
                args,
                env={**env, "RANK": str(rank), "LOCAL_RANK": str(rank)},
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    try:
        outputs = [p.communicate(timeout=TIMEOUT_S)[0] for p in procs]
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        pytest.fail(
            f"a rank did not exit within {TIMEOUT_S}s — the run hung on a collective "
            "that one rank never reached"
        )

    for rank, (proc, out) in enumerate(zip(procs, outputs)):
        assert proc.returncode == 0, f"rank {rank} exited {proc.returncode}:\n{out}"
        assert "done" in out, f"rank {rank} did not finish:\n{out}"
    return out_dir / "ddp", outputs


def reported(outputs: list[str], tag: str) -> list[list[str]]:
    return [
        line.split()[1:]
        for out in outputs
        for line in out.splitlines()
        if line.startswith(tag + " ")
    ]


def val_losses(outputs: list[str]) -> list[float]:
    return [
        float(line.split()[1])
        for out in outputs
        for line in out.splitlines()
        if line.startswith("VAL ")
    ]


@pytest.mark.slow
def test_final_checkpoint_is_written_when_the_last_step_misses_an_eval_boundary(
    tmp_path: Path,
) -> None:
    """The documented case: 19,073 steps at an eval interval of 250 leaves 73 over.

    The final `_evaluate_and_checkpoint(final=True)` was called under `is_main`, but it
    calls `evaluate()`, which ends in an all-reduce. Rank 1 exited cleanly while rank 0
    blocked forever — a run that trained to completion and then produced no `final.pt`.
    6 steps with an interval of 4 is the same off-boundary arithmetic, in seconds.
    """
    run_dir, outputs = run_two_ranks(tmp_path, max_steps=6, eval_interval=4)
    assert (run_dir / "final.pt").exists(), "trained to the last step but wrote no final.pt"
    assert (run_dir / "best.pt").exists()

    # And the final eval really is a group-wide reduction: every rank came back with the
    # same number, which is only true if the collective completed on all of them.
    losses = val_losses(outputs)
    assert len(losses) == 2 and losses[0] == losses[1], f"ranks disagree on val loss: {losses}"

    # The reduced value is what got written down, not rank 0's own shard.
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    logged = [r["val/loss"] for r in records if "val/loss" in r]
    assert logged and logged[-1] == pytest.approx(losses[0])


@pytest.mark.slow
def test_on_boundary_run_still_completes(tmp_path: Path) -> None:
    """The control. This case always passed — the final eval is skipped entirely when the
    last step already evaluated — so on its own it proves nothing about the fix."""
    run_dir, _ = run_two_ranks(tmp_path, max_steps=8, eval_interval=4)
    assert (run_dir / "final.pt").exists()


@pytest.mark.slow
def test_gradient_sync_is_suppressed_on_every_micro_step_but_the_last(tmp_path: Path) -> None:
    """`no_sync` is the reason 8 GPUs hold 95.1% over PCIe, and it ran in no test.

    DDP all-reduces the whole gradient on every backward pass. During accumulation only
    the last micro-step needs it, so the rest run under `model.no_sync()` — that is the
    claim `docs/scaling.md` builds its explanation on, and the trainer's own comment calls
    it "the difference between communication-bound and compute-bound". Losing it would not
    fail a single existing test; it would just make every multi-GPU run slower, which no
    CPU suite can see. Counting the calls is what a CPU suite can check.
    """
    _, outputs = run_two_ranks(tmp_path, max_steps=6, eval_interval=4)
    counted = reported(outputs, "NOSYNC")
    assert len(counted) == 2, f"both ranks must report: {counted}"

    for calls, grad_accum in ((int(a), int(b)) for a, b in counted):
        assert grad_accum > 1, "the test config must accumulate, or there is nothing to skip"
        # Six steps, and every micro-step but the last of each runs unsynced.
        assert calls == 6 * (grad_accum - 1), f"{calls} suppressed syncs at accum {grad_accum}"
