"""Seeding, for runs that reproduce."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and torch.

    Args:
        seed: the seed.
        deterministic: also force deterministic cuDNN/cuBLAS kernels. This makes runs
            bitwise-repeatable at a real throughput cost, so it is off by default and
            reserved for debugging a numerical discrepancy. Note that seeding alone
            already makes data order and initialisation reproducible; what it does not
            fix is non-deterministic reduction order inside GPU kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def rng_state() -> dict[str, Any]:
    """Every generator :func:`seed_everything` seeds, captured for a checkpoint.

    Seeding a resumed run reproduces the *start* of training, not the point it stopped:
    a run interrupted at step 12,000 and restarted draws the same dropout masks it drew
    at step 0. With ``dropout: 0.0`` — which every shipped config uses — nothing consumes
    the stream during a step, so the omission was invisible. It would not stay invisible
    for the first config that turned dropout on.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def load_rng_state(state: dict[str, Any] | None) -> bool:
    """Restore what :func:`rng_state` captured. False when there was nothing to restore.

    A checkpoint written before this existed simply has no entry, and a run resumed from
    one keeps the seeded stream it would have had — the old behaviour, not a crash.
    """
    if not state:
        return False
    random.setstate(tuple(state["python"]))
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu().to(torch.uint8))
    if "cuda" in state and torch.cuda.is_available():
        cuda = state["cuda"]
        # Only when the topology matches; restoring 8 GPUs' states onto 1 is not a resume.
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([s.cpu().to(torch.uint8) for s in cuda])
    return True
