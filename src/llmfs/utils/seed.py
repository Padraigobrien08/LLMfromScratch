"""Seeding, for runs that reproduce."""

from __future__ import annotations

import os
import random

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
