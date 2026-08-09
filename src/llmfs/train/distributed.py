"""Distributed training setup.

Single-process and ``torchrun`` launches go through the same code path. The rank-0
process owns all logging and checkpointing; every other rank stays silent, so
output is readable and checkpoints are written once.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DistInfo:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def setup_distributed() -> DistInfo:
    """Initialise the process group if launched under ``torchrun``, else no-op."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return DistInfo(enabled=False, rank=0, local_rank=0, world_size=1)

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])

    # NCCL on GPU; gloo lets the distributed code path be exercised on a CPU box.
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return DistInfo(enabled=True, rank=rank, local_rank=local_rank, world_size=world_size)


def cleanup_distributed(info: DistInfo) -> None:
    if info.enabled and dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_mean(value: torch.Tensor, info: DistInfo) -> torch.Tensor:
    """Average a scalar across ranks, so reported metrics describe the whole run
    rather than whatever rank 0 happened to see."""
    if not info.enabled:
        return value
    dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def resolve_device(info: DistInfo, preference: str) -> torch.device:
    """Pin each rank to its own GPU under a distributed launch."""
    from ..utils.device import get_device

    if info.enabled and torch.cuda.is_available():
        return torch.device(f"cuda:{info.local_rank}")
    return get_device(preference)
