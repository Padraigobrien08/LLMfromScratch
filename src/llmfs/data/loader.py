"""Memory-mapped shard loading.

The design constraint that shapes this file is resumption. A run that dies at step
14,000 must restart consuming *exactly* the tokens it would have consumed, or the
resumed run is no longer the run whose loss curve you are reporting — and worse,
it silently re-trains on data it has already seen.

So the loader has no hidden state. The corpus is one long token stream, and the
position in it is a pure function of the step number:

    position = step * grad_accum * world_size * micro_batch * block_size

There is nothing to checkpoint beyond the step counter, nothing to go stale, and
:meth:`ShardDataLoader.set_step` makes restarting idempotent by construction. That
property is what ``docs/fault-tolerance.md`` builds on.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class ShardDataLoader:
    """Sequential, deterministic, distributed-aware loader over ``.bin`` token shards.

    Shards are ``uint16`` — enough for any vocabulary below 65,536, and half the
    disk and page-cache footprint of ``int32``. They are memory-mapped rather than
    read, so a 10B-token corpus costs no resident memory and the OS page cache does
    the prefetching.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        micro_batch_size: int,
        block_size: int,
        rank: int = 0,
        world_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.split = split
        self.B = micro_batch_size
        self.T = block_size
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(device)

        self.shard_paths = sorted(self.data_dir.glob(f"{split}_*.bin"))
        if not self.shard_paths:
            raise FileNotFoundError(
                f"no shards matching '{split}_*.bin' in {self.data_dir}. "
                f"Run `llmfs-prepare-data` first."
            )

        self.shards = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.shard_paths]
        self.shard_lengths = [len(s) for s in self.shards]
        # Exclusive prefix sums, for translating a global offset to (shard, offset).
        self.shard_starts = np.cumsum([0, *self.shard_lengths])
        self.total_tokens = int(self.shard_starts[-1])

        tokens_per_micro_step = self.world_size * self.B * self.T
        if self.total_tokens < tokens_per_micro_step + 1:
            raise ValueError(
                f"split '{split}' has {self.total_tokens:,} tokens, too few for one "
                f"micro-step of {tokens_per_micro_step:,}"
            )

        self.position = 0
        self.epoch = 0

    # ------------------------------------------------------------------ reading

    def _read(self, start: int, length: int) -> np.ndarray:
        """Read ``length`` tokens from global offset ``start``, spanning shards and
        wrapping past the end of the corpus."""
        start %= self.total_tokens
        pieces: list[np.ndarray] = []
        remaining = length
        offset = start

        while remaining > 0:
            shard_idx = int(np.searchsorted(self.shard_starts, offset, side="right") - 1)
            shard_idx = min(shard_idx, len(self.shards) - 1)
            local = offset - int(self.shard_starts[shard_idx])
            take = min(remaining, self.shard_lengths[shard_idx] - local)
            pieces.append(np.asarray(self.shards[shard_idx][local : local + take]))
            remaining -= take
            offset = (offset + take) % self.total_tokens

        return pieces[0] if len(pieces) == 1 else np.concatenate(pieces)

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(x, y)``, each ``(micro_batch_size, block_size)``.

        Ranks read disjoint, interleaved slices of the same stream, so the union of
        what all ranks see in a step is one contiguous span — the same tokens a
        single-GPU run would have seen for that step, in the same order.
        """
        tokens_per_rank = self.B * self.T
        base = self.position + self.rank * tokens_per_rank

        # One extra token: y is x shifted left by one.
        buf = self._read(base, tokens_per_rank + 1).astype(np.int64)
        x = torch.from_numpy(buf[:-1]).view(self.B, self.T)
        y = torch.from_numpy(buf[1:]).view(self.B, self.T)

        self.advance(1)

        if self.device.type == "cuda":
            # pin_memory + non_blocking overlaps the host-to-device copy with compute.
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y

    # ----------------------------------------------------------------- position

    def advance(self, micro_steps: int = 1) -> None:
        step_tokens = self.world_size * self.B * self.T * micro_steps
        new_position = self.position + step_tokens
        self.epoch += new_position // self.total_tokens
        self.position = new_position % self.total_tokens

    def set_step(self, step: int, grad_accum_steps: int) -> None:
        """Seek to the start of optimiser step ``step``.

        This is the whole resumption mechanism: no data-loader state is stored in a
        checkpoint, so a resumed run cannot disagree with the original about which
        tokens belong to which step.
        """
        micro_steps = step * grad_accum_steps
        total = micro_steps * self.world_size * self.B * self.T
        self.epoch = total // self.total_tokens
        self.position = total % self.total_tokens

    def reset(self) -> None:
        self.position = 0
        self.epoch = 0

    def __repr__(self) -> str:
        return (
            f"ShardDataLoader(split={self.split!r}, shards={len(self.shards)}, "
            f"tokens={self.total_tokens:,}, rank={self.rank}/{self.world_size}, "
            f"B={self.B}, T={self.T})"
        )


def read_meta(data_dir: str | Path) -> dict:
    """Load the ``meta.json`` written alongside the shards.

    Carries the tokenizer name and vocab size, so training can fail loudly when a
    config's tokenizer does not match the one the data was built with — otherwise
    the mismatch shows up only as a mysteriously bad loss.
    """
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        raise FileNotFoundError(f"no meta.json in {data_dir}; re-run `llmfs-prepare-data`")
    return json.loads(path.read_text())
