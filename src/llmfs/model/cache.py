"""A static, preallocated key/value cache for incremental decoding.

Without a cache, generating token ``n`` re-runs attention over all ``n`` previous
tokens, making a full sample quadratic in length. With one, each step is linear:
the per-layer keys and values for tokens already seen are kept and only the new
token's k/v are computed.

The cache is allocated once at its maximum size rather than grown by
concatenation. Repeated ``torch.cat`` reallocates the whole tensor every step,
which both costs bandwidth and fragments the allocator — the exact thing the
inference benchmarks in ``bench/`` are meant to measure away.
"""

from __future__ import annotations

import torch


class KVCache:
    """Per-layer key/value storage for one generation session.

    Shapes are ``(batch, n_kv_head, max_seq_len, head_dim)`` per layer.
    """

    def __init__(
        self,
        n_layer: int,
        batch_size: int,
        max_seq_len: int,
        n_kv_head: int,
        head_dim: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.n_layer = n_layer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.n_kv_head = n_kv_head
        self.head_dim = head_dim
        self.device = torch.device(device)
        self.dtype = dtype

        shape = (batch_size, n_kv_head, max_seq_len, head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.values = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self._pos = 0

    @property
    def pos(self) -> int:
        """Number of tokens currently cached (also the next write offset)."""
        return self._pos

    def update(
        self, layer_idx: int, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write this layer's new k/v at the current offset and return the full history.

        Args:
            layer_idx: which block is writing.
            k, v: ``(B, n_kv_head, T_new, head_dim)``.

        Returns:
            Views over ``(B, n_kv_head, pos + T_new, head_dim)``.

        Note that ``pos`` is *not* advanced here — every layer writes at the same
        offset for a given forward pass. :meth:`advance` is called once by the
        model after the last block.
        """
        t_new = k.shape[2]
        end = self._pos + t_new
        if end > self.max_seq_len:
            raise ValueError(
                f"KV cache overflow: writing {t_new} tokens at position {self._pos} "
                f"exceeds max_seq_len={self.max_seq_len}"
            )
        self.keys[layer_idx][:, :, self._pos : end] = k.to(self.dtype)
        self.values[layer_idx][:, :, self._pos : end] = v.to(self.dtype)
        return self.keys[layer_idx][:, :, :end], self.values[layer_idx][:, :, :end]

    def advance(self, n_tokens: int) -> None:
        self._pos += n_tokens

    def rewind_to(self, pos: int) -> None:
        """Discard everything cached beyond ``pos``.

        Speculative decoding needs this: the target scores ``k`` draft tokens in one
        pass, which writes ``k`` entries into the cache, and any draft it rejects must
        be undone before the next iteration.

        Because the cache is preallocated, rolling back is only a move of the write
        offset — attention reads ``[:pos]``, so stale entries past it are never seen and
        are overwritten by the next write. That is the property that makes rejection
        cheap; a cache built by concatenation would have to reallocate here.
        """
        if not 0 <= pos <= self._pos:
            raise ValueError(f"cannot rewind to {pos}: cache holds {self._pos} tokens")
        self._pos = pos

    def reset(self) -> None:
        self._pos = 0

    def nbytes(self) -> int:
        """Allocated cache size in bytes — the headline number for memory profiling."""
        per_tensor = (
            self.batch_size * self.n_kv_head * self.max_seq_len * self.head_dim
        ) * torch.empty((), dtype=self.dtype).element_size()
        return 2 * self.n_layer * per_tensor

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={self.n_layer}, batch={self.batch_size}, "
            f"max_seq_len={self.max_seq_len}, n_kv_head={self.n_kv_head}, "
            f"head_dim={self.head_dim}, dtype={self.dtype}, "
            f"pos={self._pos}, size={self.nbytes() / 2**20:.1f} MiB)"
        )
