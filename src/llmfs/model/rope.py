"""Rotary positional embeddings (Su et al., 2021), implemented by hand.

RoPE encodes absolute position by rotating each 2-dimensional slice of the query
and key vectors by an angle proportional to the token's position. Because a
rotation by :math:`m\\theta` composed with the inverse of a rotation by
:math:`n\\theta` leaves an angle of :math:`(m-n)\\theta`, the resulting attention
logit depends only on the *relative* offset ``m - n``. That property is what
``tests/test_rope.py`` asserts numerically, and it is the whole reason RoPE
extrapolates better than a learned position table.

Convention
----------
This uses the "split-half" pairing popularised by GPT-NeoX and the HF Llama
implementation: dimension ``i`` is paired with dimension ``i + head_dim/2``. The
original paper pairs adjacent dimensions ``(2i, 2i+1)``. The two are related by a
fixed permutation of the head dimension, so they are mathematically equivalent as
long as queries and keys use the same one — but weights are *not* portable
between them, which matters if you ever load external checkpoints.
"""

from __future__ import annotations

import torch


class RotaryEmbedding(torch.nn.Module):
    """Precomputes and caches the cos/sin tables used to rotate q and k."""

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"RoPE requires an even head_dim, got {head_dim}")
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        cos, sin = self._build_tables(max_seq_len)
        # Buffers, not parameters: these are constants, and persistent=False keeps
        # them out of the checkpoint so max_seq_len can change on reload.
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def _build_tables(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        # inv_freq[i] = 1 / theta^(2i/head_dim) for i in [0, head_dim/2)
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        pos = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # (seq_len, head_dim/2)
        # Duplicated to (seq_len, head_dim) to match the split-half pairing.
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def forward(
        self, seq_len: int, offset: int = 0, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(cos, sin)`` of shape ``(seq_len, head_dim)`` for positions
        ``[offset, offset + seq_len)``.

        ``offset`` is what makes incremental decoding work: with a KV cache the
        model only forwards one token at a time, but that token's absolute
        position is the current cache length, not zero.
        """
        end = offset + seq_len
        if end > self.cos_cached.shape[0]:
            # Grow the table rather than fail — useful for length-extrapolation tests.
            cos, sin = self._build_tables(end)
            self.cos_cached = cos.to(self.cos_cached.device, self.cos_cached.dtype)
            self.sin_cached = sin.to(self.sin_cached.device, self.sin_cached.dtype)
        cos = self.cos_cached[offset:end]
        sin = self.sin_cached[offset:end]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map ``[x1, x2]`` (split along the last axis) to ``[-x2, x1]``.

    This is the 90-degree rotation that, combined with the cos/sin scaling below,
    implements the 2D rotation on each ``(i, i + head_dim/2)`` pair.
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``q`` and ``k`` in place of adding a positional embedding.

    Args:
        q: ``(B, n_head, T, head_dim)``
        k: ``(B, n_kv_head, T, head_dim)``
        cos, sin: ``(T, head_dim)`` from :meth:`RotaryEmbedding.forward`

    Returns:
        The rotated ``(q, k)``, same shapes and dtype as the inputs.
    """
    # (T, hd) -> (1, 1, T, hd) so it broadcasts over batch and heads. q and k can
    # have a different number of heads under GQA; broadcasting handles both.
    cos = cos[None, None, :, :].to(q.dtype)
    sin = sin[None, None, :, :].to(q.dtype)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out, k_out
