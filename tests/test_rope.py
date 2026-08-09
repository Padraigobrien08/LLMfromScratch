"""RoPE correctness.

The defining property of rotary embeddings is that the attention logit between a
query at position m and a key at position n depends only on m - n. If that holds
numerically, the implementation is right; if it does not, the model will still
train and still produce text, just worse — which is exactly the kind of bug that
survives a "does it generate English?" check.
"""

from __future__ import annotations

import pytest
import torch

from llmfs.model.rope import RotaryEmbedding, apply_rotary_emb, rotate_half


@pytest.fixture
def rope() -> RotaryEmbedding:
    return RotaryEmbedding(head_dim=16, max_seq_len=64, theta=10_000.0)


def _rotate(rope: RotaryEmbedding, x: torch.Tensor, pos: int) -> torch.Tensor:
    """Rotate a single (head_dim,) vector as if it sat at absolute position ``pos``."""
    cos, sin = rope(seq_len=1, offset=pos)
    v = x.view(1, 1, 1, -1)
    out, _ = apply_rotary_emb(v, v, cos, sin)
    return out.view(-1)


def test_relative_position_property(rope: RotaryEmbedding) -> None:
    """<R(q, m), R(k, n)> depends only on m - n."""
    q = torch.randn(16, dtype=torch.float64)
    k = torch.randn(16, dtype=torch.float64)

    for offset in (0, 1, 5, 13):
        reference = torch.dot(_rotate(rope, q, 0 + offset), _rotate(rope, k, 0))
        for m in (2, 7, 20, 40):
            shifted = torch.dot(_rotate(rope, q, m + offset), _rotate(rope, k, m))
            torch.testing.assert_close(shifted, reference, rtol=1e-5, atol=1e-6)


def test_rotation_preserves_norm(rope: RotaryEmbedding) -> None:
    """A rotation is orthogonal, so vector lengths are untouched."""
    x = torch.randn(16, dtype=torch.float64)
    for pos in (0, 3, 31, 63):
        torch.testing.assert_close(_rotate(rope, x, pos).norm(), x.norm(), rtol=1e-6, atol=1e-6)


def test_position_zero_is_identity(rope: RotaryEmbedding) -> None:
    x = torch.randn(16, dtype=torch.float64)
    torch.testing.assert_close(_rotate(rope, x, 0), x, rtol=1e-6, atol=1e-6)


def test_offset_matches_slice_of_full_table(rope: RotaryEmbedding) -> None:
    """Decoding one token at offset t must match position t of a full-sequence pass.

    This is the invariant that makes RoPE and the KV cache compose correctly.
    """
    full_cos, full_sin = rope(seq_len=32, offset=0)
    for t in (0, 1, 17, 31):
        cos, sin = rope(seq_len=1, offset=t)
        torch.testing.assert_close(cos[0], full_cos[t])
        torch.testing.assert_close(sin[0], full_sin[t])


def test_rotate_half() -> None:
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    torch.testing.assert_close(rotate_half(x), torch.tensor([-3.0, -4.0, 1.0, 2.0]))


def test_table_grows_beyond_max_seq_len() -> None:
    """Requesting a position past the cached table extends it instead of failing."""
    small = RotaryEmbedding(head_dim=8, max_seq_len=4)
    cos, sin = small(seq_len=2, offset=10)
    assert cos.shape == (2, 8) and sin.shape == (2, 8)


def test_odd_head_dim_rejected() -> None:
    with pytest.raises(ValueError, match="even head_dim"):
        RotaryEmbedding(head_dim=7, max_seq_len=8)


def test_broadcasts_across_gqa_head_counts(rope: RotaryEmbedding) -> None:
    """q and k have different head counts under GQA; the same tables must serve both."""
    cos, sin = rope(seq_len=5, offset=0)
    q = torch.randn(2, 8, 5, 16)
    k = torch.randn(2, 2, 5, 16)
    q_out, k_out = apply_rotary_emb(q, k, cos, sin)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
