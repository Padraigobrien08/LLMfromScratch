"""KV cache correctness.

The cache is an optimisation, so its only acceptable behaviour is to change
nothing. These tests pin that: incremental decoding must produce the same logits
as re-running the full prefix every step, for every architecture variant.

This matters more than it looks. A cache bug — a stale write offset, a mask
aligned to the wrong corner, RoPE applied twice — degrades generation quality
subtly rather than crashing, and is invisible in training metrics because
training never uses the cache.
"""

from __future__ import annotations

import pytest
import torch

from llmfs.model import GenerationConfig, KVCache

from conftest import ARCH_VARIANTS, tiny_model


@pytest.mark.parametrize("overrides", ARCH_VARIANTS)
def test_incremental_decode_matches_full_forward(overrides: dict) -> None:
    """Token-at-a-time with a cache == full forward pass, at every position."""
    model = tiny_model(**overrides)
    idx = torch.randint(0, 97, (2, 12))

    reference = model(idx, targets=idx).logits

    cache = model.make_cache(batch_size=2, max_seq_len=16)
    for t in range(idx.shape[1]):
        step = model(idx[:, t : t + 1], cache=cache).logits[:, -1]
        torch.testing.assert_close(step, reference[:, t], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("overrides", ARCH_VARIANTS)
def test_prefill_then_decode_matches_full_forward(overrides: dict) -> None:
    """The realistic path: forward the prompt in one pass, then decode one at a time.

    Exercises the ``q_len > 1, kv_len > q_len`` mask branch that a pure
    token-at-a-time test never reaches.
    """
    model = tiny_model(**overrides)
    idx = torch.randint(0, 97, (1, 12))
    reference = model(idx, targets=idx).logits

    cache = model.make_cache(batch_size=1, max_seq_len=16)
    prefill = model(idx[:, :7], cache=cache).logits[:, -1]
    torch.testing.assert_close(prefill, reference[:, 6], rtol=1e-4, atol=1e-5)

    for t in range(7, 12):
        step = model(idx[:, t : t + 1], cache=cache).logits[:, -1]
        torch.testing.assert_close(step, reference[:, t], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("overrides", ARCH_VARIANTS)
def test_cached_and_uncached_generation_agree(overrides: dict) -> None:
    """Same seed, greedy decoding: the cache must not change a single sampled token."""
    model = tiny_model(**overrides)
    prompt = torch.randint(0, 97, (1, 5))
    gen_cfg = GenerationConfig(max_new_tokens=10, temperature=0.0, top_k=None)

    cached = model.generate(prompt, gen_cfg, use_cache=True)
    uncached = model.generate(prompt, gen_cfg, use_cache=False)
    assert torch.equal(cached, uncached)


def test_cache_position_advances_once_per_forward() -> None:
    """Every layer writes at the same offset; the position moves once, after them all."""
    model = tiny_model()
    cache = model.make_cache(batch_size=1, max_seq_len=16)
    assert cache.pos == 0
    model(torch.randint(0, 97, (1, 5)), cache=cache)
    assert cache.pos == 5
    model(torch.randint(0, 97, (1, 1)), cache=cache)
    assert cache.pos == 6


def test_cache_overflow_raises() -> None:
    model = tiny_model()
    cache = model.make_cache(batch_size=1, max_seq_len=4)
    with pytest.raises(ValueError, match="overflow"):
        model(torch.randint(0, 97, (1, 5)), cache=cache)


def test_reset_allows_reuse() -> None:
    model = tiny_model()
    idx = torch.randint(0, 97, (1, 6))
    cache = model.make_cache(batch_size=1, max_seq_len=8)

    first = model(idx, cache=cache).logits
    cache.reset()
    second = model(idx, cache=cache).logits
    torch.testing.assert_close(first, second)


def test_nbytes_accounting() -> None:
    cache = KVCache(
        n_layer=4, batch_size=2, max_seq_len=128, n_kv_head=3, head_dim=16, dtype=torch.float32
    )
    # 2 tensors (K and V) * layers * B * heads * T * head_dim * 4 bytes
    assert cache.nbytes() == 2 * 4 * 2 * 3 * 128 * 16 * 4


def test_generation_beyond_block_size_rejected() -> None:
    model = tiny_model()  # block_size=32
    prompt = torch.randint(0, 97, (1, 30))
    with pytest.raises(ValueError, match="exceeds block_size"):
        model.generate(prompt, GenerationConfig(max_new_tokens=10))
