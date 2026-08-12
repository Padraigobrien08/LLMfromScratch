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

from conftest import ARCH_VARIANTS, tiny_model
from llmfs.model import GenerationConfig, KVCache


@pytest.mark.showcase(
    pins="that token-at-a-time decoding with a cache reproduces a full forward pass at "
    "every position",
    why="Training never exercises the cache, so nothing else in the suite would catch "
    "a stale offset or a double-rotated key. The model would train perfectly and "
    "generate subtly wrong text.",
)
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


@pytest.mark.showcase(
    pins="that a single-token decode step reaches SDPA with attn_mask=None",
    why="The one test here that checks *speed* rather than an answer. Passing a mask "
    "disqualifies SDPA from its fused kernels and drops it onto the math backend; "
    "when q_len == 1 that mask is all-True and therefore pure cost. Its absence let "
    "a 30% regression sit behind a green suite for weeks.",
)
def test_decode_step_passes_no_attn_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """A single-token decode step must reach SDPA with ``attn_mask=None``.

    Not a style preference: an explicit mask disqualifies SDPA from its fused
    flash/mem-efficient kernels and silently drops it onto the math backend. When
    ``q_len == 1`` the causal mask is all-True and therefore pure cost — building it
    made cached decoding *slower* than recomputing from scratch on an RTX 4090
    (0.66x at 1024 tokens). This pins the fix so it cannot regress unnoticed.
    """
    import torch.nn.functional as F

    from llmfs.model import attention as attention_module

    seen: list[object] = []
    real_sdpa = F.scaled_dot_product_attention

    def recording_sdpa(q, k, v, attn_mask=None, **kwargs):  # type: ignore[no-untyped-def]
        if q.shape[2] == 1:  # only the decode steps, not the prefill
            seen.append(attn_mask)
        return real_sdpa(q, k, v, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(attention_module.F, "scaled_dot_product_attention", recording_sdpa)

    model = tiny_model()
    cache = model.make_cache(batch_size=1, max_seq_len=16)
    model(torch.randint(0, 97, (1, 4)), cache=cache)  # prefill
    for _ in range(3):
        model(torch.randint(0, 97, (1, 1)), cache=cache)

    assert seen, "no single-token attention calls were recorded"
    assert all(mask is None for mask in seen), "decode step built a mask it does not need"


@pytest.mark.showcase(
    pins="that a multi-token block against a filled cache still gets a real mask, and "
    "its *interior* queries agree with a full forward pass",
    why="The speculative-verification shape. The q_len == 1 fast path must not swallow "
    "it — and checking only the final position would pass regardless, because that "
    "one query is correct even with no mask at all.",
)
def test_multi_token_verify_step_still_masks() -> None:
    """The speculative-verification shape (q_len > 1 against a filled cache) still needs
    a real bottom-right aligned mask — the q_len == 1 shortcut must not swallow it."""
    model = tiny_model()
    idx = torch.randint(0, 97, (1, 9))

    # ``targets`` is what makes the model return logits for every position rather
    # than only the last — needed here because the interior queries are the point.
    full = model(idx, targets=idx).logits

    cache = model.make_cache(batch_size=1, max_seq_len=16)
    model(idx[:, :5], cache=cache)
    chunk = idx[:, 5:]
    chunked = model(chunk, targets=chunk, cache=cache).logits

    # Without the mask each of these 4 queries would see all 9 keys, so the first
    # three would attend to their own future and diverge. Only the last would agree —
    # which is exactly why checking just the final position is not enough.
    torch.testing.assert_close(chunked, full[:, 5:], atol=1e-4, rtol=1e-5)
