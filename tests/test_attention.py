"""Attention correctness: causality, GQA grouping, and mask alignment."""

from __future__ import annotations

import pytest
import torch

from conftest import ARCH_VARIANTS, tiny_model
from llmfs.model import ModelConfig, Transformer
from llmfs.model.attention import CausalSelfAttention, build_causal_mask, repeat_kv


@pytest.mark.parametrize("overrides", ARCH_VARIANTS)
def test_causality(overrides: dict) -> None:
    """Changing token t must not alter the hidden states at any position < t.

    This catches an off-by-one in the mask, which a loss curve will not: a model
    that peeks one token ahead simply reports a suspiciously good loss.
    """
    model = tiny_model(**overrides)
    idx = torch.randint(0, 97, (1, 16))
    baseline = model(idx, targets=idx).logits

    perturbed = idx.clone()
    t = 9
    perturbed[0, t] = (perturbed[0, t] + 1) % 97
    after = model(perturbed, targets=idx).logits

    torch.testing.assert_close(baseline[:, :t], after[:, :t], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(baseline[:, t:], after[:, t:]), "position t should have changed"


def test_gqa_reduces_to_mha() -> None:
    """With n_kv_head == n_head, GQA must be numerically identical to plain MHA.

    Same weights, same inputs — the grouping code is the only difference, so any
    discrepancy is a bug in the repeat/grouping logic.
    """
    cfg_mha = ModelConfig(vocab_size=97, n_layer=2, n_head=4, n_embd=64, block_size=16, dropout=0.0)
    cfg_gqa = ModelConfig(
        vocab_size=97, n_layer=2, n_head=4, n_kv_head=4, n_embd=64, block_size=16, dropout=0.0
    )
    torch.manual_seed(1234)
    mha = Transformer(cfg_mha).eval()
    torch.manual_seed(1234)
    gqa = Transformer(cfg_gqa).eval()

    idx = torch.randint(0, 97, (2, 12))
    torch.testing.assert_close(mha(idx, targets=idx).logits, gqa(idx, targets=idx).logits)


def test_gqa_shrinks_kv_cache() -> None:
    """The entire point of GQA: the cache gets proportionally smaller."""
    mha = Transformer(ModelConfig(vocab_size=97, n_layer=2, n_head=8, n_embd=64, block_size=16))
    mqa = Transformer(
        ModelConfig(vocab_size=97, n_layer=2, n_head=8, n_kv_head=1, n_embd=64, block_size=16)
    )
    assert mha.make_cache(1).nbytes() == 8 * mqa.make_cache(1).nbytes()


@pytest.mark.parametrize("n_kv_head", [1, 2, 4])
def test_eager_matches_sdpa(n_kv_head: int) -> None:
    """The weight-exporting eager path must compute the same thing as the fused kernel.

    The visualizer reads its attention maps from the eager path, so if the two ever
    disagree it is showing weights the model never actually used.
    """
    cfg = ModelConfig(
        vocab_size=97,
        n_layer=2,
        n_head=4,
        n_kv_head=n_kv_head,
        n_embd=64,
        block_size=16,
        dropout=0.0,
        pos_emb="rope",
    )
    torch.manual_seed(7)
    model = Transformer(cfg).eval()
    idx = torch.randint(0, 97, (2, 11))

    sdpa = model(idx, targets=idx).logits
    for block in model.blocks:
        block.attn.cfg.attn_impl = "eager"
    eager = model(idx, targets=idx).logits

    torch.testing.assert_close(sdpa, eager, rtol=1e-4, atol=1e-5)


def test_attention_weights_are_causal_and_normalised() -> None:
    model = tiny_model(pos_emb="rope")
    idx = torch.randint(0, 97, (1, 8))
    attentions = model(idx, need_weights=True).attentions
    assert attentions is not None and len(attentions) == model.cfg.n_layer

    for weights in attentions:
        assert weights.shape == (1, model.cfg.n_head, 8, 8)
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        upper = torch.triu(torch.ones(8, 8, dtype=torch.bool), diagonal=1)
        assert weights[..., upper].abs().max() == 0, "attends to the future"


def test_build_causal_mask_is_bottom_right_aligned() -> None:
    """During decode the query block sits at the end of the key sequence.

    A top-left aligned mask (what ``is_causal=True`` gives) would let the newest
    token see only the oldest keys — the bug this function exists to prevent.
    """
    mask = build_causal_mask(q_len=2, kv_len=5, device=torch.device("cpu"))
    expected = torch.tensor(
        [
            [True, True, True, True, False],  # query at absolute position 3
            [True, True, True, True, True],  # query at absolute position 4
        ]
    )
    assert torch.equal(mask, expected)

    square = build_causal_mask(q_len=4, kv_len=4, device=torch.device("cpu"))
    assert torch.equal(square, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_repeat_kv_grouping_order() -> None:
    """Query head i must map to KV head i // n_rep — contiguous repeats, not interleaved."""
    x = torch.arange(2 * 3 * 1 * 4, dtype=torch.float32).view(2, 3, 1, 4)
    out = repeat_kv(x, n_rep=2)
    assert out.shape == (2, 6, 1, 4)
    for kv_head in range(3):
        for rep in range(2):
            torch.testing.assert_close(out[:, kv_head * 2 + rep], x[:, kv_head])


def test_qkv_projection_widths_under_gqa() -> None:
    cfg = ModelConfig(vocab_size=97, n_layer=1, n_head=8, n_kv_head=2, n_embd=64, block_size=16)
    attn = CausalSelfAttention(cfg, layer_idx=0)
    # 8 query heads + 2 key heads + 2 value heads, each 8 wide.
    assert attn.qkv_proj.out_features == (8 + 2 + 2) * 8
    assert attn.n_rep == 4


def test_invalid_head_configs_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=64, n_head=5)
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(n_embd=64, n_head=8, n_kv_head=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        ModelConfig(n_embd=64, n_head=4, n_kv_head=8)
