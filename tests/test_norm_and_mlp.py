"""Norm and feed-forward components, checked against independent references."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from llmfs.model import ModelConfig, build_mlp
from llmfs.model.norm import LayerNorm, RMSNorm, build_norm


def test_rmsnorm_matches_reference_formula() -> None:
    """Checked against the definition, not against another library."""
    dim, eps = 32, 1e-5
    norm = RMSNorm(dim, eps=eps)
    with torch.no_grad():
        norm.weight.normal_()

    x = torch.randn(4, 7, dim)
    expected = x / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps) * norm.weight
    torch.testing.assert_close(norm(x), expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_does_not_centre() -> None:
    """The distinguishing property: RMSNorm rescales but does not remove the mean.

    Compared directly against LayerNorm on the same off-centre input: LayerNorm
    subtracts the mean and lands on zero, RMSNorm only divides by the RMS and so
    keeps the offset, scaled.
    """
    x = torch.randn(2, 3, 64) + 5.0  # deliberately off-centre

    rms_mean = RMSNorm(64)(x).mean().abs()
    ln_mean = LayerNorm(64, bias=True)(x).mean().abs()

    assert ln_mean < 1e-5, "LayerNorm should centre"
    # E[x] = 5 and RMS = sqrt(1 + 25), so the surviving offset is ~5/sqrt(26) ≈ 0.98.
    torch.testing.assert_close(rms_mean, torch.tensor(5.0 / 26**0.5), rtol=0.05, atol=0.05)


def test_rmsnorm_is_scale_invariant() -> None:
    norm = RMSNorm(64)
    x = torch.randn(2, 3, 64)
    torch.testing.assert_close(norm(x), norm(x * 7.0), rtol=1e-4, atol=1e-5)


def test_rmsnorm_statistic_computed_in_fp32() -> None:
    """Under bf16 the sum of squares must not be accumulated in bf16.

    Comparing against an fp32 reference: if the reduction happened in bf16 the
    error would be an order of magnitude larger than bf16's own rounding.
    """
    norm = RMSNorm(768).to(torch.bfloat16)
    x = torch.randn(2, 8, 768).to(torch.bfloat16)
    reference = RMSNorm(768)(x.float())
    torch.testing.assert_close(norm(x).float(), reference, rtol=2e-2, atol=2e-2)


def test_layernorm_matches_torch() -> None:
    ours = LayerNorm(32, eps=1e-5, bias=True)
    theirs = torch.nn.LayerNorm(32, eps=1e-5)
    with torch.no_grad():
        ours.weight.normal_()
        ours.bias.normal_()
        theirs.weight.copy_(ours.weight)
        theirs.bias.copy_(ours.bias)

    x = torch.randn(4, 7, 32)
    torch.testing.assert_close(ours(x), theirs(x))


def test_layernorm_without_bias() -> None:
    norm = LayerNorm(32, bias=False)
    assert norm.bias is None
    assert norm(torch.randn(2, 32)).shape == (2, 32)


def test_build_norm_dispatch() -> None:
    cfg = ModelConfig(n_embd=64, n_head=4)
    assert isinstance(build_norm(cfg, kind="rmsnorm"), RMSNorm)
    assert isinstance(build_norm(cfg, kind="layernorm"), LayerNorm)
    with pytest.raises(ValueError, match="unknown norm"):
        build_norm(cfg, kind="batchnorm")  # type: ignore[arg-type]


def test_swiglu_matches_reference_formula() -> None:
    cfg = ModelConfig(n_embd=64, n_head=4, mlp="swiglu", bias=False, dropout=0.0)
    mlp = build_mlp(cfg).eval()

    x = torch.randn(2, 5, 64)
    gate_w, up_w = mlp.gate_up_proj.weight.chunk(2, dim=0)
    expected = F.linear(F.silu(F.linear(x, gate_w)) * F.linear(x, up_w), mlp.down_proj.weight)
    torch.testing.assert_close(mlp(x), expected, rtol=1e-5, atol=1e-6)


def test_swiglu_and_gelu_have_matched_parameter_counts() -> None:
    """The 2/3 width scaling exists so the GELU-vs-SwiGLU ablation is a fair fight.

    At the 124M model's width the two blocks must come out within a few percent,
    otherwise the ablation is really measuring parameter count.
    """
    common = dict(n_embd=768, n_head=12, bias=False, dropout=0.0)
    gelu = build_mlp(ModelConfig(mlp="gelu", **common))
    swiglu = build_mlp(ModelConfig(mlp="swiglu", **common))

    n_gelu = sum(p.numel() for p in gelu.parameters())
    n_swiglu = sum(p.numel() for p in swiglu.parameters())
    assert abs(n_swiglu - n_gelu) / n_gelu < 0.05, (
        f"SwiGLU has {n_swiglu:,} params vs GELU's {n_gelu:,} — the comparison is unfair"
    )


def test_swiglu_hidden_width_rounds_up() -> None:
    cfg = ModelConfig(n_embd=768, n_head=12, mlp="swiglu", mlp_hidden_multiple_of=256)
    assert cfg.mlp_hidden % 256 == 0
    assert cfg.mlp_hidden == 2048  # int(2/3 * 4 * 768) = 2048, already aligned


def test_gelu_hidden_width_is_the_plain_ratio() -> None:
    cfg = ModelConfig(n_embd=768, n_head=12, mlp="gelu", mlp_ratio=4.0)
    assert cfg.mlp_hidden == 3072
