"""Norm and feed-forward components, checked against independent references."""

from __future__ import annotations

from pathlib import Path

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

    Comparing against an fp32 reference: if the reduction happened in bf16 the error
    would be an order of magnitude larger than bf16's own rounding.

    The tolerance is the test. It was `rtol=atol=2e-2`, which is wider than the error the
    bug produces — deleting the `.float()` in `RMSNorm.forward` left every test in this
    file green. The margin is now measured rather than guessed: with the fp32 reduction,
    every element is inside 4e-3 relative, which is the bf16 rounding of the *output* and
    nothing else. With a bf16 reduction the worst element needs roughly 1e-2 of slack on
    top of that, so the two regimes do not overlap.
    """
    norm = RMSNorm(768).to(torch.bfloat16)
    x = torch.randn(2, 8, 768).to(torch.bfloat16)
    reference = RMSNorm(768)(x.float())
    torch.testing.assert_close(norm(x).float(), reference, rtol=4e-3, atol=1e-4)

    # And say it the other way round, so the claim does not rest on a constant: the
    # module's output must sit closer to the fp32 reference than to what a bf16
    # reduction would have produced.
    bf16_reduction = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)) * norm.weight
    to_fp32 = (norm(x).float() - reference).abs().max()
    to_bf16 = (bf16_reduction.float() - reference).abs().max()
    assert to_fp32 < to_bf16 / 1.5, f"fp32 err {to_fp32:.5f} vs bf16-reduction err {to_bf16:.5f}"


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


def swiglu_vs_gelu(n_embd: int, n_head: int) -> float:
    common = dict(n_embd=n_embd, n_head=n_head, bias=False, dropout=0.0)
    counts = [
        sum(p.numel() for p in build_mlp(ModelConfig(mlp=mlp, **common)).parameters())
        for mlp in ("gelu", "swiglu")
    ]
    return counts[1] / counts[0]


def test_swiglu_and_gelu_have_matched_parameter_counts() -> None:
    """The 2/3 width scaling exists so the GELU-vs-SwiGLU ablation is a fair fight.

    At the 124M model's width the two blocks come out within a few percent, otherwise
    the ablation would really be measuring parameter count.
    """
    assert abs(swiglu_vs_gelu(768, 12) - 1) < 0.05


@pytest.mark.parametrize(
    "n_embd,n_head,ratio",
    [(128, 4, 1.500), (384, 6, 1.000), (512, 8, 1.125), (768, 12, 1.000), (1024, 16, 1.031)],
)
def test_the_parameter_match_depends_on_the_width(n_embd: int, n_head: int, ratio: float) -> None:
    """The match is not a property of the 2/3 rule; it is a property of the width.

    ``mlp_hidden`` rounds the scaled width up to a multiple of 256, and whether that
    rounding lands on the GELU count depends entirely on ``n_embd``. At 768 and 384 it is
    exact, at 512 SwiGLU carries 12.5% more, at 128 fully 50% more. The suite checked
    only 768 — the one width where the claim is trivially true — so "matched parameters"
    was asserted at the reproduction's width and *used* at the ablation's, which is 512.

    These are the measured ratios, pinned so a change to the rounding rule has to be
    argued for rather than absorbed.
    """
    assert swiglu_vs_gelu(n_embd, n_head) == pytest.approx(ratio, abs=5e-4)


def test_the_ablation_arm_carries_the_parameter_advantage_the_docs_disclose() -> None:
    """`mlp-swiglu` is 4.11% larger than its baseline, and the write-up has to say so."""
    from llmfs.config import CONFIG_ROOT, load_config
    from llmfs.model import Transformer

    base = Transformer(load_config(CONFIG_ROOT / "ablations" / "_base.yaml").model)
    arm = Transformer(load_config(CONFIG_ROOT / "ablations" / "mlp-swiglu.yaml").model)
    excess = (arm.num_params() / base.num_params() - 1) * 100

    assert excess == pytest.approx(4.11, abs=0.01)
    docs = (Path(__file__).resolve().parents[1] / "docs" / "ablations.md").read_text()
    assert f"{excess:.2f}%" in docs, "the write-up must state the arm's parameter advantage"


def test_swiglu_hidden_width_rounds_up() -> None:
    cfg = ModelConfig(n_embd=768, n_head=12, mlp="swiglu", mlp_hidden_multiple_of=256)
    assert cfg.mlp_hidden % 256 == 0
    assert cfg.mlp_hidden == 2048  # int(2/3 * 4 * 768) = 2048, already aligned


def test_gelu_hidden_width_is_the_plain_ratio() -> None:
    cfg = ModelConfig(n_embd=768, n_head=12, mlp="gelu", mlp_ratio=4.0)
    assert cfg.mlp_hidden == 3072
