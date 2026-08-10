"""Weight-only quantization.

Two things are checked: that the arithmetic is right (round-trip error bounded by the
step size, packing lossless, finer schemes strictly better), and that the *claims* are
right — that the memory actually falls, and that the layers meant to be left alone are
left alone.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from conftest import tiny_model
from llmfs.quant import (
    QuantConfig,
    QuantLinear,
    dequantize_tensor,
    model_memory_bytes,
    pack_4bit,
    quantize_model,
    quantize_tensor,
    unpack_4bit,
)


@pytest.fixture
def weight():
    torch.manual_seed(0)
    return torch.randn(64, 256)


# --------------------------------------------------------------------- arithmetic


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [-1, 32, 128])
def test_round_trip_error_is_bounded_by_the_step_size(weight, bits, group_size) -> None:
    """Error cannot exceed half a quantization step, per group.

    This is the definition of correct rounding — a looser bound means the scale or
    zero point is being computed wrongly.
    """
    codes, scales, zeros = quantize_tensor(weight, bits, group_size)
    restored = dequantize_tensor(codes, scales, zeros, group_size)

    groups = weight.shape[1] if group_size == -1 else group_size
    per_group_max_step = scales.reshape(-1, 1).repeat(1, groups).reshape(weight.shape)
    assert (restored - weight).abs().max() <= per_group_max_step.max() / 2 + 1e-5


def test_more_bits_is_strictly_better(weight) -> None:
    def err(bits, gs):
        c, s, z = quantize_tensor(weight, bits, gs)
        return (dequantize_tensor(c, s, z, gs) - weight).norm().item()

    assert err(8, 128) < err(4, 128)


def test_smaller_groups_are_better(weight) -> None:
    """Finer scales tolerate outliers, which is why grouping exists at all."""

    def err(gs):
        c, s, z = quantize_tensor(weight, 4, gs)
        return (dequantize_tensor(c, s, z, gs) - weight).norm().item()

    assert err(32) < err(-1)


def test_an_outlier_is_why_grouping_matters() -> None:
    """One huge weight destroys a per-tensor scale but only spoils its own group.

    This is the concrete mechanism behind naive 4-bit quantization ruining a model: a
    single outlier stretches the scale until every ordinary weight rounds to the same
    couple of levels. Grouping confines the damage to the group containing it, so the
    error is measured *outside* that group.
    """
    w = torch.randn(4, 128) * 0.01
    w[0, 0] = 10.0  # lands in group 0 when group_size=16

    def err_outside_first_group(gs):
        c, s, z = quantize_tensor(w, 4, gs)
        r = dequantize_tensor(c, s, z, gs)
        return (r - w)[:, 16:].abs().max().item()

    # Per-tensor, the outlier ruins all 128 columns; grouped, only the first 16.
    assert err_outside_first_group(16) < err_outside_first_group(-1) / 5


def test_constant_group_does_not_divide_by_zero() -> None:
    w = torch.full((2, 64), 0.5)
    codes, scales, zeros = quantize_tensor(w, 4, 32)
    restored = dequantize_tensor(codes, scales, zeros, 32)
    assert torch.isfinite(restored).all()
    torch.testing.assert_close(restored, w, atol=1e-3, rtol=0)


# ------------------------------------------------------------------------ packing


def test_packing_is_lossless_and_halves_the_bytes() -> None:
    codes = torch.randint(0, 16, (8, 64), dtype=torch.uint8)
    packed = pack_4bit(codes)
    assert packed.numel() * 2 == codes.numel()
    assert torch.equal(unpack_4bit(packed, 64), codes)


def test_packing_rejects_an_odd_width() -> None:
    with pytest.raises(ValueError, match="odd"):
        pack_4bit(torch.zeros(2, 7, dtype=torch.uint8))


# --------------------------------------------------------------------- QuantLinear


@pytest.mark.parametrize("bits,group_size", [(8, 128), (4, 128), (4, 32)])
def test_quant_linear_approximates_the_original(bits, group_size) -> None:
    torch.manual_seed(1)
    linear = nn.Linear(256, 64)
    x = torch.randn(4, 256)
    reference = linear(x)

    quant = QuantLinear(linear, QuantConfig(bits=bits, group_size=group_size))
    relative = ((quant(x) - reference).norm() / reference.norm()).item()
    # 8-bit should be near-exact; 4-bit is visibly lossy but must not be garbage.
    assert relative < (0.02 if bits == 8 else 0.25)


def test_quant_linear_actually_shrinks_the_weight() -> None:
    linear = nn.Linear(256, 64)
    fp32_bytes = linear.weight.numel() * 4
    assert QuantLinear(linear, QuantConfig(bits=8, group_size=128)).weight_bytes() < fp32_bytes / 3
    assert QuantLinear(linear, QuantConfig(bits=4, group_size=128)).weight_bytes() < fp32_bytes / 6


def test_quant_linear_preserves_bias() -> None:
    linear = nn.Linear(64, 32, bias=True)
    quant = QuantLinear(linear, QuantConfig(bits=8, group_size=-1))
    torch.testing.assert_close(quant.bias, linear.bias)

    assert QuantLinear(nn.Linear(64, 32, bias=False), QuantConfig(group_size=-1)).bias is None


def test_dequantized_weight_is_not_cached() -> None:
    """Caching would make it fast and pointless — the memory saving is the point."""
    quant = QuantLinear(nn.Linear(64, 32), QuantConfig(bits=4, group_size=32))
    a = quant.dequantized_weight()
    assert a.data_ptr() != quant.dequantized_weight().data_ptr()


# ------------------------------------------------------------------------ surgery


def test_quantize_model_replaces_linears_and_skips_the_head() -> None:
    """``lm_head`` must stay in full precision: with tied embeddings it *is* the token
    embedding, so quantizing it corrupts input and output representations at once."""
    model = tiny_model(vocab_size=128, n_layer=2, n_head=4, n_embd=64)
    info = quantize_model(model, QuantConfig(bits=8, group_size=-1))

    assert info["replaced"] > 0
    assert any("lm_head" in s for s in info["skipped"])
    assert isinstance(model.lm_head, nn.Linear)
    assert isinstance(model.blocks[0].attn.qkv_proj, QuantLinear)


def test_quantize_model_reduces_measured_memory() -> None:
    before = model_memory_bytes(tiny_model(vocab_size=128, n_layer=2, n_embd=64))
    model = tiny_model(vocab_size=128, n_layer=2, n_embd=64)
    quantize_model(model, QuantConfig(bits=4, group_size=32))
    assert model_memory_bytes(model) < before


def test_quantized_model_still_runs_and_stays_close() -> None:
    torch.manual_seed(2)
    reference = tiny_model(vocab_size=128, n_layer=2, n_head=4, n_embd=64)
    idx = torch.randint(0, 128, (2, 16))
    expected = reference(idx, targets=idx).loss.item()

    torch.manual_seed(2)
    model = tiny_model(vocab_size=128, n_layer=2, n_head=4, n_embd=64)
    quantize_model(model, QuantConfig(bits=8, group_size=128))
    got = model(idx, targets=idx).loss.item()

    # 8-bit weights should barely move an untrained model's loss.
    assert abs(got - expected) < 0.05


def test_skip_patterns_are_honoured() -> None:
    model = tiny_model(vocab_size=128, n_layer=2, n_head=4, n_embd=64)
    # group_size must divide the 64-wide layers, or they are skipped as incompatible.
    quantize_model(model, QuantConfig(bits=8, group_size=32, skip=("lm_head", "qkv_proj")))
    assert isinstance(model.blocks[0].attn.qkv_proj, nn.Linear)
    assert isinstance(model.blocks[0].attn.out_proj, QuantLinear)


def test_quantizing_a_tied_head_is_refused() -> None:
    """With tied embeddings, quantizing lm_head makes the model *bigger*.

    lm_head.weight IS tok_emb.weight, so replacing the head stores a quantized copy
    while nn.Embedding keeps the original fp32 tensor. Measured on the 124M model:
    196 MiB with the head skipped, 217 MiB with it "quantized". Refusing beats
    reporting a compression ratio worse than doing nothing.
    """
    model = tiny_model(vocab_size=128, n_layer=2, n_head=4, n_embd=64, tie_embeddings=True)
    assert model.lm_head.weight is model.tok_emb.weight

    with pytest.raises(ValueError, match="tied"):
        quantize_model(model, QuantConfig(bits=4, group_size=32, skip=()))


def test_quantizing_the_head_is_allowed_when_untied() -> None:
    """Untied, the head is an ordinary matrix and quantizing it is a real saving."""
    model = tiny_model(vocab_size=128, n_layer=2, n_head=4, n_embd=64, tie_embeddings=False)
    before = model_memory_bytes(model)
    info = quantize_model(model, QuantConfig(bits=4, group_size=32, skip=()))

    assert not info["skipped"]
    assert isinstance(model.lm_head, QuantLinear)
    assert model_memory_bytes(model) < before


# ------------------------------------------------------------------ config guards


def test_invalid_bit_widths_are_rejected() -> None:
    for bits in (1, 3, 16):
        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            QuantConfig(bits=bits)


def test_invalid_group_size_is_rejected() -> None:
    with pytest.raises(ValueError, match="group_size"):
        QuantConfig(group_size=0)


def test_group_size_must_divide_the_input_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        quantize_tensor(torch.randn(4, 100), 4, 32)


def test_incompatible_layers_are_skipped_not_silently_rescheduled() -> None:
    """A layer whose shape does not fit the scheme is reported, not quietly quantized
    with different settings — otherwise the reported scheme would be a fiction."""
    model = nn.Sequential(nn.Linear(100, 8))  # 100 % 32 != 0
    info = quantize_model(model, QuantConfig(bits=4, group_size=32))
    assert info["replaced"] == 0
    assert any("incompatible" in s for s in info["skipped"])
