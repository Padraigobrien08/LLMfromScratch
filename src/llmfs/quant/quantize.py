"""Weight-only quantization, implemented by hand.

During single-stream decoding a transformer is **memory-bound, not compute-bound**:
every token requires reading the entire weight matrix from HBM, and the arithmetic per
weight is one multiply-accumulate. So the thing to shrink is the bytes moved, and
weight-only quantization does exactly that — store weights at 8 or 4 bits, keep
activations in bf16, and dequantize on the way into the matmul.

What that buys, and what it does not
------------------------------------
The memory saving is unconditional and large: 4-bit weights are a quarter the size of
fp16. The *speed* saving is not, and this implementation is honest about why.
Dequantizing in PyTorch means materialising an fp16 copy of the weight before calling
the matmul, so the bytes read from HBM go *up*, not down. A real speedup needs a fused
kernel that reads packed integers and dequantizes inside the matmul's inner loop —
which is what Marlin, GPTQ's CUDA kernels and bitsandbytes provide, and what a Triton
kernel would do here. The benchmarks report both numbers rather than only the
flattering one.

Scheme
------
Asymmetric per-group affine quantization:

.. math:: w \\approx (q - z) \\cdot s

with ``s`` and ``z`` computed per group of ``group_size`` consecutive input features.
Asymmetric (with a zero point) rather than symmetric because a weight distribution is
rarely exactly centred, and at 4 bits — 16 levels — wasting one to a phantom symmetry
is expensive. Grouping along the input dimension because a scale shared across many
features is dominated by the largest outlier among them, which is the main reason naive
4-bit quantization destroys quality.

``group_size=-1`` means one group per row: a scale per output channel, which is the
coarsest granularity this API expresses. It is not per-tensor — a single scale for the
whole matrix is coarser still, and is not implemented here.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class QuantConfig:
    bits: int = 8
    """8 or 4. 4-bit weights are packed two-per-byte."""
    group_size: int = 128
    """Input features sharing one scale/zero. ``-1`` means one group per output row.
    Smaller groups cost more metadata but tolerate outliers better; 128 is the usual
    compromise and what GPTQ/AWQ default to."""
    skip: tuple[str, ...] = ("lm_head",)
    """Module name substrings to leave in full precision.

    ``lm_head`` is skipped by default and it matters more than it looks: with tied
    embeddings it *is* the token embedding, so quantizing it degrades both the input
    representation and the output logits at once, and it is the layer whose errors
    land directly on the sampled distribution."""

    def __post_init__(self) -> None:
        if self.bits not in (4, 8):
            raise ValueError(f"bits must be 4 or 8, got {self.bits}")
        if self.group_size == 0 or self.group_size < -1:
            raise ValueError(f"group_size must be positive or -1, got {self.group_size}")


def quantize_tensor(
    weight: torch.Tensor, bits: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize ``(out_features, in_features)`` to integer codes plus scales/zeros.

    Returns ``(codes, scales, zeros)`` where codes are ``uint8`` in ``[0, 2**bits)``
    and scales/zeros are ``(out_features, n_groups)``.
    """
    out_features, in_features = weight.shape
    groups = in_features if group_size == -1 else group_size
    if in_features % groups != 0:
        raise ValueError(f"in_features ({in_features}) must be divisible by group_size ({groups})")

    grouped = weight.reshape(out_features, -1, groups).float()
    w_min = grouped.amin(dim=-1, keepdim=True)
    w_max = grouped.amax(dim=-1, keepdim=True)

    levels = 2**bits - 1
    # A constant group would give scale 0 and a division by zero on dequantize.
    scales = ((w_max - w_min) / levels).clamp(min=1e-8)
    zeros = torch.round(-w_min / scales)

    codes = torch.clamp(torch.round(grouped / scales) + zeros, 0, levels)
    return (
        codes.to(torch.uint8).reshape(out_features, in_features),
        scales.squeeze(-1).contiguous(),
        zeros.squeeze(-1).contiguous(),
    )


def dequantize_tensor(
    codes: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor, group_size: int
) -> torch.Tensor:
    out_features, in_features = codes.shape
    groups = in_features if group_size == -1 else group_size
    grouped = codes.reshape(out_features, -1, groups).float()
    return ((grouped - zeros.unsqueeze(-1)) * scales.unsqueeze(-1)).reshape(
        out_features, in_features
    )


def pack_4bit(codes: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit codes into each byte along the last dimension.

    Without this, 4-bit "quantization" would still occupy a byte per weight and save
    nothing — the whole point is the bytes on the wire.
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"cannot pack an odd number of codes: {codes.shape[-1]}")
    low, high = codes[..., 0::2], codes[..., 1::2]
    return (low | (high << 4)).to(torch.uint8)


def unpack_4bit(packed: torch.Tensor, in_features: int) -> torch.Tensor:
    out = torch.empty(*packed.shape[:-1], in_features, dtype=torch.uint8, device=packed.device)
    out[..., 0::2] = packed & 0x0F
    out[..., 1::2] = (packed >> 4) & 0x0F
    return out


class QuantLinear(nn.Module):
    """A ``nn.Linear`` whose weight is stored quantized and dequantized per call.

    The dequantized weight is deliberately *not* cached. Caching it would make this
    fast and pointless: the memory saving is the entire benefit, and holding an fp16
    copy alongside the codes gives back more than quantization saved.
    """

    def __init__(self, linear: nn.Linear, cfg: QuantConfig) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.bits = cfg.bits
        self.group_size = cfg.group_size

        codes, scales, zeros = quantize_tensor(linear.weight.data, cfg.bits, cfg.group_size)
        if cfg.bits == 4:
            codes = pack_4bit(codes)

        # Buffers, not parameters: quantized weights are not trainable here. This is
        # post-training quantization, not quantization-aware training.
        self.register_buffer("codes", codes)
        self.register_buffer("scales", scales.to(linear.weight.dtype))
        self.register_buffer("zeros", zeros.to(linear.weight.dtype))
        self.bias = (
            nn.Parameter(linear.bias.data.clone(), requires_grad=False)
            if linear.bias is not None
            else None
        )

    def dequantized_weight(self) -> torch.Tensor:
        codes = unpack_4bit(self.codes, self.in_features) if self.bits == 4 else self.codes
        return dequantize_tensor(codes, self.scales.float(), self.zeros.float(), self.group_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantized_weight().to(x.dtype)
        return F.linear(x, weight, self.bias)

    def weight_bytes(self) -> int:
        return (
            self.codes.numel() * self.codes.element_size()
            + self.scales.numel() * self.scales.element_size()
            + self.zeros.numel() * self.zeros.element_size()
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}"
        )


# --------------------------------------------------------------------- surgery


def quantize_model(model: nn.Module, cfg: QuantConfig | None = None) -> dict[str, object]:
    """Replace every eligible ``nn.Linear`` in ``model`` with a :class:`QuantLinear`.

    Mutates ``model`` in place and returns a summary of what changed, including the
    real byte counts before and after — the number the memory claim rests on.
    """
    cfg = cfg or QuantConfig()

    # With tied embeddings, lm_head.weight *is* tok_emb.weight. Replacing lm_head with
    # a QuantLinear stores a quantized copy but leaves nn.Embedding holding the
    # original fp32 tensor, so memory goes UP — measured: 196 MiB unquantized head vs
    # 217 MiB "quantized". Quantizing it for real needs a QuantEmbedding sharing one
    # set of codes with the head, which is not implemented. Refusing is better than
    # reporting a compression ratio that is worse than doing nothing.
    head, embedding = getattr(model, "lm_head", None), getattr(model, "tok_emb", None)
    tied = head is not None and embedding is not None and head.weight is embedding.weight
    if tied and not any("lm_head" in pattern for pattern in cfg.skip):
        raise ValueError(
            "cannot quantize lm_head while embeddings are tied: it shares its weight "
            "with tok_emb, so a quantized copy is added rather than substituted and the "
            "model gets larger (measured: 196 -> 217 MiB). Keep 'lm_head' in "
            "QuantConfig.skip, or train with tie_embeddings=False."
        )

    before = after = 0
    replaced: list[str] = []
    skipped: list[str] = []

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name

            if any(pattern in full for pattern in cfg.skip):
                skipped.append(full)
                before += child.weight.numel() * child.weight.element_size()
                after += child.weight.numel() * child.weight.element_size()
                continue
            # A group must divide the input dimension; skip rather than silently
            # changing the scheme for one layer and making the report inconsistent.
            groups = child.in_features if cfg.group_size == -1 else cfg.group_size
            if child.in_features % groups != 0 or (cfg.bits == 4 and child.in_features % 2 != 0):
                skipped.append(f"{full} (shape {child.in_features} incompatible)")
                before += child.weight.numel() * child.weight.element_size()
                after += child.weight.numel() * child.weight.element_size()
                continue

            before += child.weight.numel() * child.weight.element_size()
            quant = QuantLinear(child, cfg)
            after += quant.weight_bytes()
            setattr(module, child_name, quant)
            replaced.append(full)

    return {
        "bits": cfg.bits,
        "group_size": cfg.group_size,
        "replaced": len(replaced),
        "skipped": skipped,
        "linear_bytes_before": before,
        "linear_bytes_after": after,
        "compression": before / after if after else 1.0,
    }


def model_memory_bytes(model: nn.Module) -> int:
    """Total bytes of parameters and buffers — what actually sits in memory."""
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    total += sum(b.numel() * b.element_size() for b in model.buffers())
    return total
