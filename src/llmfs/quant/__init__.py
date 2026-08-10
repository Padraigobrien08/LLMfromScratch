"""Weight-only quantization, hand-implemented.

The memory saving is real and unconditional; the speed saving needs a fused kernel,
which PyTorch's dequantize-then-matmul does not provide. Both are reported.
"""

from .quantize import (
    QuantConfig,
    QuantLinear,
    dequantize_tensor,
    model_memory_bytes,
    pack_4bit,
    quantize_model,
    quantize_tensor,
    unpack_4bit,
)

__all__ = [
    "QuantConfig",
    "QuantLinear",
    "dequantize_tensor",
    "model_memory_bytes",
    "pack_4bit",
    "quantize_model",
    "quantize_tensor",
    "unpack_4bit",
]
