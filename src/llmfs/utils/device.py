"""Device and precision selection.

Development happens on Apple silicon and real runs happen on rented CUDA boxes, so
every entrypoint has to work on both without a code change. The rules that differ
between them — bf16 support, autocast availability, TF32 matmuls — are resolved
here once rather than sprinkled through the trainer.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch

DTYPES = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def get_device(preference: str = "auto") -> torch.device:
    """Resolve ``auto`` to the best available backend, or honour an explicit choice."""
    if preference != "auto":
        return torch.device(preference)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    """Resolve a requested compute dtype, downgrading where unsupported.

    ``auto`` picks bf16 on CUDA hardware that supports it (Ampere and later) and
    fp32 everywhere else. bf16 rather than fp16 because its exponent range matches
    fp32's, so training needs no loss scaler and cannot silently diverge on an
    overflow. MPS is kept at fp32: its autocast support is incomplete enough that
    a "mixed precision" number measured there would not transfer.
    """
    if name != "auto":
        if name not in DTYPES:
            raise ValueError(f"unknown dtype {name!r}; expected one of {sorted(DTYPES)}")
        return DTYPES[name]
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


@contextlib.contextmanager
def autocast_context(device: torch.device, dtype: torch.dtype) -> Iterator[None]:
    """Autocast when it buys something, and a no-op otherwise."""
    if device.type == "cuda" and dtype in (torch.bfloat16, torch.float16):
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        yield


def enable_tf32() -> None:
    """Allow TF32 for matmuls and convolutions on Ampere+.

    Roughly a free 2-3x on fp32 matmuls in exchange for 10 bits of mantissa, which
    is far below the noise floor of a training run.
    """
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def peak_flops(device: torch.device, dtype: torch.dtype) -> float | None:
    """Advertised dense peak FLOP/s, for MFU. ``None`` when the device is unknown.

    These are the vendor's dense (non-sparse) numbers; MFU computed against them is
    directly comparable to published figures.
    """
    if device.type != "cuda":
        return None
    name = torch.cuda.get_device_name(device).lower()
    bf16 = dtype in (torch.bfloat16, torch.float16)
    table = {
        "h100": 989e12 if bf16 else 67e12,
        "a100": 312e12 if bf16 else 19.5e12,
        "l40": 181e12 if bf16 else 90.5e12,
        "4090": 165e12 if bf16 else 82.6e12,
        "a10g": 125e12 if bf16 else 31.2e12,
        "t4": 65e12 if bf16 else 8.1e12,
    }
    for key, value in table.items():
        if key in name:
            return value
    return None
