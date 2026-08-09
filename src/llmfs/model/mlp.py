"""Feed-forward blocks: the GPT-2 GELU baseline and the SwiGLU variant.

SwiGLU (Shazeer, 2020) replaces the single up-projection with a gated pair: one
branch is passed through SiLU and multiplies the other. It buys consistently
lower loss at equal parameter count — the "equal parameter count" part being the
job of ``ModelConfig.mlp_hidden``, which shrinks the hidden width by 2/3 to pay
for the third projection matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class GELUMLP(nn.Module):
    """The GPT-2 feed-forward block: ``down(gelu(up(x)))``."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_hidden
        self.up_proj = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.down_proj._is_residual_proj = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.gelu(self.up_proj(x), approximate="tanh")))


class SwiGLUMLP(nn.Module):
    """``down(silu(gate(x)) * up(x))``.

    ``gate`` and ``up`` are fused into a single GEMM and split, the same trick used
    for QKV — one larger matmul beats two smaller ones on every accelerator worth
    benchmarking.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.hidden = cfg.mlp_hidden
        self.gate_up_proj = nn.Linear(cfg.n_embd, 2 * self.hidden, bias=cfg.bias)
        self.down_proj = nn.Linear(self.hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)
        self.down_proj._is_residual_proj = True  # type: ignore[attr-defined]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.dropout(self.down_proj(F.silu(gate) * up))


def build_mlp(cfg: ModelConfig) -> nn.Module:
    if cfg.mlp == "swiglu":
        return SwiGLUMLP(cfg)
    if cfg.mlp == "gelu":
        return GELUMLP(cfg)
    raise ValueError(f"unknown mlp type: {cfg.mlp!r}")
