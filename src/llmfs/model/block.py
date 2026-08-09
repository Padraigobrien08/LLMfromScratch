"""A single pre-norm transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .cache import KVCache
from .config import ModelConfig
from .mlp import build_mlp
from .norm import build_norm


class Block(nn.Module):
    """Pre-norm residual block: ``x + attn(norm(x))`` then ``x + mlp(norm(x))``.

    Pre-norm, not the post-norm of the original tutorial code. Post-norm leaves the
    residual stream un-normalised at the point where gradients flow through it, and
    needs a warmup-heavy schedule to train at depth at all; pre-norm gives every
    layer a clean identity path and is what every modern decoder uses.
    """

    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn_norm = build_norm(cfg)
        self.attn = CausalSelfAttention(cfg, layer_idx)
        self.mlp_norm = build_norm(cfg)
        self.mlp = build_mlp(cfg)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: KVCache | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, weights = self.attn(
            self.attn_norm(x), rope=rope, cache=cache, need_weights=need_weights
        )
        x = x + attn_out
        x = x + self.mlp(self.mlp_norm(x))
        return x, weights
