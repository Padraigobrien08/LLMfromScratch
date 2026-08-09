"""Normalisation layers, implemented by hand.

``RMSNorm`` (Zhang & Sennrich, 2019) drops LayerNorm's mean-centring and its bias.
It is the norm used by Llama, Gemma and Mistral: one fewer reduction over the
feature axis, and empirically no loss in quality.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, NormType


class RMSNorm(nn.Module):
    r"""Root-mean-square layer normalisation.

    .. math:: y = \frac{x}{\sqrt{\mathrm{mean}(x^2) + \epsilon}} \odot g

    The statistic is computed in float32 regardless of the input dtype: under bf16
    the sum of squares over a 768-wide feature axis loses enough precision to shift
    the norm noticeably, and this is the standard mitigation.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._norm(x.float()).type_as(x)
        return out * self.weight

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}"


class LayerNorm(nn.Module):
    """LayerNorm with an optional bias.

    ``nn.LayerNorm`` cannot drop its bias while keeping its weight, which the
    bias/no-bias ablation needs, so this wraps the functional form instead.
    """

    def __init__(self, dim: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.layer_norm(
            x, self.weight.shape, self.weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return f"dim={tuple(self.weight.shape)}, eps={self.eps}, bias={self.bias is not None}"


def build_norm(cfg: ModelConfig, dim: int | None = None, kind: NormType | None = None) -> nn.Module:
    """Instantiate the norm selected by the config."""
    dim = cfg.n_embd if dim is None else dim
    kind = cfg.norm if kind is None else kind
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=cfg.norm_eps)
    if kind == "layernorm":
        return LayerNorm(dim, eps=cfg.norm_eps, bias=cfg.bias)
    raise ValueError(f"unknown norm type: {kind!r}")
