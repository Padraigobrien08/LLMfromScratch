"""Causal self-attention with grouped-query attention, RoPE and KV-cache support.

Grouped-query attention (Ainslie et al., 2023) is an inference-memory
optimisation: query heads are split into ``n_kv_head`` groups that share a single
key/value head. The KV cache — which dominates memory during long-context decoding
— shrinks by ``n_head / n_kv_head``, while quality degrades far less than the
equivalent reduction in query heads would cost. ``n_kv_head == n_head`` recovers
ordinary multi-head attention exactly, which is what the GQA test asserts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cache import KVCache
from .config import ModelConfig
from .rope import apply_rotary_emb


def _detect_enable_gqa() -> bool:
    """torch >= 2.5 can broadcast KV heads inside the fused kernel.

    Where it exists we use it and skip materialising the repeated K/V entirely,
    which is the difference between reading the KV cache once and reading it
    ``n_head / n_kv_head`` times. Probed by calling it rather than by parsing a
    version string, so it stays correct across builds.
    """
    try:
        q = torch.zeros(1, 2, 1, 8)
        kv = torch.zeros(1, 1, 1, 8)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except (TypeError, RuntimeError):
        return False


_SDPA_HAS_ENABLE_GQA = _detect_enable_gqa()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand ``(B, n_kv_head, T, hd)`` to ``(B, n_kv_head * n_rep, T, hd)``.

    Each KV head is repeated ``n_rep`` times contiguously so that query head ``i``
    lines up with KV head ``i // n_rep`` — matching the grouping used everywhere else.
    """
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return x[:, :, None, :, :].expand(B, n_kv, n_rep, T, hd).reshape(B, n_kv * n_rep, T, hd)


def build_causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    """Boolean mask (True = attend) for queries that start at offset ``kv_len - q_len``.

    During incremental decoding the query block sits at the *end* of the key
    sequence, so the mask must be bottom-right aligned. ``is_causal=True`` in SDPA
    is top-left aligned, which silently produces the wrong mask whenever
    ``q_len != kv_len`` — a classic KV-cache bug, and why this is explicit.
    """
    offset = kv_len - q_len
    q_idx = torch.arange(q_len, device=device).unsqueeze(1) + offset
    k_idx = torch.arange(kv_len, device=device).unsqueeze(0)
    return k_idx <= q_idx


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_kv_groups

        # One fused GEMM for Q, K and V instead of three. Under GQA the three
        # slices have different widths, hence the explicit split sizes below.
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        self.qkv_proj = nn.Linear(cfg.n_embd, self.q_size + 2 * self.kv_size, bias=cfg.bias)
        self.out_proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)

        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

        # Marks the projection whose init is downscaled by 1/sqrt(2 * n_layer).
        self.out_proj._is_residual_proj = True  # type: ignore[attr-defined]

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
        cache: KVCache | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Args:
            x: ``(B, T, C)``
            rope: ``(cos, sin)`` tables already sliced to this call's positions.
            cache: if given, new keys/values are appended and the full history attended over.
            need_weights: also return the ``(B, n_head, T, kv_len)`` attention
                probabilities. Forces the eager path.

        Returns:
            ``(output, attn_weights_or_None)``
        """
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)

        # RoPE is applied *before* the cache write, so cached keys are already
        # rotated with their own absolute positions and never need re-rotating.
        if rope is not None:
            cos, sin = rope
            q, k = apply_rotary_emb(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(self.layer_idx, k, v)

        kv_len = k.shape[2]
        use_eager = need_weights or self.cfg.attn_impl == "eager"

        if use_eager:
            out, weights = self._eager_attention(q, k, v, kv_len)
        else:
            out, weights = self._sdpa_attention(q, k, v, T, kv_len), None

        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out, weights

    def _sdpa_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, q_len: int, kv_len: int
    ) -> torch.Tensor:
        dropout_p = self.attn_dropout_p if self.training else 0.0

        # Which of these three branches we take is the single biggest lever on decode
        # speed, because passing an explicit ``attn_mask`` disqualifies SDPA from its
        # fused flash/mem-efficient kernels and drops it onto the math backend. So a
        # mask is built only when one is genuinely needed.
        if q_len == kv_len:
            # No cache: query and key sequences coincide, so SDPA's own causal flag
            # is both correct and fused.
            attn_mask, is_causal = None, True
        elif q_len == 1:
            # Single-token decode step. Every cached key precedes this query, so the
            # causal mask is all-True and carries no information — building it would
            # cost a mask allocation per layer per token *and* forfeit the fused
            # kernel, which is why the cache used to lose to plain recomputation.
            attn_mask, is_causal = None, False
        else:
            # Prefill against a partly-filled cache, or speculative verification of
            # several draft tokens at once: the query block sits at the end of the
            # key sequence and needs a real bottom-right aligned mask.
            attn_mask = build_causal_mask(q_len, kv_len, q.device)[None, None]
            is_causal = False

        if _SDPA_HAS_ENABLE_GQA and self.n_rep > 1:
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                enable_gqa=True,
            )

        return F.scaled_dot_product_attention(
            q,
            repeat_kv(k, self.n_rep),
            repeat_kv(v, self.n_rep),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

    def _eager_attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kv_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialise the attention matrix. Slower, but exportable for the visualizer."""
        q_len = q.shape[2]
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = build_causal_mask(q_len, kv_len, q.device)[None, None]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).type_as(q)
        out = F.dropout(weights, p=self.attn_dropout_p, training=self.training) @ v
        return out, weights
