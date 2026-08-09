"""The decoder-only transformer.

One class covers both the GPT-2 baseline and the modern Llama-style stack; which
one you get is entirely determined by :class:`ModelConfig`. That is deliberate —
an ablation that swaps LayerNorm for RMSNorm should differ from its baseline by a
single line of YAML, not by a forked model file that can silently drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .cache import KVCache
from .config import GenerationConfig, ModelConfig
from .norm import build_norm
from .rope import RotaryEmbedding


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    attentions: list[torch.Tensor] | None = None
    """Per-layer ``(B, n_head, T, kv_len)`` attention probabilities, when requested."""


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = (
            nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos_emb == "learned" else None
        )
        self.rope = (
            RotaryEmbedding(cfg.head_dim, cfg.block_size, theta=cfg.rope_theta)
            if cfg.pos_emb == "rope"
            else None
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg, i) for i in range(cfg.n_layer)])
        self.final_norm = build_norm(cfg)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            # Tying is a real architectural choice, not just a saving: for a 124M
            # model the embedding matrix is ~31% of all parameters, so untying
            # changes the parameter budget as much as adding two layers would.
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        if cfg.scale_residual_init:
            self._scale_residual_init()

    # ------------------------------------------------------------------ init

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)

    def _scale_residual_init(self) -> None:
        """GPT-2's ``1/sqrt(2 * n_layer)`` downscaling of residual output projections.

        Each block writes twice into the residual stream (attention and MLP), so
        without this the variance of the stream grows linearly with depth and the
        late layers start out saturated.
        """
        scale = 1.0 / math.sqrt(2 * self.cfg.n_layer)
        for module in self.modules():
            if getattr(module, "_is_residual_proj", False):
                with torch.no_grad():
                    module.weight.mul_(scale)

    # --------------------------------------------------------------- forward

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        cache: KVCache | None = None,
        need_weights: bool = False,
    ) -> ModelOutput:
        """
        Args:
            idx: ``(B, T)`` token ids.
            targets: ``(B, T)`` next-token labels; ``-1`` is ignored in the loss.
            cache: KV cache for incremental decoding. Positions continue from ``cache.pos``.
            need_weights: return per-layer attention probabilities (forces the eager path).
        """
        B, T = idx.shape
        offset = cache.pos if cache is not None else 0
        if offset + T > self.cfg.block_size:
            raise ValueError(
                f"sequence of length {offset + T} exceeds block_size={self.cfg.block_size}"
            )

        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            pos = torch.arange(offset, offset + T, device=idx.device)
            x = x + self.pos_emb(pos)
        x = self.drop(x)

        rope = None
        if self.rope is not None:
            rope = self.rope(T, offset=offset, device=idx.device)

        attentions: list[torch.Tensor] | None = [] if need_weights else None
        for block in self.blocks:
            x, weights = block(x, rope=rope, cache=cache, need_weights=need_weights)
            if attentions is not None and weights is not None:
                attentions.append(weights)

        if cache is not None:
            # Advanced once per forward pass, after every layer has written at
            # the same offset.
            cache.advance(T)

        x = self.final_norm(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        else:
            # Inference: only the last position's logits are needed, and skipping
            # the rest avoids a (B, T, 50k) tensor during prefill.
            logits = self.lm_head(x[:, -1:, :])
            loss = None

        return ModelOutput(logits=logits, loss=loss, attentions=attentions)

    # ------------------------------------------------------------ generation

    def make_cache(
        self,
        batch_size: int,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> KVCache:
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return KVCache(
            n_layer=self.cfg.n_layer,
            batch_size=batch_size,
            max_seq_len=max_seq_len or self.cfg.block_size,
            n_kv_head=self.cfg.n_kv_head or self.cfg.n_head,
            head_dim=self.cfg.head_dim,
            device=device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def generate(
        self,
        idx: torch.Tensor,
        cfg: GenerationConfig | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Autoregressively sample a continuation of ``idx`` ``(B, T)``.

        With ``use_cache=True`` the prompt is processed in one prefill pass and each
        subsequent step forwards a single token. ``use_cache=False`` re-runs the
        whole prefix every step — kept because it is the naive baseline the
        inference benchmarks measure against, not because it is ever preferable.
        """
        cfg = cfg or GenerationConfig()
        self.eval()

        generator = None
        if cfg.seed is not None:
            generator = torch.Generator(device=idx.device).manual_seed(cfg.seed)

        cache = None
        if use_cache:
            total = idx.shape[1] + cfg.max_new_tokens
            if total > self.cfg.block_size:
                raise ValueError(
                    f"prompt ({idx.shape[1]}) + max_new_tokens ({cfg.max_new_tokens}) "
                    f"= {total} exceeds block_size={self.cfg.block_size}"
                )
            cache = self.make_cache(idx.shape[0], max_seq_len=total, device=idx.device)

        out = idx
        step_input = idx
        for _ in range(cfg.max_new_tokens):
            if cache is None:
                step_input = out[:, -self.cfg.block_size :]
            logits = self(step_input, cache=cache).logits[:, -1, :]
            next_token = self._sample(logits, cfg, generator)
            out = torch.cat([out, next_token], dim=1)
            step_input = next_token
            if cfg.stop_tokens and out.shape[0] == 1 and next_token.item() in cfg.stop_tokens:
                break
        return out

    @staticmethod
    def _sample(
        logits: torch.Tensor, cfg: GenerationConfig, generator: torch.Generator | None
    ) -> torch.Tensor:
        if cfg.temperature <= 0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits.float() / cfg.temperature

        if cfg.top_k is not None:
            k = min(cfg.top_k, logits.size(-1))
            kth = logits.topk(k, dim=-1).values[:, -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if cfg.top_p is not None:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            probs = sorted_logits.softmax(-1)
            # Exclusive cumulative sum: a token is dropped only if the mass *before*
            # it already reached top_p, so the token that crosses the threshold is kept
            # and the nucleus is never empty.
            drop_sorted = (probs.cumsum(-1) - probs) > cfg.top_p
            drop = torch.zeros_like(drop_sorted).scatter(1, sorted_idx, drop_sorted)
            logits = logits.masked_fill(drop, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    # ------------------------------------------------------------- accounting

    def num_params(self, non_embedding: bool = False) -> int:
        """Total parameters. ``non_embedding`` matches the convention used when
        people quote "124M": it excludes the position table (and, when untied, the
        output head is still counted, since it is a real matmul)."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding and self.pos_emb is not None:
            n -= self.pos_emb.weight.numel()
        return n

    def flops_per_token(self) -> float:
        """Forward+backward FLOPs per token, PaLM appendix-B style.

        ``6N`` for the parameter matmuls plus ``12 * n_layer * n_embd * block_size``
        for attention's sequence-length-dependent term. Used for MFU, which is the
        number that actually tells you whether a training run is compute-bound.
        """
        cfg = self.cfg
        n = self.num_params(non_embedding=True)
        if cfg.tie_embeddings:
            # The tied head is one matmul that 6N under-counts, since the weight
            # is shared and therefore counted once.
            n += cfg.vocab_size * cfg.n_embd
        return 6 * n + 12 * cfg.n_layer * cfg.n_embd * cfg.block_size

    def estimate_mfu(self, tokens_per_second: float, peak_flops: float) -> float:
        """Model FLOPs Utilisation: achieved FLOP/s as a fraction of hardware peak."""
        return (tokens_per_second * self.flops_per_token()) / peak_flops

    def param_groups(self, weight_decay: float) -> list[dict]:
        """Split parameters into decayed and non-decayed groups.

        Weight decay applies to matmul weights only. Applying it to norm gains and
        biases penalises the very parameters whose job is to set a scale, and
        measurably hurts — this is one of the axes in the ablation study.
        """
        decay, no_decay = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            (decay if p.dim() >= 2 else no_decay).append((name, p))
        return [
            {"params": [p for _, p in decay], "weight_decay": weight_decay, "name": "decay"},
            {"params": [p for _, p in no_decay], "weight_decay": 0.0, "name": "no_decay"},
        ]
