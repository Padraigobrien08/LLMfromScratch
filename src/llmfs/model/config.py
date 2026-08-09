"""Model configuration.

Every architectural choice that the ablation study varies is a field here, so an
ablation is a config diff rather than a code fork. The defaults describe a
GPT-2 124M baseline; ``configs/llama-124m.yaml`` flips them to the modern stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

NormType = Literal["layernorm", "rmsnorm"]
PosEmbType = Literal["learned", "rope", "none"]
MLPType = Literal["gelu", "swiglu"]
AttnImpl = Literal["sdpa", "eager"]


@dataclass
class ModelConfig:
    # --- Shape ---
    vocab_size: int = 50304  # GPT-2's 50257 padded to a multiple of 64 for tensor-core alignment
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    block_size: int = 1024
    """Maximum sequence length the model is trained on."""

    n_kv_head: int | None = None
    """Number of key/value heads. ``None`` means multi-head attention (n_kv_head == n_head).
    Values < n_head give grouped-query attention; 1 gives multi-query attention."""

    # --- Architectural switches (the ablation axes) ---
    norm: NormType = "layernorm"
    pos_emb: PosEmbType = "learned"
    mlp: MLPType = "gelu"
    tie_embeddings: bool = True
    bias: bool = True
    """Use bias terms in Linear/Norm layers. GPT-2 does; Llama does not."""

    # --- Component hyperparameters ---
    mlp_ratio: float = 4.0
    """Hidden width of the feed-forward block as a multiple of n_embd. For SwiGLU the
    hidden width is additionally scaled by 2/3 so parameter count is comparable."""
    mlp_hidden_multiple_of: int = 256
    """Round the SwiGLU hidden width up to a multiple of this, for kernel efficiency."""
    norm_eps: float = 1e-5
    rope_theta: float = 10_000.0

    # --- Regularisation ---
    dropout: float = 0.0
    """GPT-2 replications typically use 0.0 for a single epoch over a large corpus."""

    # --- Initialisation ---
    init_std: float = 0.02
    scale_residual_init: bool = True
    """Scale the init of residual output projections by 1/sqrt(2 * n_layer) (GPT-2)."""

    # --- Implementation ---
    attn_impl: AttnImpl = "sdpa"
    """``sdpa`` uses torch's fused kernel (fast). ``eager`` materialises the attention
    matrix — slower, but required to export attention weights for the visualizer."""

    def __post_init__(self) -> None:
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_head > self.n_head:
            raise ValueError(
                f"n_kv_head ({self.n_kv_head}) cannot exceed n_head ({self.n_head})"
            )
        if self.n_head % self.n_kv_head != 0:
            raise ValueError(
                f"n_head ({self.n_head}) must be divisible by n_kv_head ({self.n_kv_head})"
            )
        if self.pos_emb == "rope" and self.head_dim % 2 != 0:
            raise ValueError(f"RoPE needs an even head_dim, got {self.head_dim}")

    # --- Derived ---
    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head

    @property
    def n_kv_groups(self) -> int:
        """How many query heads share each key/value head."""
        assert self.n_kv_head is not None
        return self.n_head // self.n_kv_head

    @property
    def mlp_hidden(self) -> int:
        """Feed-forward hidden width.

        A SwiGLU block has three projections instead of two, so the naive 4*d width
        would give it 1.5x the parameters of the GELU block and make any comparison
        between them meaningless. Scaling by 2/3 is the standard fix (PaLM, Llama) and
        keeps the ablation honest.
        """
        hidden = int(self.mlp_ratio * self.n_embd)
        if self.mlp == "swiglu":
            hidden = int(2 * hidden / 3)
            m = self.mlp_hidden_multiple_of
            hidden = m * ((hidden + m - 1) // m)
        return hidden


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.8
    top_k: int | None = 200
    top_p: float | None = None
    seed: int | None = None
    stop_tokens: list[int] = field(default_factory=list)
