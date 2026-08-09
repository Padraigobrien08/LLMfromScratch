"""Extract attention maps and per-head statistics from a trained model.

A token-by-token heatmap shows *what* a head did on one input. The statistics
computed here try to characterise *what a head is for*, which is the part that
transfers between inputs:

``entropy``
    How concentrated the head's attention is, normalised against the uniform
    distribution over the positions it can legally see. Near 0 means the head
    attends to one token; near 1 means it spreads evenly and is probably not doing
    much on this input.
``mean_distance``
    Attention-weighted mean of ``|query - key|``. Small means local, large means the
    head is moving information across the sequence.
``prev_token_fraction``
    Mass placed exactly one position back. Heads that score high here are the
    previous-token heads that induction circuits are built from.
``sink_fraction``
    Mass placed on position 0. Large values are usually the attention-sink effect —
    softmax must put its mass somewhere, so heads with nothing to do park it on the
    first token rather than distributing it.

None of these are ground truth about a head's role; they are cheap summary
statistics that make an interesting head easy to find in a grid of 144.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..data.tokenizer import Tokenizer, load_tokenizer
from ..model import Transformer
from ..train.checkpoint import model_from_checkpoint
from ..utils.device import get_device


@dataclass
class HeadStats:
    layer: int
    head: int
    entropy: float
    mean_distance: float
    prev_token_fraction: float
    sink_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "head": self.head,
            "entropy": round(self.entropy, 4),
            "mean_distance": round(self.mean_distance, 3),
            "prev_token_fraction": round(self.prev_token_fraction, 4),
            "sink_fraction": round(self.sink_fraction, 4),
        }


@dataclass
class AttentionView:
    """Everything the front end needs for one prompt."""

    prompt: str
    tokens: list[str]
    token_ids: list[int]
    n_layer: int
    n_head: int
    weights: np.ndarray  # (n_layer, n_head, T, T), float32
    stats: list[HeadStats] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        return len(self.tokens)

    def to_payload(self) -> dict[str, Any]:
        """Serialise for the browser.

        Weights are quantised to uint8 and base64-encoded rather than written as JSON
        numbers. For a 12-layer, 12-head model over 64 tokens that is 590k values: as
        JSON text it runs to several megabytes, which is a slow page load for data
        that is about to be drawn into cells a few pixels wide. One part in 255 is
        finer than the display can resolve, so nothing visible is lost — but the
        quantisation floor does mean very small weights render as exactly zero, which
        is why the statistics above are computed from the full-precision array before
        this conversion.
        """
        quantised = np.clip(np.rint(self.weights * 255.0), 0, 255).astype(np.uint8)
        return {
            "prompt": self.prompt,
            "tokens": self.tokens,
            "token_ids": self.token_ids,
            "n_layer": self.n_layer,
            "n_head": self.n_head,
            "n_tokens": self.n_tokens,
            "weights_b64": base64.b64encode(quantised.tobytes()).decode("ascii"),
            "weights_shape": list(quantised.shape),
            "stats": [s.to_dict() for s in self.stats],
            "meta": self.meta,
        }


def display_token(tokenizer: Tokenizer, token_id: int) -> str:
    """Render one token so whitespace and newlines stay visible.

    Byte-level BPE encodes a leading space into the token itself, so "the" and " the"
    are different tokens. Collapsing that in the display would make the visualisation
    lie about what the model actually attended to.
    """
    text = tokenizer.decode([token_id])
    if text == "":
        return "∅"
    return text.replace(" ", "␣").replace("\n", "⏎").replace("\t", "⇥")


def compute_head_stats(weights: np.ndarray) -> list[HeadStats]:
    """Summary statistics per head, from the full-precision ``(L, H, T, T)`` array."""
    n_layer, n_head, T, _ = weights.shape
    positions = np.arange(T)
    # Query i may attend to i + 1 positions, so the uniform-attention entropy that
    # position could reach is log(i + 1). Normalising by it makes rows comparable:
    # otherwise early tokens always look "focused" purely because they have fewer
    # options, and every head appears focused near the start of the sequence.
    max_entropy = np.log(positions + 1.0)
    max_entropy[0] = 1.0  # position 0 can only attend to itself; define its ratio as 0

    stats: list[HeadStats] = []
    for layer in range(n_layer):
        for head in range(n_head):
            w = weights[layer, head].astype(np.float64)

            entropy = -(w * np.log(w + 1e-12)).sum(axis=-1)
            normalised = entropy / max_entropy
            normalised[0] = 0.0

            distance = (w * np.abs(positions[None, :] - positions[:, None])).sum(axis=-1)

            prev = np.zeros(T)
            if T > 1:
                prev[1:] = w[np.arange(1, T), np.arange(0, T - 1)]

            stats.append(
                HeadStats(
                    layer=layer,
                    head=head,
                    entropy=float(normalised.mean()),
                    mean_distance=float(distance.mean()),
                    # Position 0 is excluded from both fractions below: it trivially
                    # attends to itself with weight 1, which would inflate every head's
                    # sink score by 1/T for no reason.
                    prev_token_fraction=float(prev[1:].mean()) if T > 1 else 0.0,
                    sink_fraction=float(w[1:, 0].mean()) if T > 1 else 0.0,
                )
            )
    return stats


@torch.no_grad()
def attention_for_prompt(
    model: Transformer,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 64,
    device: torch.device | str = "cpu",
) -> AttentionView:
    """Run one forward pass and collect every layer's attention weights."""
    if not prompt.strip():
        raise ValueError("prompt is empty")

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise ValueError("prompt tokenised to nothing")

    # The payload and the render cost both grow with T^2, and a 64-token window is
    # already 4,096 cells per head. Truncation is from the left so the most recent
    # context — the part a reader is usually asking about — is what survives.
    truncated = len(token_ids) > max_tokens
    if truncated:
        token_ids = token_ids[-max_tokens:]

    if len(token_ids) > model.cfg.block_size:
        token_ids = token_ids[-model.cfg.block_size :]
        truncated = True

    device = torch.device(device)
    idx = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    out = model(idx, need_weights=True)
    assert out.attentions is not None

    # (n_layer, n_head, T, T), batch dimension dropped.
    weights = torch.stack([a[0] for a in out.attentions]).float().cpu().numpy()

    return AttentionView(
        prompt=prompt,
        tokens=[display_token(tokenizer, t) for t in token_ids],
        token_ids=token_ids,
        n_layer=model.cfg.n_layer,
        n_head=model.cfg.n_head,
        weights=weights,
        stats=compute_head_stats(weights),
        meta={
            "truncated": truncated,
            "n_kv_head": model.cfg.n_kv_head,
            "norm": model.cfg.norm,
            "pos_emb": model.cfg.pos_emb,
            "mlp": model.cfg.mlp,
            "params": model.num_params(),
        },
    )


def load_model_and_tokenizer(
    checkpoint: str | Path, device: str = "auto"
) -> tuple[Transformer, Tokenizer, dict[str, Any]]:
    """Load a checkpoint plus the tokenizer it was trained with."""
    dev = get_device(device)
    model, ckpt = model_from_checkpoint(checkpoint, device=dev)
    tokenizer = load_tokenizer(ckpt["config"]["data"]["tokenizer"])
    info = {
        "checkpoint": str(checkpoint),
        "step": ckpt["step"],
        "val_loss": ckpt.get("metrics", {}).get("val_loss"),
        "run_name": ckpt["config"]["log"]["run_name"],
        "device": str(dev),
    }
    return model, tokenizer, info
