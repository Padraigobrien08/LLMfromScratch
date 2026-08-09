from __future__ import annotations

import pytest
import torch

from llmfs.model import ModelConfig, Transformer


@pytest.fixture(autouse=True)
def _deterministic():
    torch.manual_seed(0)


def tiny_config(**overrides) -> ModelConfig:
    """A model small enough to test exhaustively on CPU in CI."""
    base = dict(
        vocab_size=97,
        n_layer=3,
        n_head=4,
        n_embd=64,
        block_size=32,
        dropout=0.0,
    )
    base.update(overrides)
    return ModelConfig(**base)


def tiny_model(**overrides) -> Transformer:
    model = Transformer(tiny_config(**overrides))
    model.eval()  # dropout off, so every comparison below is exact up to fp error
    return model


# The full cross-product of architectural switches, used to assert that the
# invariants (causality, cache equivalence) hold for *every* configuration the
# ablation study can produce — not just the default one.
ARCH_VARIANTS = [
    pytest.param({}, id="gpt2-baseline"),
    pytest.param({"norm": "rmsnorm"}, id="rmsnorm"),
    pytest.param({"pos_emb": "rope"}, id="rope"),
    pytest.param({"pos_emb": "none"}, id="no-pos"),
    pytest.param({"mlp": "swiglu"}, id="swiglu"),
    pytest.param({"bias": False}, id="no-bias"),
    pytest.param({"tie_embeddings": False}, id="untied"),
    pytest.param({"n_kv_head": 2}, id="gqa-2"),
    pytest.param({"n_kv_head": 1}, id="mqa"),
    pytest.param(
        {
            "norm": "rmsnorm",
            "pos_emb": "rope",
            "mlp": "swiglu",
            "bias": False,
            "n_kv_head": 2,
        },
        id="llama-style",
    ),
]
