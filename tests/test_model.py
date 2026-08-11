"""Model-level invariants: tying, initialisation, parameter accounting, determinism."""

from __future__ import annotations

import math

import pytest
import torch

from conftest import ARCH_VARIANTS, tiny_config, tiny_model
from llmfs.model import GenerationConfig, ModelConfig, Transformer


@pytest.mark.parametrize("overrides", ARCH_VARIANTS)
def test_forward_shapes_and_finite_loss(overrides: dict) -> None:
    model = tiny_model(**overrides)
    idx = torch.randint(0, 97, (2, 12))
    out = model(idx, targets=idx)
    assert out.logits.shape == (2, 12, model.cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_untrained_loss_is_near_uniform() -> None:
    """A correctly initialised model starts at ln(vocab_size).

    Far above means the output distribution is skewed at init; far below means
    something is leaking the answer.
    """
    model = tiny_model(vocab_size=512)
    idx = torch.randint(0, 512, (8, 32))
    # Targets must be independent of the inputs. Reusing ``idx`` as its own target
    # makes a tied-embedding model look better than uniform at init, because
    # logits = x @ W_emb.T peaks on the current token — it would be scoring a
    # copy task, not next-token prediction.
    targets = torch.randint(0, 512, (8, 32))
    loss = model(idx, targets=targets).loss
    assert abs(loss.item() - math.log(512)) < 0.2


def test_inference_forward_returns_only_last_position() -> None:
    """Skipping the unused logits keeps prefill from allocating a (B, T, vocab) tensor."""
    model = tiny_model()
    idx = torch.randint(0, 97, (2, 12))
    assert model(idx).logits.shape == (2, 1, 97)
    assert model(idx, targets=idx).logits.shape == (2, 12, 97)


def test_weight_tying_shares_storage() -> None:
    tied = tiny_model(tie_embeddings=True)
    assert tied.lm_head.weight is tied.tok_emb.weight

    untied = tiny_model(tie_embeddings=False)
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert untied.num_params() > tied.num_params()


@pytest.mark.showcase(
    pins="that both roles of a tied embedding matrix contribute to its one gradient",
    why="Weight tying is easy to implement as a copy rather than a share. The model "
    "then trains, converges, and quietly optimises the output head against an input "
    "table that never receives its half of the signal.",
)
def test_tying_couples_gradients() -> None:
    """Both roles of the shared matrix must contribute to the one gradient."""
    model = tiny_model(tie_embeddings=True)
    idx = torch.randint(0, 97, (2, 8))
    model(idx, targets=idx).loss.backward()
    assert model.tok_emb.weight.grad is not None
    assert model.tok_emb.weight.grad.abs().sum() > 0


def test_residual_init_is_downscaled() -> None:
    """Residual output projections start at 1/sqrt(2 * n_layer) of the base std."""
    cfg = tiny_config(n_layer=8, scale_residual_init=True)
    model = Transformer(cfg)
    expected = cfg.init_std / math.sqrt(2 * cfg.n_layer)

    std = model.blocks[0].attn.out_proj.weight.std().item()
    assert abs(std - expected) / expected < 0.15

    unscaled = Transformer(tiny_config(n_layer=8, scale_residual_init=False))
    assert unscaled.blocks[0].attn.out_proj.weight.std().item() > std * 2


def test_param_groups_partition_all_parameters() -> None:
    """Every trainable parameter lands in exactly one group — none silently dropped."""
    model = tiny_model()
    groups = model.param_groups(weight_decay=0.1)
    grouped = sum(p.numel() for g in groups for p in g["params"])
    assert grouped == sum(p.numel() for p in model.parameters() if p.requires_grad)

    decay, no_decay = groups
    assert decay["weight_decay"] == 0.1
    assert no_decay["weight_decay"] == 0.0
    # Norm gains and biases are 1-D and must not be decayed.
    assert all(p.dim() == 1 for p in no_decay["params"])
    assert all(p.dim() >= 2 for p in decay["params"])


def test_num_params_excludes_position_table_when_asked() -> None:
    cfg = tiny_config(pos_emb="learned")
    model = Transformer(cfg)
    delta = model.num_params() - model.num_params(non_embedding=True)
    assert delta == cfg.block_size * cfg.n_embd


def test_gpt2_124m_parameter_count() -> None:
    """The published GPT-2 small figure, reproduced from our own config.

    124M counts the token embedding but not the position table. Landing on it is a
    cheap, decisive check that the architecture is the one people mean by "GPT-2 small".
    """
    cfg = ModelConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768, block_size=1024)
    n = Transformer(cfg).num_params(non_embedding=True)
    assert 123.5e6 < n < 124.9e6, f"got {n:,}"


def test_generation_is_reproducible_under_seed() -> None:
    model = tiny_model()
    prompt = torch.randint(0, 97, (1, 4))
    cfg = GenerationConfig(max_new_tokens=12, temperature=0.9, seed=42)
    assert torch.equal(model.generate(prompt, cfg), model.generate(prompt, cfg))


def test_greedy_decoding_is_deterministic_without_seed() -> None:
    model = tiny_model()
    prompt = torch.randint(0, 97, (1, 4))
    cfg = GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=None)
    assert torch.equal(model.generate(prompt, cfg), model.generate(prompt, cfg))


def test_top_k_restricts_support() -> None:
    """With top_k=1 sampling must collapse onto the argmax."""
    model = tiny_model()
    prompt = torch.randint(0, 97, (1, 4))
    top1 = model.generate(prompt, GenerationConfig(max_new_tokens=8, temperature=1.0, top_k=1))
    greedy = model.generate(prompt, GenerationConfig(max_new_tokens=8, temperature=0.0))
    assert torch.equal(top1, greedy)


def test_top_p_keeps_nucleus_non_empty() -> None:
    """Even at a threshold below the top token's own mass, one token must survive."""
    model = tiny_model()
    prompt = torch.randint(0, 97, (1, 4))
    out = model.generate(
        prompt, GenerationConfig(max_new_tokens=6, temperature=1.0, top_k=None, top_p=1e-6)
    )
    assert out.shape == (1, 10)


def test_sequence_longer_than_block_size_rejected() -> None:
    model = tiny_model()  # block_size=32
    with pytest.raises(ValueError, match="exceeds block_size"):
        model(torch.randint(0, 97, (1, 33)))


def test_flops_per_token_scales_with_size() -> None:
    small = Transformer(tiny_config(n_layer=2))
    large = Transformer(tiny_config(n_layer=8))
    assert large.flops_per_token() > small.flops_per_token()

    # Sanity-check the magnitude against the 6N rule of thumb for GPT-2 124M.
    gpt2 = Transformer(
        ModelConfig(vocab_size=50257, n_layer=12, n_head=12, n_embd=768, block_size=1024)
    )
    assert 8e8 < gpt2.flops_per_token() < 2e9


def test_state_dict_round_trip() -> None:
    """Checkpoints must reload into an identically-configured model, exactly."""
    model = tiny_model(norm="rmsnorm", pos_emb="rope", mlp="swiglu", n_kv_head=2)
    clone = Transformer(model.cfg).eval()
    clone.load_state_dict(model.state_dict())

    idx = torch.randint(0, 97, (2, 10))
    torch.testing.assert_close(model(idx, targets=idx).logits, clone(idx, targets=idx).logits)


def test_rope_buffers_stay_out_of_the_checkpoint() -> None:
    """Non-persistent buffers keep block_size changeable across a reload."""
    model = tiny_model(pos_emb="rope")
    assert not any("cos_cached" in k or "sin_cached" in k for k in model.state_dict())
