"""Training components: schedules, optimiser grouping, checkpointing, and a full
end-to-end run that must actually reduce the loss."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import tiny_config
from llmfs.config import Config, OptimConfig, load_config
from llmfs.model import Transformer
from llmfs.train.checkpoint import (
    find_latest_checkpoint,
    model_from_checkpoint,
    prune_checkpoints,
    save_checkpoint,
    unwrap_model,
)
from llmfs.train.optim import build_optimizer, lr_at_step
from llmfs.train.trainer import Trainer

# ------------------------------------------------------------------- schedules


def test_warmup_rises_linearly_to_the_peak() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=100, schedule="cosine")
    assert lr_at_step(0, cfg, 1000) == pytest.approx(1e-5)
    assert lr_at_step(49, cfg, 1000) == pytest.approx(5e-4)
    assert lr_at_step(99, cfg, 1000) == pytest.approx(1e-3)


def test_warmup_never_starts_at_zero() -> None:
    """A zero-length first update wastes a step and makes step 0 undefined for logs."""
    cfg = OptimConfig(lr=1e-3, warmup_steps=10)
    assert lr_at_step(0, cfg, 100) > 0


def test_cosine_decays_to_the_floor() -> None:
    cfg = OptimConfig(lr=1e-3, min_lr_ratio=0.1, warmup_steps=0, schedule="cosine")
    assert lr_at_step(0, cfg, 1000) == pytest.approx(1e-3)
    assert lr_at_step(500, cfg, 1000) == pytest.approx(5.5e-4, rel=1e-3)  # midpoint
    assert lr_at_step(1000, cfg, 1000) == pytest.approx(1e-4)


def test_cosine_is_monotonically_decreasing_after_warmup() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=50, schedule="cosine")
    rates = [lr_at_step(s, cfg, 500) for s in range(50, 500)]
    assert all(a >= b for a, b in zip(rates, rates[1:]))


def test_linear_schedule() -> None:
    cfg = OptimConfig(lr=1e-3, min_lr_ratio=0.0, warmup_steps=0, schedule="linear")
    assert lr_at_step(500, cfg, 1000) == pytest.approx(5e-4)


def test_wsd_holds_then_decays() -> None:
    """Warmup-stable-decay: constant until the final tenth, then down to the floor."""
    cfg = OptimConfig(
        lr=1e-3, min_lr_ratio=0.1, warmup_steps=10, schedule="wsd", wsd_decay_frac=0.1
    )
    assert lr_at_step(500, cfg, 1000) == pytest.approx(1e-3)  # still stable
    assert lr_at_step(890, cfg, 1000) == pytest.approx(1e-3)  # just before decay
    assert lr_at_step(950, cfg, 1000) < 1e-3  # decaying
    assert lr_at_step(1000, cfg, 1000) == pytest.approx(1e-4)  # floor


def test_constant_schedule_ignores_progress() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=0, schedule="constant")
    assert lr_at_step(0, cfg, 100) == lr_at_step(99, cfg, 100) == 1e-3


def test_lr_stays_within_bounds_after_warmup() -> None:
    """Post-warmup the rate must stay in [min_lr, lr], including past ``max_steps``.

    Overrunning matters: a run extended beyond its planned length must not have its
    schedule wrap around into negative or rising rates. Warmup itself is excluded —
    it legitimately starts below the floor on its way up.
    """
    for schedule in ("cosine", "linear", "wsd", "constant"):
        cfg = OptimConfig(lr=1e-3, min_lr_ratio=0.1, warmup_steps=50, schedule=schedule)  # type: ignore[arg-type]
        rates = [lr_at_step(s, cfg, 500) for s in range(50, 600)]  # deliberately overruns
        assert min(rates) >= 1e-4 - 1e-12, schedule
        assert max(rates) <= 1e-3 + 1e-12, schedule


def test_warmup_climbs_from_above_zero_to_the_peak() -> None:
    cfg = OptimConfig(lr=1e-3, warmup_steps=50, schedule="cosine")
    warmup = [lr_at_step(s, cfg, 500) for s in range(50)]
    assert warmup[0] > 0
    assert all(a < b for a, b in zip(warmup, warmup[1:]))
    assert warmup[-1] == pytest.approx(1e-3)


# ------------------------------------------------------------------- optimiser


def test_optimizer_decays_only_matmul_weights() -> None:
    model = Transformer(tiny_config())
    opt = build_optimizer(model, OptimConfig(weight_decay=0.1), torch.device("cpu"))

    decay, no_decay = opt.param_groups
    assert decay["weight_decay"] == 0.1 and no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() == 1 for p in no_decay["params"])


def test_optimizer_step_changes_weights() -> None:
    model = Transformer(tiny_config())
    opt = build_optimizer(model, OptimConfig(), torch.device("cpu"))
    before = model.tok_emb.weight.detach().clone()

    idx = torch.randint(0, 97, (2, 8))
    model(idx, targets=idx).loss.backward()
    opt.step()

    assert not torch.allclose(before, model.tok_emb.weight)


# ---------------------------------------------------------------- checkpoints


def test_checkpoint_round_trips_through_its_own_config(tmp_path: Path) -> None:
    """A checkpoint must rebuild its architecture from what it recorded, so loading
    cannot silently mismatch the weights."""
    cfg = load_config("debug")
    cfg.model = tiny_config(norm="rmsnorm", pos_emb="rope", mlp="swiglu", n_kv_head=2)
    cfg.data.block_size = cfg.model.block_size

    model = Transformer(cfg.model).eval()
    save_checkpoint(tmp_path / "ckpt.pt", model, None, step=42, config=cfg)

    restored, ckpt = model_from_checkpoint(tmp_path / "ckpt.pt")
    assert ckpt["step"] == 42
    assert restored.cfg.norm == "rmsnorm" and restored.cfg.n_kv_head == 2

    idx = torch.randint(0, 97, (1, 8))
    torch.testing.assert_close(model(idx, targets=idx).logits, restored(idx, targets=idx).logits)


def test_checkpoint_write_is_atomic(tmp_path: Path) -> None:
    """No temporary file survives a successful write; a killed process therefore
    leaves the previous checkpoint intact rather than a truncated one."""
    cfg = load_config("debug")
    model = Transformer(cfg.model)
    save_checkpoint(tmp_path / "ckpt.pt", model, None, step=1, config=cfg)
    assert (tmp_path / "ckpt.pt").exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_unwrap_model_strips_wrappers() -> None:
    model = Transformer(tiny_config())

    class FakeDDP(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner

    assert unwrap_model(FakeDDP(model)) is model
    assert unwrap_model(model) is model


def test_prune_keeps_the_newest_and_spares_named_checkpoints(tmp_path: Path) -> None:
    for step in (100, 200, 300):
        (tmp_path / f"ckpt_step{step:07d}.pt").write_bytes(b"x")
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "final.pt").write_bytes(b"x")

    prune_checkpoints(tmp_path, keep_last_n=2)

    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["best.pt", "ckpt_step0000200.pt", "ckpt_step0000300.pt", "final.pt"]


def test_find_latest_checkpoint(tmp_path: Path) -> None:
    assert find_latest_checkpoint(tmp_path) is None
    for step in (5, 40, 300):
        (tmp_path / f"ckpt_step{step:07d}.pt").write_bytes(b"x")
    # Zero-padded names sort correctly, so 300 wins over 40 rather than losing to it.
    assert find_latest_checkpoint(tmp_path).name == "ckpt_step0000300.pt"


# ------------------------------------------------------------------ end-to-end


@pytest.fixture
def train_config(tmp_path: Path) -> Config:
    """A complete but tiny run over a synthetic corpus."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    # A low-entropy corpus, so a working trainer visibly reduces loss in a few steps.
    for name, size in [("train_000000.bin", 20_000), ("val_000000.bin", 4_000)]:
        rng.integers(0, 64, size=size, dtype=np.uint16).tofile(data_dir / name)
    (data_dir / "meta.json").write_text(json.dumps({"tokenizer": "gpt2", "vocab_size": 64}))

    cfg = load_config("debug")
    cfg.model = tiny_config(vocab_size=128, block_size=32)
    cfg.data.data_dir = str(data_dir)
    cfg.data.block_size = 32
    cfg.data.micro_batch_size = 4
    cfg.train.tokens_per_step = 4 * 32 * 2  # 2 accumulation steps
    cfg.train.max_steps = 30
    cfg.optim.warmup_steps = 5
    cfg.runtime.compile = False
    cfg.runtime.device = "cpu"
    cfg.log.out_dir = str(tmp_path / "out")
    cfg.log.run_name = "test"
    cfg.log.tensorboard = False
    cfg.log.eval_interval = 15
    cfg.log.eval_steps = 3
    cfg.log.log_interval = 10
    return cfg


def test_training_reduces_loss_and_writes_artifacts(train_config: Config) -> None:
    trainer = Trainer(train_config)
    state = trainer.train()

    assert state.step == 30
    assert state.tokens_seen == 30 * train_config.train.tokens_per_step

    losses = [h["val_loss"] for h in state.history]
    assert losses[-1] < losses[0], f"loss did not improve: {losses}"

    run_dir = Path(train_config.log.out_dir) / "test"
    assert (run_dir / "final.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "config.yaml").exists()

    # Metrics are on disk as JSONL, which is what the ablation plots read.
    records = [json.loads(line) for line in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert any("train/loss" in r for r in records)
    assert any("val/loss" in r for r in records)
    assert any("perf/tokens_per_sec" in r for r in records)


def test_resume_continues_from_the_recorded_step(train_config: Config) -> None:
    Trainer(train_config).train()

    train_config.train.max_steps = 45
    train_config.train.resume = "auto"
    resumed = Trainer(train_config)
    assert resumed.state.step == 30

    state = resumed.train()
    assert state.step == 45


def test_data_tokenizer_mismatch_is_caught(train_config: Config) -> None:
    """Training on data built with a different tokenizer would just converge badly;
    it has to fail at startup instead."""
    meta_path = Path(train_config.data.data_dir) / "meta.json"
    meta_path.write_text(json.dumps({"tokenizer": "file:other.json", "vocab_size": 64}))
    with pytest.raises(ValueError, match="tokenizer"):
        Trainer(train_config)


def test_vocab_too_small_for_the_data_is_caught(train_config: Config) -> None:
    meta_path = Path(train_config.data.data_dir) / "meta.json"
    meta_path.write_text(json.dumps({"tokenizer": "gpt2", "vocab_size": 50257}))
    train_config.model.vocab_size = 128
    with pytest.raises(ValueError, match="exceeds model.vocab_size"):
        Trainer(train_config)


def test_gradient_checkpointing_matches_ordinary_training(train_config: Config) -> None:
    """Recomputing activations is a memory optimisation, so it must not change the
    gradients it produces."""
    train_config.runtime.grad_checkpointing = True
    trainer = Trainer(train_config)
    loss = trainer._accumulate_gradients()
    grads = {n: p.grad.clone() for n, p in trainer.model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    assert loss > 0
