"""Training components: schedules, optimiser grouping, checkpointing, and a full
end-to-end run that must actually reduce the loss."""

from __future__ import annotations

import dataclasses
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


def test_checkpoint_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The bytes go to a temporary path and are renamed into place, in that order.

    The previous version of this test asserted that the file existed afterwards and that
    no `*.tmp` was left behind — both of which are equally true of a plain `torch.save`
    straight to the destination, which is exactly the write this is supposed to forbid.
    Replacing the tmp-then-rename with a direct save left it green. So the sequence is
    observed rather than inferred from its leftovers.
    """
    import os

    cfg = load_config("debug")
    model = Transformer(cfg.model)
    path = tmp_path / "ckpt.pt"

    events: list[tuple] = []
    real_save, real_replace = torch.save, os.replace

    def save_spy(obj, f, *args, **kwargs):
        events.append(("save", Path(f)))
        return real_save(obj, f, *args, **kwargs)

    def replace_spy(src, dst, *args, **kwargs):
        events.append(("replace", Path(src), Path(dst)))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(torch, "save", save_spy)
    monkeypatch.setattr(os, "replace", replace_spy)
    save_checkpoint(path, model, None, step=1, config=cfg)
    monkeypatch.undo()

    assert [e[0] for e in events] == ["save", "replace"], f"wrong write sequence: {events}"
    written, (_, src, dst) = events[0][1], events[1]
    assert written != path, "the payload was written straight to the destination"
    assert written == src and dst == path
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_a_crash_mid_write_leaves_the_previous_checkpoint_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The property the atomic write exists for, stated as the failure it prevents.

    Twelve hours into a rented run, a process killed while serialising 500 MB of weights
    must leave the last good checkpoint readable. Writing in place would leave a truncated
    file with a valid name, and the run would be unrecoverable from its own artifacts.
    """
    cfg = load_config("debug")
    model = Transformer(cfg.model)
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, step=1, config=cfg)
    good = path.read_bytes()

    real_save = torch.save

    def dies_after_writing(obj, f, *args, **kwargs):
        real_save(obj, f, *args, **kwargs)
        raise RuntimeError("pod terminated mid-checkpoint")

    monkeypatch.setattr(torch, "save", dies_after_writing)
    with pytest.raises(RuntimeError, match="terminated"):
        save_checkpoint(path, model, None, step=2, config=cfg)
    monkeypatch.undo()

    assert path.read_bytes() == good, "the destination was overwritten before the write finished"
    assert model_from_checkpoint(path)[1]["step"] == 1


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


def test_resuming_reproduces_an_uninterrupted_run_with_dropout_on(train_config: Config) -> None:
    """The "as though it had never stopped" claim, with something consuming the RNG.

    The checkpoint stored no RNG state, so a resumed run reseeded from the config and
    replayed step 0's dropout masks. Nothing caught it because every shipped config sets
    `dropout: 0.0`, the learning rate is a pure function of the step, and the loader
    derives its position from the step too — so with nothing drawing from the stream, the
    trajectories matched anyway. Dropout is what makes the difference observable, and this
    is the only test that turns it on.

    The comparison is against a genuinely uninterrupted run, not against a second resume:
    two resumes agreeing only proves they are consistently wrong.
    """
    train_config.model.dropout = 0.2
    train_config.log.eval_interval = 10
    train_config.log.keep_last_n = 5
    train_config.train.max_steps = 20

    straight_through = Trainer(train_config).train()
    midpoint = Path(train_config.log.out_dir) / train_config.log.run_name / "ckpt_step0000010.pt"
    assert midpoint.exists()

    # Resume from that run's own step-10 checkpoint, with `max_steps` unchanged — the
    # cosine schedule is a function of it, so shortening the first leg would change the
    # learning rates and make the two runs incomparable for reasons that have nothing to
    # do with the RNG.
    train_config.log.run_name = "resumed"
    train_config.train.resume = str(midpoint)
    resumed = Trainer(train_config).train()

    # The resumed trainer's history begins at the resume, so the shared point is the last.
    end, reference = resumed.history[-1], straight_through.history[-1]
    assert end["step"] == reference["step"] == 20
    assert end["val_loss"] == reference["val_loss"], (
        f"resumed {end['val_loss']!r} vs uninterrupted {reference['val_loss']!r} — "
        "the second half of the run drew different dropout masks"
    )


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
    gradients it produces.

    The first version of this test asserted only that the checkpointed gradients existed
    and were finite — it never built the baseline it was named after, so it could not have
    failed for any reason short of a crash. Putting every block under `torch.no_grad()`
    left it green. The comparison is the whole test, so it is made here: same seed, same
    batches, gradients differenced parameter by parameter.
    """

    def grads_with(checkpointing: bool) -> dict[str, torch.Tensor]:
        train_config.runtime.grad_checkpointing = checkpointing
        torch.manual_seed(0)  # same init and, with a step-derived loader, the same batches
        trainer = Trainer(train_config)
        loss = trainer._accumulate_gradients()
        assert loss > 0
        params = dict(trainer.model.named_parameters())
        assert any(
            "checkpoint" not in n for n in params
        )  # sanity: the wrapper does not rename anything
        return {n: p.grad.clone() for n, p in params.items() if p.grad is not None}

    plain = grads_with(False)
    checkpointed = grads_with(True)

    assert plain and set(plain) == set(checkpointed)
    for name, g in plain.items():
        assert torch.isfinite(g).all(), name
        # Recomputation reruns the same ops on the same inputs, so on CPU in fp32 this is
        # exact. A tolerance here would admit a subtly different backward pass.
        torch.testing.assert_close(
            checkpointed[name], g, rtol=0, atol=0, msg=lambda m, n=name: f"{n}: {m}"
        )


def test_divergence_stops_immediately_and_spares_the_last_checkpoint(train_config: Config) -> None:
    """A NaN must halt the run rather than propagate into every later checkpoint.

    Without the guard the weights are poisoned from the first bad step onward and
    `final.pt` is overwritten with them, so the newest recoverable state can be
    thousands of steps behind by the time anyone notices.
    """
    from llmfs.train.trainer import TrainingDiverged

    train_config.optim.lr = 50.0  # guaranteed to blow up
    train_config.optim.grad_clip = 1e9  # clipping would otherwise mask it
    trainer = Trainer(train_config)

    with pytest.raises(TrainingDiverged, match="non-finite"):
        trainer.train()

    assert trainer.diverged is True
    assert trainer.state.step < train_config.train.max_steps, "should stop early"
    # No final checkpoint: writing one would destroy the last good state.
    assert not (Path(train_config.log.out_dir) / "test" / "final.pt").exists()


def test_finite_training_does_not_trip_the_guard(train_config: Config) -> None:
    trainer = Trainer(train_config)
    trainer.train()
    assert trainer.diverged is False
    assert (Path(train_config.log.out_dir) / "test" / "final.pt").exists()


def test_prune_keep_zero_removes_all_rolling_checkpoints(tmp_path: Path) -> None:
    """keep_last_n=0 means keep none — not, as a `<= 0` guard would have it, keep all.

    This matters at sweep scale: 39 run directories at the default of 2 rolling
    checkpoints each is ~109 GiB, more than the prepared corpus and more than the
    volume it lives on.
    """
    for step in (100, 200, 300):
        (tmp_path / f"ckpt_step{step:07d}.pt").write_bytes(b"x")
    (tmp_path / "best.pt").write_bytes(b"x")
    (tmp_path / "final.pt").write_bytes(b"x")

    prune_checkpoints(tmp_path, keep_last_n=0)

    # Rolling checkpoints gone; the two named ones survive, so the run is still
    # recoverable and its best model intact.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["best.pt", "final.pt"]


def test_prune_negative_disables_pruning(tmp_path: Path) -> None:
    for step in (100, 200):
        (tmp_path / f"ckpt_step{step:07d}.pt").write_bytes(b"x")
    prune_checkpoints(tmp_path, keep_last_n=-1)
    assert len(list(tmp_path.iterdir())) == 2


def test_milestone_checkpoints_are_written_and_never_pruned(train_config: Config) -> None:
    """Intermediate training states are the one artifact that cannot be recovered.

    Reconstructing step N of a finished run means paying for the run again, so
    milestones are written at fixed fractions and deliberately named outside the
    ``ckpt_step*`` glob that pruning uses.
    """
    train_config.log.milestone_fracs = [0.25, 0.5]
    train_config.log.keep_last_n = 0  # aggressive pruning must not touch them
    train_config.log.checkpoint_interval = 5

    Trainer(train_config).train()

    run_dir = Path(train_config.log.out_dir) / "test"
    milestones = sorted(p.name for p in run_dir.glob("milestone_*.pt"))
    assert len(milestones) == 2, milestones
    # 25% and 50% of 30 steps.
    assert "milestone_025pct_step0000007.pt" in milestones
    assert "milestone_050pct_step0000015.pt" in milestones
    # Pruning cleared the rolling checkpoints but left the milestones alone.
    assert not list(run_dir.glob("ckpt_step*.pt"))


def test_milestones_can_be_disabled(train_config: Config) -> None:
    train_config.log.milestone_fracs = []
    Trainer(train_config).train()
    run_dir = Path(train_config.log.out_dir) / "test"
    assert not list(run_dir.glob("milestone_*.pt"))


def test_the_same_seed_gives_the_same_run_and_a_different_seed_does_not(
    train_config: Config,
) -> None:
    """Determinism, asserted rather than assumed.

    Every ablation delta in the study is a difference between runs, and the whole design
    — paired seeds, a noise floor measured from three baselines — presumes that a seed
    fixes a trajectory. That had been verified by hand and never by the suite, which
    means nothing would have caught a change that quietly introduced run-to-run variation
    and inflated every delta into it.

    The second half matters as much as the first: a run that ignored its seed entirely
    would be perfectly reproducible and completely useless as a control.
    """
    train_config.train.max_steps = 8
    train_config.log.eval_interval = 4

    def losses(seed: int, run: str) -> list[float]:
        train_config.runtime.seed = seed
        train_config.log.run_name = run
        return [h["val_loss"] for h in Trainer(train_config).train().history]

    first = losses(1337, "a")
    again = losses(1337, "b")
    other = losses(1338, "c")

    assert first == again, f"same seed diverged: {first} vs {again}"
    assert first != other, "the seed had no effect — the runs are not seeded at all"


def test_compiled_training_saves_clean_checkpoints(train_config: Config, monkeypatch) -> None:
    """The compile path had zero coverage: every test forces `compile=False`, so the
    wrapper plumbing — building on the OptimizedModule, unwrapping `_orig_mod` on save —
    could break without a test noticing. The backend is swapped to `eager` so this
    exercises the repository's plumbing rather than paying for an inductor build; what
    it pins is dynamo wrapping plus the unwrap on every save path."""
    real_compile = torch.compile
    monkeypatch.setattr(
        torch, "compile", lambda model, **kwargs: real_compile(model, backend="eager")
    )

    train_config.runtime.compile = True
    train_config.train.max_steps = 2
    trainer = Trainer(train_config)
    state = trainer.train()
    assert state.step == 2

    ckpt_path = Path(train_config.log.out_dir) / "test" / "final.pt"
    model, ckpt = model_from_checkpoint(ckpt_path)
    assert not any(key.startswith("_orig_mod") for key in ckpt["model"]), (
        "the checkpoint carries compile-wrapper prefixes and would not reload into a bare model"
    )


def test_resume_refuses_a_checkpoint_from_a_different_run(train_config: Config) -> None:
    """The data position is derived from the step, so a resume under a changed
    tokens_per_step or corpus silently seeks somewhere else in the stream — the one
    drift the no-stored-state design left possible, moved up a level. Refused now."""
    train_config.log.checkpoint_interval = 1  # a rolling ckpt for auto-resume to find
    train_config.train.max_steps = 2
    Trainer(train_config).train()

    stale = dataclasses.replace(
        train_config,
        train=dataclasses.replace(
            train_config.train,
            resume="auto",
            tokens_per_step=train_config.train.tokens_per_step * 2,
        ),
    )
    with pytest.raises(ValueError, match="tokens_per_step") as excinfo:
        Trainer(stale)
    assert "resume_force" in str(excinfo.value), "the message must name the override"


def test_resume_force_overrides_the_config_check(train_config: Config) -> None:
    train_config.log.checkpoint_interval = 1  # a rolling ckpt for auto-resume to find
    train_config.train.max_steps = 2
    Trainer(train_config).train()

    forced = dataclasses.replace(
        train_config,
        train=dataclasses.replace(
            train_config.train,
            resume="auto",
            resume_force=True,
            tokens_per_step=train_config.train.tokens_per_step * 2,
            max_steps=3,
        ),
    )
    trainer = Trainer(forced)
    assert trainer.state.step == 2


def test_resume_allows_the_documented_resharding(train_config: Config) -> None:
    """micro_batch_size (and world size) are deliberately outside the check: the
    position depends on their product only through tokens_per_step, and re-sharding a
    run across different hardware is the design's stated point."""
    train_config.log.checkpoint_interval = 1  # a rolling ckpt for auto-resume to find
    train_config.train.max_steps = 2
    Trainer(train_config).train()

    resharded = dataclasses.replace(
        train_config,
        data=dataclasses.replace(train_config.data, micro_batch_size=2),
        train=dataclasses.replace(train_config.train, resume="auto", max_steps=3),
    )
    trainer = Trainer(resharded)
    assert trainer.state.step == 2
