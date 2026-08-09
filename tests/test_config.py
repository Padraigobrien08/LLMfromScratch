"""Config loading, inheritance, overrides, and validation.

Config bugs are the expensive kind: a silently-ignored key means a GPU run
measures something other than what the write-up claims it measured.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from llmfs.config import CONFIG_ROOT, Config, load_config


def write(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_defaults_are_a_valid_config() -> None:
    cfg = Config()
    assert cfg.model.block_size == cfg.data.block_size


def test_block_size_must_agree_across_sections() -> None:
    with pytest.raises(ValueError, match="must equal"):
        load_config(overrides=["model.block_size=512", "data.block_size=1024"])


def test_unknown_key_is_rejected() -> None:
    """A typo must fail loudly, not be silently dropped."""
    with pytest.raises(ValueError, match="unknown key"):
        load_config(overrides=["model.n_layerz=6"])


def test_invalid_literal_is_rejected() -> None:
    with pytest.raises(ValueError, match="expected one of"):
        load_config(overrides=["model.norm=batchnorm"])
    with pytest.raises(ValueError, match="expected one of"):
        load_config(overrides=["optim.schedule=exponential"])


def test_override_values_are_typed_not_strings() -> None:
    cfg = load_config(
        overrides=[
            "optim.lr=3e-4",
            "model.tie_embeddings=false",
            "model.n_kv_head=null",
            "train.max_steps=100",
        ]
    )
    assert cfg.optim.lr == 3e-4 and isinstance(cfg.optim.lr, float)
    assert cfg.model.tie_embeddings is False
    assert cfg.train.max_steps == 100
    # n_kv_head=None is normalised to n_head by ModelConfig.__post_init__.
    assert cfg.model.n_kv_head == cfg.model.n_head


def test_base_inheritance_merges_deeply(tmp_path: Path) -> None:
    write(tmp_path / "base.yaml", {"model": {"n_layer": 8, "n_head": 8}, "optim": {"lr": 1e-3}})
    write(tmp_path / "child.yaml", {"_base_": "base.yaml", "model": {"n_layer": 12}})

    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.model.n_layer == 12  # overridden
    assert cfg.model.n_head == 8  # inherited
    assert cfg.optim.lr == 1e-3  # untouched section survives


def test_circular_base_is_detected(tmp_path: Path) -> None:
    write(tmp_path / "a.yaml", {"_base_": "b.yaml"})
    write(tmp_path / "b.yaml", {"_base_": "a.yaml"})
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def test_grad_accum_is_derived_from_token_budget() -> None:
    cfg = load_config(
        overrides=[
            "train.tokens_per_step=524288",
            "data.micro_batch_size=16",
            "data.block_size=1024",
            "model.block_size=1024",
        ]
    )
    assert cfg.grad_accum_steps(world_size=1) == 32
    # More GPUs, same effective batch: accumulation absorbs the difference, so the
    # optimisation is unchanged and only throughput moves.
    assert cfg.grad_accum_steps(world_size=8) == 4


def test_indivisible_batch_is_rejected() -> None:
    cfg = load_config(overrides=["train.tokens_per_step=524288", "data.micro_batch_size=7"])
    with pytest.raises(ValueError, match="not divisible"):
        cfg.grad_accum_steps(world_size=1)


def test_config_round_trips_through_yaml(tmp_path: Path) -> None:
    original = load_config("gpt2-124m")
    original.save(tmp_path / "saved.yaml")
    assert load_config(tmp_path / "saved.yaml").to_dict() == original.to_dict()


def test_missing_config_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist")


# --- The shipped configs must all be loadable; a broken one is only discovered
# --- otherwise when someone tries to start a paid GPU run with it.

SHIPPED = sorted(p for p in CONFIG_ROOT.rglob("*.yaml") if not p.name.startswith("_"))


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: str(p.relative_to(CONFIG_ROOT)))
def test_shipped_config_loads_and_has_consistent_batching(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.grad_accum_steps(world_size=1) >= 1
    assert cfg.train.max_steps > cfg.optim.warmup_steps


def test_ablation_arms_differ_from_their_baseline_in_one_axis_only() -> None:
    """The discipline the ablation study rests on.

    If an arm differs from ``_base.yaml`` in more than its own axis, whatever the
    run measures is not the thing named on the tin.
    """
    base = load_config(CONFIG_ROOT / "ablations" / "_base.yaml").to_dict()
    expected_axes = {
        "norm-rmsnorm": {("model", "norm")},
        "pos-rope": {("model", "pos_emb")},
        "pos-none": {("model", "pos_emb")},
        "mlp-swiglu": {("model", "mlp")},
        "untied-embeddings": {("model", "tie_embeddings")},
        "no-bias": {("model", "bias")},
        "gqa-2": {("model", "n_kv_head")},
        # wsd_decay_frac is restated in the arm but equals the default, so it is
        # not a difference — the arm genuinely varies only the schedule.
        "sched-wsd": {("optim", "schedule")},
        "wd-zero": {("optim", "weight_decay")},
        "lr-3e-4": {("optim", "lr")},
        "lr-3e-3": {("optim", "lr")},
    }

    for name, axes in expected_axes.items():
        arm = load_config(CONFIG_ROOT / "ablations" / f"{name}.yaml").to_dict()
        differences = {
            (section, key)
            for section, values in base.items()
            for key, value in values.items()
            if arm[section][key] != value and (section, key) != ("log", "run_name")
        }
        assert differences == axes, f"{name} varies {differences}, expected {axes}"
