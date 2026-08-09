"""Configuration: typed dataclasses, YAML files with inheritance, CLI overrides.

Three properties matter here, all of them in service of reproducibility:

1. **Typed.** Configs are dataclasses, so a typo in a YAML key is an error at load
   time rather than a silently-ignored setting that quietly changes a result.
2. **Composable.** A YAML file may declare ``_base_``, so every ablation is a file
   containing only the field it varies. Nothing can drift between an ablation and
   its baseline.
3. **Recorded.** The fully resolved config is written next to every checkpoint, so
   a run can always be reconstructed from its own output.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml

from .model.config import ModelConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


@dataclass
class DataConfig:
    data_dir: str = "data/fineweb-edu-10B"
    """Directory of prepared ``.bin`` shards, written by ``llmfs-prepare-data``."""
    tokenizer: str = "gpt2"
    """``gpt2`` for the reproduction, or ``file:<path>`` for a local tokenizer.json."""
    micro_batch_size: int = 12
    """Sequences per forward pass. Chosen to fit memory; the *effective* batch size
    is set by ``TrainConfig.tokens_per_step`` and reached via gradient accumulation."""
    block_size: int = 1024
    num_workers: int = 2


@dataclass
class OptimConfig:
    lr: float = 6e-4
    """Peak learning rate. GPT-2 124M uses 6e-4 at a 0.5M-token batch."""
    min_lr_ratio: float = 0.1
    """Floor of the decay schedule, as a fraction of ``lr``."""
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 700
    schedule: Literal["cosine", "linear", "wsd", "constant"] = "cosine"
    decay_steps: int | None = None
    """Steps over which to decay after warmup. Defaults to the full run length."""
    wsd_decay_frac: float = 0.1
    """For the ``wsd`` (warmup-stable-decay) schedule: fraction of the run spent decaying.
    Unlike cosine it does not need the run length fixed up front, which makes it the
    practical choice when a run might be extended."""


@dataclass
class RuntimeConfig:
    device: str = "auto"
    dtype: str = "auto"
    """``auto`` resolves to bf16 on capable CUDA hardware and fp32 elsewhere."""
    compile: bool = False
    grad_checkpointing: bool = False
    """Recompute activations in the backward pass: roughly 30% slower per step in
    exchange for a large memory saving, which usually buys back more than 30% via a
    bigger micro-batch. Measured in the training-efficiency benchmark."""
    tf32: bool = True
    seed: int = 1337
    deterministic: bool = False


@dataclass
class LogConfig:
    out_dir: str = "out"
    run_name: str = "run"
    log_interval: int = 10
    eval_interval: int = 250
    eval_steps: int = 50
    checkpoint_interval: int = 1000
    keep_last_n: int = 2
    """Rolling checkpoints to retain, in addition to the best-by-val-loss one.
    ``0`` keeps none; ``best.pt`` and ``final.pt`` are never pruned either way."""
    milestone_fracs: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75])
    """Fractions of the run at which to save a permanent, never-pruned checkpoint.

    Intermediate training states are the one artifact that cannot be recovered after
    the fact — reconstructing step 5,000 of a finished run means paying for the run
    again. They cost a few GB and make anything that studies training *dynamics*
    possible later: how attention heads specialise, when capabilities appear, how the
    loss curve decomposes. Set to ``[]`` to disable."""
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "llmfs"
    wandb_entity: str | None = None


@dataclass
class TrainConfig:
    max_steps: int = 19_073
    """Default is one epoch of FineWeb-Edu 10B at 524,288 tokens per step."""
    tokens_per_step: int = 524_288
    """Effective batch size in tokens (GPT-2's 0.5M). Gradient accumulation is derived
    from this, the micro-batch and the world size, so the optimisation is identical
    whether the run has one GPU or eight."""
    resume: str | None = None
    """Path to a checkpoint, or ``auto`` to pick up the latest in ``out_dir``."""


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    log: LogConfig = field(default_factory=LogConfig)

    def __post_init__(self) -> None:
        # The model's context length and the loader's sequence length are the same
        # number; keeping two fields in sync by hand is how configs quietly diverge.
        if self.model.block_size != self.data.block_size:
            raise ValueError(
                f"model.block_size ({self.model.block_size}) must equal "
                f"data.block_size ({self.data.block_size})"
            )

    def grad_accum_steps(self, world_size: int = 1) -> int:
        """Micro-batches per optimiser step needed to reach ``tokens_per_step``."""
        per_micro_step = self.data.micro_batch_size * self.data.block_size * world_size
        if self.train.tokens_per_step % per_micro_step != 0:
            raise ValueError(
                f"tokens_per_step ({self.train.tokens_per_step:,}) is not divisible by "
                f"micro_batch_size * block_size * world_size ({per_micro_step:,}). "
                f"Adjust micro_batch_size so the effective batch is exact."
            )
        return self.train.tokens_per_step // per_micro_step

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        text = (
            json.dumps(self.to_dict(), indent=2)
            if path.suffix == ".json"
            else yaml.safe_dump(self.to_dict(), sort_keys=False)
        )
        path.write_text(text)


# --------------------------------------------------------------------- loading


def _resolve_value(field_type: Any, value: Any) -> Any:
    """Coerce a YAML scalar to the field's declared type, rejecting bad literals.

    The coercion is not cosmetic. YAML 1.1's float pattern requires a decimal point
    before an exponent, so ``lr: 1e-4`` and ``--set optim.lr=3e-4`` both parse as
    *strings*. Left alone that reaches the optimiser as ``"3e-4"`` — the kind of
    fault that surfaces as a confusing crash hours into a paid run, or worse, does
    not surface at all. Coercing against the declared type fixes both the YAML and
    CLI paths at once.
    """
    origin = get_origin(field_type)

    if origin is Literal:
        allowed = get_args(field_type)
        if value not in allowed:
            raise ValueError(f"expected one of {allowed}, got {value!r}")
        return value

    # Unwrap optionals and unions to the concrete types they permit.
    candidates = [t for t in get_args(field_type) if t is not type(None)] or [field_type]

    if value is None:
        return None

    if isinstance(value, str):
        if float in candidates:
            try:
                return float(value)
            except ValueError:
                raise ValueError(f"expected a float, got {value!r}") from None
        if int in candidates and bool not in candidates:
            try:
                return int(value)
            except ValueError:
                raise ValueError(f"expected an int, got {value!r}") from None

    # An int where a float is declared (`lr: 1`) would otherwise stay an int and
    # propagate integer semantics into arithmetic downstream.
    if isinstance(value, int) and not isinstance(value, bool) and candidates == [float]:
        return float(value)

    return value


def _from_dict(cls: type, data: dict[str, Any]) -> Any:
    """Build a (possibly nested) dataclass, erroring on unknown keys.

    Type hints are resolved with ``get_type_hints`` rather than read off
    ``field.type``: under ``from __future__ import annotations`` the latter is the
    *string* ``"Literal['cosine', ...]"``, against which every Literal check would
    silently pass.
    """
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(
            f"unknown key(s) for {cls.__name__}: {sorted(unknown)}. Valid keys: {sorted(known)}"
        )

    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        hint = hints.get(name, Any)
        if isinstance(value, dict) and is_dataclass(hint):
            kwargs[name] = _from_dict(hint, value)
        else:
            try:
                kwargs[name] = _resolve_value(hint, value)
            except ValueError as exc:
                raise ValueError(f"{cls.__name__}.{name}: {exc}") from None
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _load_yaml_with_bases(path: Path, _seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config, recursively merging any ``_base_`` it declares."""
    path = path.resolve()
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"circular _base_ chain at {path}")
    _seen.add(path)

    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    raw = yaml.safe_load(path.read_text()) or {}

    bases = raw.pop("_base_", None)
    if bases is None:
        return raw
    if isinstance(bases, str):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        if not base_path.exists():
            base_path = (CONFIG_ROOT / base).resolve()
        merged = _deep_merge(merged, _load_yaml_with_bases(base_path, _seen))
    return _deep_merge(merged, raw)


def _apply_override(data: dict[str, Any], dotted: str) -> None:
    """Apply one ``section.key=value`` CLI override in place."""
    if "=" not in dotted:
        raise ValueError(f"override must look like key=value, got {dotted!r}")
    key, raw_value = dotted.split("=", 1)
    # Parsed as YAML so `true`, `3`, `1e-4`, `null` and `[1,2]` all arrive typed.
    value = yaml.safe_load(raw_value)

    parts = key.split(".")
    node = data
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"cannot descend into {part!r} in override {dotted!r}")
    node[parts[-1]] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a config from YAML and apply ``section.key=value`` overrides.

    Args:
        path: a YAML path, or a bare name resolved against ``configs/``.
        overrides: e.g. ``["model.norm=rmsnorm", "optim.lr=3e-4"]``.
    """
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.exists() and not p.is_absolute():
            candidate = CONFIG_ROOT / p
            p = candidate if candidate.exists() else CONFIG_ROOT / f"{p}.yaml"
        data = _load_yaml_with_bases(p)

    for override in overrides or []:
        _apply_override(data, override)

    return _from_dict(Config, data)
