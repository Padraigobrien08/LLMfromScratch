"""Checkpoint saving and loading.

A checkpoint has to carry everything needed to continue a run as though it had
never stopped: weights, optimiser moments, the step counter, and the config that
produced them. Data-loader position is deliberately *not* stored — it is derived
from the step (see :mod:`llmfs.data.loader`), which removes a whole class of
resume bugs where the stored position and the stored step disagree.

Writes are atomic: a checkpoint is written to a temporary file and renamed. A
process killed mid-write therefore leaves the previous checkpoint intact rather
than a truncated file that fails to load — the difference between losing an hour
and losing the run.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import torch

from ..config import Config
from ..model import ModelConfig, Transformer


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    config: Config,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Unwrap DDP/compile wrappers so the checkpoint reloads into a bare model and
    # is not tied to the topology or torch version that produced it.
    raw_model = unwrap_model(model)

    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "config": dataclasses.asdict(config),
        "metrics": metrics or {},
        "torch_version": torch.__version__,
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)  # atomic on POSIX


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Strip DDP and ``torch.compile`` wrappers to get at the underlying module."""
    while True:
        if hasattr(model, "module") and isinstance(model.module, torch.nn.Module):
            model = model.module
            continue
        # torch.compile stores the original under _orig_mod.
        orig = getattr(model, "_orig_mod", None)
        if isinstance(orig, torch.nn.Module):
            model = orig
            continue
        return model


def model_from_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[Transformer, dict]:
    """Rebuild a model from a checkpoint's own recorded config.

    The architecture is read from the checkpoint rather than passed in, so loading
    cannot silently mismatch the weights it is loading.
    """
    ckpt = load_checkpoint(path, map_location=device)
    model_cfg = ModelConfig(**ckpt["config"]["model"])
    model = Transformer(model_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model, ckpt


def find_latest_checkpoint(out_dir: str | Path) -> Path | None:
    """Most recent rolling checkpoint in ``out_dir``, or ``None``."""
    candidates = sorted(Path(out_dir).glob("ckpt_step*.pt"))
    return candidates[-1] if candidates else None


def prune_checkpoints(out_dir: str | Path, keep_last_n: int) -> None:
    """Delete all but the newest ``keep_last_n`` rolling checkpoints.

    ``best.pt`` and ``final.pt`` do not match the glob and are never pruned.
    """
    if keep_last_n <= 0:
        return
    checkpoints = sorted(Path(out_dir).glob("ckpt_step*.pt"))
    for stale in checkpoints[:-keep_last_n]:
        stale.unlink(missing_ok=True)
