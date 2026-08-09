"""Optimiser construction and learning-rate schedules."""

from __future__ import annotations

import inspect
import math

import torch

from ..config import OptimConfig


def build_optimizer(
    model: torch.nn.Module, cfg: OptimConfig, device: torch.device
) -> torch.optim.AdamW:
    """AdamW with decay applied only to matmul weights.

    Uses the fused CUDA implementation where available: it collapses the per-tensor
    element-wise update into a handful of kernels, which is a measurable win for a
    model with hundreds of small parameter tensors.
    """
    groups = model.param_groups(weight_decay=cfg.weight_decay)

    kwargs: dict = dict(lr=cfg.lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps)
    if device.type == "cuda" and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True

    return torch.optim.AdamW(groups, **kwargs)


def lr_at_step(step: int, cfg: OptimConfig, max_steps: int) -> float:
    """Learning rate for ``step``, under the configured schedule.

    All schedules share a linear warmup. Warmup exists because Adam's second-moment
    estimate is meaningless for the first few hundred steps, and a full-rate update
    against a garbage denominator is the most common cause of an early loss spike.
    """
    warmup = cfg.warmup_steps
    min_lr = cfg.lr * cfg.min_lr_ratio

    if warmup > 0 and step < warmup:
        # step + 1 so the very first step is not a zero-length update.
        return cfg.lr * (step + 1) / warmup

    if cfg.schedule == "constant":
        return cfg.lr

    decay_steps = cfg.decay_steps or (max_steps - warmup)

    if cfg.schedule == "wsd":
        # Warmup-stable-decay: hold the peak rate, then decay over the last
        # `wsd_decay_frac` of the run. Unlike cosine this does not need the run
        # length fixed in advance — a run can be extended by moving the decay.
        decay_start = max_steps - int(max_steps * cfg.wsd_decay_frac)
        if step < decay_start:
            return cfg.lr
        progress = (step - decay_start) / max(max_steps - decay_start, 1)
        return cfg.lr - (cfg.lr - min_lr) * min(progress, 1.0)

    progress = (step - warmup) / max(decay_steps, 1)
    progress = min(max(progress, 0.0), 1.0)

    if cfg.schedule == "linear":
        return cfg.lr - (cfg.lr - min_lr) * progress
    if cfg.schedule == "cosine":
        return min_lr + 0.5 * (cfg.lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    raise ValueError(f"unknown schedule: {cfg.schedule!r}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr
