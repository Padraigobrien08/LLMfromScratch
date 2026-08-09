"""Ablation sweep: running the arms, and reporting on them honestly."""

from .report import Comparison, baseline_noise, compare, render_markdown
from .sweep import ArmResult, run_arm, run_sweep

__all__ = [
    "ArmResult",
    "Comparison",
    "baseline_noise",
    "compare",
    "render_markdown",
    "run_arm",
    "run_sweep",
]
