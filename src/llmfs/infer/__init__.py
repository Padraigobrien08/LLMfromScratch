"""Inference optimisations: speculative decoding, and the drafters that feed it."""

from .speculative import (
    Drafter,
    ModelDrafter,
    PromptLookupDrafter,
    SpecStats,
    greedy_generate,
    speculative_generate,
)

__all__ = [
    "Drafter",
    "ModelDrafter",
    "PromptLookupDrafter",
    "SpecStats",
    "greedy_generate",
    "speculative_generate",
]
