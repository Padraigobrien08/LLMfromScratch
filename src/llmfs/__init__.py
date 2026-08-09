"""llmfs — a decoder-only language model built from scratch.

Layout:
    ``llmfs.model``  hand-written architecture components (RoPE, RMSNorm, SwiGLU, GQA, KV cache)
    ``llmfs.data``   tokenisation and the memory-mapped shard pipeline
    ``llmfs.train``  the training loop, optimiser and schedules
    ``llmfs.eval``   evaluation and generation
    ``llmfs.bench``  throughput, memory and cost benchmarks
"""

from .model import GenerationConfig, ModelConfig, Transformer

__version__ = "0.1.0"

__all__ = ["GenerationConfig", "ModelConfig", "Transformer", "__version__"]
