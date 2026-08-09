"""Throughput, memory and cost benchmarks.

``throughput`` covers the training-side knobs (compile, gradient checkpointing,
micro-batch size) and the inference baseline (naive re-forward vs KV cache, across
batch sizes). Quantization and speculative decoding will be measured against the
inference numbers established here.
"""

from .throughput import BenchResult, bench_inference, bench_training, write_results

__all__ = ["BenchResult", "bench_inference", "bench_training", "write_results"]
