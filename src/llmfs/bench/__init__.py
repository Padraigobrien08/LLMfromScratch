"""Throughput, memory and cost benchmarks.

Populated in the efficiency phase: the consolidated naive -> KV cache -> quantized
-> speculative-decoding comparison, plus training-side throughput measurements for
torch.compile, gradient checkpointing and mixed precision.
"""
