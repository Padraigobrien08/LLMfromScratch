"""``llmfs-quant-eval`` — the quality/memory/speed trade-off table.

Runs a sweep of quantization settings against the same checkpoint and reports, for
each: model bytes, perplexity, HellaSwag accuracy, and decode throughput. These move
in different directions, so no single number decides the question — 4-bit is a clear
win on memory, a small loss on quality, and (without a fused kernel) a large loss on
speed.

Perplexity is the metric that resolves the quality question. HellaSwag cannot: a 4-way
accuracy over 1,000 examples carries a 1.5-point standard error, and every scheme here
lands inside it. It is still reported, as a check that nothing is catastrophically
broken.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..data.tokenizer import load_tokenizer
from ..eval.hellaswag import GPT2_124M_ACC_NORM, download, load_examples
from ..eval.hellaswag import evaluate as eval_hellaswag
from ..model import GenerationConfig
from ..train.checkpoint import model_from_checkpoint
from ..utils.device import autocast_context, get_device, resolve_dtype
from ..utils.provenance import capture
from .quantize import QuantConfig, model_memory_bytes, quantize_model

# The sweep. The per-tensor rows are included as the control that shows why grouping
# exists: one scale over a whole matrix is set by its largest outlier, and at 4 bits
# that is the difference between a usable model and a broken one.
#
# The tied embedding is never quantized here — with tie_embeddings the head shares its
# weight with nn.Embedding, so replacing it adds a quantized copy instead of
# substituting one, and quantize_model refuses. See docs/efficiency.md.
SWEEP: list[dict[str, Any]] = [
    {"name": "fp32 baseline", "bits": None},
    {"name": "int8 per-tensor", "bits": 8, "group_size": -1},
    {"name": "int8 g128", "bits": 8, "group_size": 128},
    {"name": "int4 per-tensor", "bits": 4, "group_size": -1},
    {"name": "int4 g128", "bits": 4, "group_size": 128},
    {"name": "int4 g32", "bits": 4, "group_size": 32},
]


@torch.no_grad()
def perplexity(model, tokens: torch.Tensor, dtype, block_size: int, stride: int = 512) -> float:
    """Perplexity over a token stream, in non-overlapping blocks.

    HellaSwag cannot answer this question: a 4-way accuracy over 1,000 examples has a
    standard error of 1.5 points, and every quantization scheme here lands inside it.
    Perplexity is continuous and computed over every token, so it resolves differences
    two orders of magnitude smaller for a fraction of the compute.
    """
    device = next(model.parameters()).device
    total_nll, total_tokens = 0.0, 0
    for start in range(0, tokens.numel() - block_size - 1, stride):
        window = tokens[start : start + block_size + 1].to(device)
        x, y = window[:-1].unsqueeze(0), window[1:].unsqueeze(0)
        with autocast_context(device, dtype):
            loss = model(x, targets=y).loss
        total_nll += loss.item() * y.numel()
        total_tokens += y.numel()
    return math.exp(total_nll / total_tokens) if total_tokens else float("nan")


def build_perplexity_corpus(examples, tokenizer, max_tokens: int = 200_000) -> torch.Tensor:
    """Real English from the HellaSwag validation text — local, and large enough.

    Held out from training in the sense that matters here: every scheme sees exactly
    the same tokens, so the comparison between them is what is being measured.
    """
    ids: list[int] = []
    for example in examples:
        correct = example["endings"][int(example["label"])]
        ids.extend(tokenizer.encode(example["ctx"] + " " + correct))
        if len(ids) >= max_tokens:
            break
    return torch.tensor(ids[:max_tokens], dtype=torch.long)


@dataclass
class QuantResult:
    name: str
    bits: int | None
    group_size: int | None
    quantized_embedding: bool
    memory_mib: float
    compression: float
    perplexity: float | None = None
    acc_norm: float | None = None
    acc: float | None = None
    decode_tok_s: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def measure_decode(model, tokenizer, device, gen_len: int = 64, prompt_len: int = 32) -> float:
    """Tokens/sec for single-stream greedy decoding with the KV cache."""
    prompt = torch.randint(0, model.cfg.vocab_size, (1, prompt_len), device=device)
    cfg = GenerationConfig(max_new_tokens=8, temperature=0.0, top_k=None)
    model.generate(prompt, cfg)  # warm up compile/alloc paths
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    model.generate(prompt, GenerationConfig(max_new_tokens=gen_len, temperature=0.0, top_k=None))
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    return gen_len / (time.perf_counter() - start)


def run_sweep(
    checkpoint: str | Path,
    device_str: str = "auto",
    hellaswag_limit: int | None = 1000,
    decode: bool = True,
    sweep: list[dict[str, Any]] | None = None,
) -> tuple[list[QuantResult], dict[str, Any]]:
    device = get_device(device_str)
    sweep = sweep or SWEEP

    _, ckpt = model_from_checkpoint(checkpoint, device="cpu")
    tokenizer = load_tokenizer(ckpt["config"]["data"]["tokenizer"])
    dtype = resolve_dtype(ckpt["config"]["runtime"]["dtype"], device)
    examples = load_examples(download())
    corpus = build_perplexity_corpus(examples, tokenizer)
    print(f"perplexity corpus: {corpus.numel():,} tokens")

    baseline_bytes: float | None = None
    results: list[QuantResult] = []

    for spec in sweep:
        name = spec["name"]
        print(f"\n--- {name} ---")
        # Reloaded each time: quantization mutates the model in place, so reusing one
        # would quantize an already-quantized model and report nonsense.
        model, _ = model_from_checkpoint(checkpoint, device="cpu")

        bits = spec.get("bits")
        info: dict[str, Any] = {}
        if bits is not None:
            cfg = QuantConfig(
                bits=bits,
                group_size=spec.get("group_size", 128),
                skip=spec.get("skip", ("lm_head",)),
            )
            info = quantize_model(model, cfg)
            print(f"  replaced {info['replaced']} layers, skipped {len(info['skipped'])}")

        memory = model_memory_bytes(model)
        if baseline_bytes is None:
            baseline_bytes = memory
        print(f"  memory {memory / 2**20:,.0f} MiB ({baseline_bytes / memory:.2f}x)")

        model = model.to(device).eval()
        result = QuantResult(
            name=name,
            bits=bits,
            group_size=spec.get("group_size") if bits else None,
            quantized_embedding=bits is not None and spec.get("skip", ("lm_head",)) == (),
            memory_mib=memory / 2**20,
            compression=baseline_bytes / memory,
            extra={k: v for k, v in info.items() if k != "skipped"},
        )

        result.perplexity = perplexity(model, corpus, dtype, ckpt["config"]["model"]["block_size"])
        print(f"  perplexity {result.perplexity:.4f}")

        if hellaswag_limit:
            scores = eval_hellaswag(model, tokenizer, examples, dtype, limit=hellaswag_limit)
            result.acc_norm, result.acc = scores["acc_norm"], scores["acc"]
            print(f"  hellaswag acc_norm {scores['acc_norm']:.4f} (n={scores['n_evaluated']})")

        if decode:
            result.decode_tok_s = measure_decode(model, tokenizer, device)
            print(f"  decode {result.decode_tok_s:.1f} tok/s")

        results.append(result)
        del model

    meta = {
        "checkpoint": str(checkpoint),
        "step": ckpt["step"],
        "device": str(device),
        "hellaswag_limit": hellaswag_limit,
        "hellaswag_reference": GPT2_124M_ACC_NORM,
        "provenance": capture(device, measure=False),
    }
    return results, meta


def render_table(results: list[QuantResult]) -> str:
    lines = [
        "| Scheme | Memory | vs fp32 | Perplexity | Δ ppl | HellaSwag | Decode |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    base = next((r for r in results if r.bits is None), None)
    for r in results:
        ppl = f"{r.perplexity:.3f}" if r.perplexity is not None else "—"
        dppl = (
            "—"
            if (r.perplexity is None or base is None or base.perplexity is None or r is base)
            else f"{r.perplexity - base.perplexity:+.3f}"
        )
        acc = f"{r.acc_norm:.4f}" if r.acc_norm is not None else "—"
        tok = f"{r.decode_tok_s:,.1f}" if r.decode_tok_s else "—"
        lines.append(
            f"| {r.name} | {r.memory_mib:,.0f} MiB | {r.compression:.2f}× | {ppl} | {dppl} "
            f"| {acc} | {tok} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Quantization quality/memory/speed sweep.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument(
        "--hellaswag-limit",
        type=int,
        default=1000,
        help="examples to score per scheme; 0 skips the quality measurement",
    )
    parser.add_argument("--no-decode", action="store_true")
    parser.add_argument("--out", type=str, default="results/quantization.json")
    args = parser.parse_args(argv)

    results, meta = run_sweep(
        args.checkpoint,
        device_str=args.device,
        hellaswag_limit=args.hellaswag_limit or None,
        decode=not args.no_decode,
    )

    print("\n" + render_table(results))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
