"""Throughput, memory and cost benchmarks.

Two suites, both designed to run in minutes on the pod that just finished training —
because the alternative is renting a second one.

**Training** sweeps the knobs that trade memory against speed: ``torch.compile``,
gradient checkpointing, micro-batch size, precision. It uses synthetic batches rather
than the corpus, so it depends on nothing and measures only the model.

**Inference** measures decoding: the naive re-forward baseline against the KV cache,
across batch sizes, reporting tokens/sec, time-to-first-token and peak memory. This
is the substrate the quantization and speculative-decoding work will later be
compared against, so the baseline numbers need to exist first.

Every result carries the provenance block, because a throughput number without the
hardware that produced it cannot be checked by anyone.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import Config, load_config
from ..model import GenerationConfig, ModelConfig, Transformer
from ..train.optim import build_optimizer
from ..utils.device import autocast_context, get_device, peak_flops, resolve_dtype
from ..utils.provenance import capture


@dataclass
class BenchResult:
    suite: str
    variant: str
    settings: dict[str, Any]
    tokens_per_sec: float
    ms_per_step: float
    peak_memory_gib: float
    mfu: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _reset_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory_gib(device: torch.device) -> float:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**30
    return 0.0


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# ------------------------------------------------------------------- training


TRAINING_VARIANTS: list[dict[str, Any]] = [
    {"name": "baseline", "compile": False, "grad_checkpointing": False},
    {"name": "compile", "compile": True, "grad_checkpointing": False},
    {"name": "grad-checkpoint", "compile": False, "grad_checkpointing": True},
    {"name": "compile+checkpoint", "compile": True, "grad_checkpointing": True},
    {"name": "micro-batch x2", "compile": True, "grad_checkpointing": False, "batch_scale": 2},
    {"name": "checkpoint+batch x4", "compile": True, "grad_checkpointing": True, "batch_scale": 4},
]


def bench_training(
    cfg: Config,
    variants: list[dict[str, Any]] | None = None,
    steps: int = 20,
    warmup: int = 5,
    device: torch.device | None = None,
) -> list[BenchResult]:
    """Measure training throughput and peak memory across runtime settings.

    The final two variants exist to answer the question gradient checkpointing is
    actually for. It costs roughly 30% throughput on its own, which looks like a pure
    loss — but it frees enough activation memory to raise the micro-batch, and the
    comparison worth reporting is *checkpointing with a larger batch* against
    *no checkpointing at the batch that fits*.
    """
    variants = variants or TRAINING_VARIANTS
    device = device or get_device(cfg.runtime.device)
    dtype = resolve_dtype(cfg.runtime.dtype, device)
    results: list[BenchResult] = []

    for variant in variants:
        name = variant["name"]
        scale = variant.get("batch_scale", 1)
        micro_batch = cfg.data.micro_batch_size * scale
        settings = {
            "compile": variant.get("compile", False),
            "grad_checkpointing": variant.get("grad_checkpointing", False),
            "micro_batch_size": micro_batch,
            "block_size": cfg.data.block_size,
            "dtype": str(dtype).replace("torch.", ""),
        }

        model = runnable = optimizer = x = None
        try:
            _reset_memory(device)
            model = Transformer(cfg.model).to(device)
            if settings["grad_checkpointing"]:
                from ..train.trainer import Trainer

                Trainer._enable_grad_checkpointing(model)
            runnable = torch.compile(model) if settings["compile"] else model
            optimizer = build_optimizer(model, cfg.optim, device)

            # Synthetic batches: this measures the model, not the data pipeline, and
            # removes any dependency on a prepared corpus.
            x = torch.randint(
                0, cfg.model.vocab_size, (micro_batch, cfg.data.block_size), device=device
            )

            for i in range(warmup + steps):
                if i == warmup:
                    _sync(device)
                    _reset_memory(device)
                    start = time.perf_counter()
                with autocast_context(device, dtype):
                    loss = runnable(x, targets=x).loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            _sync(device)
            elapsed = time.perf_counter() - start

            tokens = steps * micro_batch * cfg.data.block_size
            tps = tokens / elapsed
            hw = peak_flops(device, dtype)
            results.append(
                BenchResult(
                    suite="training",
                    variant=name,
                    settings=settings,
                    tokens_per_sec=tps,
                    ms_per_step=elapsed / steps * 1000,
                    peak_memory_gib=_peak_memory_gib(device),
                    mfu=(model.estimate_mfu(tps, hw) if hw else None),
                )
            )
            print(
                f"  {name:<22} {tps:>10,.0f} tok/s  "
                f"{elapsed / steps * 1000:>7.1f} ms/step  "
                f"{_peak_memory_gib(device):>6.2f} GiB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            # A variant that does not fit is a result: it is the boundary the memory
            # optimisations exist to move.
            results.append(
                BenchResult("training", name, settings, 0.0, 0.0, 0.0, error=f"OOM: {exc}")
            )
            print(f"  {name:<22} OOM at micro_batch={micro_batch}")
        finally:
            # Release this variant's model and optimiser before the next one builds
            # its own. Without this the allocator holds both at once and every later
            # peak-memory reading is inflated by the variant before it. (Rebinding
            # the names is what actually drops the references — mutating `locals()`
            # would not, since it is a copy inside a function.)
            del model, runnable, optimizer, x
            _reset_memory(device)

    return results


# ------------------------------------------------------------------ inference


def bench_inference(
    model: Transformer,
    prompt_len: int = 64,
    gen_len: int = 128,
    batch_sizes: tuple[int, ...] = (1, 4, 16),
    device: torch.device | None = None,
    include_naive: bool = True,
) -> list[BenchResult]:
    """Decoding throughput and memory, cached against the naive baseline.

    The naive path re-runs the whole prefix for every token, so its cost grows with
    the square of the sequence. It is measured at batch 1 only — at larger batches it
    is slow enough to dominate the benchmark's own runtime without teaching anything
    the batch-1 number does not.
    """
    device = device or next(model.parameters()).device
    model.eval()
    results: list[BenchResult] = []

    for batch in batch_sizes:
        for use_cache in (True, False) if (include_naive and batch == 1) else (True,):
            label = "kv-cache" if use_cache else "naive (no cache)"
            name = f"{label} b{batch}"
            prompt = torch.randint(0, model.cfg.vocab_size, (batch, prompt_len), device=device)
            gen_cfg = GenerationConfig(max_new_tokens=gen_len, temperature=0.0, top_k=None)

            try:
                _reset_memory(device)
                model.generate(
                    prompt[:, :8], GenerationConfig(max_new_tokens=4), use_cache=use_cache
                )
                _sync(device)
                _reset_memory(device)

                # Time to first token: the prefill pass. It is what a user waits for
                # before anything appears, and it scales differently from the rest.
                start = time.perf_counter()
                cache = model.make_cache(batch, prompt_len + gen_len, device) if use_cache else None
                with torch.inference_mode():
                    model(prompt, cache=cache)
                _sync(device)
                ttft = time.perf_counter() - start

                start = time.perf_counter()
                model.generate(prompt, gen_cfg, use_cache=use_cache)
                _sync(device)
                elapsed = time.perf_counter() - start

                generated = batch * gen_len
                results.append(
                    BenchResult(
                        suite="inference",
                        variant=name,
                        settings={
                            "use_cache": use_cache,
                            "batch_size": batch,
                            "prompt_len": prompt_len,
                            "gen_len": gen_len,
                            "n_kv_head": model.cfg.n_kv_head,
                        },
                        tokens_per_sec=generated / elapsed,
                        ms_per_step=elapsed / gen_len * 1000,
                        peak_memory_gib=_peak_memory_gib(device),
                        extra={
                            "time_to_first_token_ms": ttft * 1000,
                            "total_seconds": elapsed,
                            "kv_cache_mib": (
                                model.make_cache(batch, prompt_len + gen_len, device).nbytes()
                                / 2**20
                                if use_cache
                                else 0.0
                            ),
                        },
                    )
                )
                print(
                    f"  {name:<22} {generated / elapsed:>10,.1f} tok/s  "
                    f"ttft {ttft * 1000:>6.1f} ms  {_peak_memory_gib(device):>6.2f} GiB"
                )
            except torch.cuda.OutOfMemoryError as exc:
                results.append(
                    BenchResult(
                        "inference", name, {"batch_size": batch}, 0.0, 0.0, 0.0, error=f"OOM: {exc}"
                    )
                )
                print(f"  {name:<22} OOM")
            finally:
                _reset_memory(device)

    return results


# ----------------------------------------------------------------------- CLI


def write_results(
    results: list[BenchResult], out: Path, device: torch.device, extra: dict | None = None
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": capture(device),
        "meta": extra or {},
        "results": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Throughput and memory benchmarks.")
    parser.add_argument("--suite", choices=["training", "inference", "both"], default="both")
    parser.add_argument("--config", type=str, default="gpt2-124m")
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="trained model for the inference suite"
    )
    parser.add_argument("--out", type=str, default="results/benchmarks.json")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--gen-len", type=int, default=128)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    device = get_device(args.device)
    results: list[BenchResult] = []

    if args.suite in ("training", "both"):
        print(f"training throughput ({args.config} on {device})")
        results += bench_training(cfg, steps=args.steps, device=device)

    if args.suite in ("inference", "both"):
        print(f"\ninference throughput ({device})")
        if args.checkpoint:
            from ..train.checkpoint import model_from_checkpoint

            model, _ = model_from_checkpoint(args.checkpoint, device=device)
        else:
            # Architecture determines decoding speed; the weights do not. An
            # untrained model gives the same throughput numbers.
            model = Transformer(ModelConfig(**cfg.to_dict()["model"])).to(device).eval()
        results += bench_inference(model, gen_len=args.gen_len, device=device)

    write_results(
        results, Path(args.out), device, extra={"config": args.config, "suite": args.suite}
    )


if __name__ == "__main__":
    main()
