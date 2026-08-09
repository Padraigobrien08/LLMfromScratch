"""``llmfs-generate`` — sample from a trained checkpoint."""

from __future__ import annotations

import argparse
import time

import torch

from ..data.tokenizer import load_tokenizer
from ..model import GenerationConfig
from ..train.checkpoint import model_from_checkpoint
from ..utils.device import get_device


def generate_text(
    checkpoint: str,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int | None = 200,
    top_p: float | None = None,
    seed: int | None = None,
    device: str = "auto",
    use_cache: bool = True,
    num_samples: int = 1,
) -> list[str]:
    dev = get_device(device)
    model, ckpt = model_from_checkpoint(checkpoint, device=dev)
    tokenizer = load_tokenizer(ckpt["config"]["data"]["tokenizer"])

    ids = tokenizer.encode(prompt) if prompt else [tokenizer.eot_token]
    idx = torch.tensor(ids, dtype=torch.long, device=dev).unsqueeze(0)

    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        seed=seed,
    )

    outputs = []
    for i in range(num_samples):
        if seed is not None:
            gen_cfg.seed = seed + i  # distinct samples, still reproducible as a set
        start = time.perf_counter()
        out = model.generate(idx, gen_cfg, use_cache=use_cache)
        elapsed = time.perf_counter() - start

        generated = out[0, idx.shape[1] :].tolist()
        outputs.append(tokenizer.decode(generated))
        print(
            f"[sample {i + 1}/{num_samples}] {len(generated)} tokens in {elapsed:.2f}s "
            f"({len(generated) / elapsed:.1f} tok/s, cache={'on' if use_cache else 'off'})"
        )
    return outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate text from a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="disable the KV cache (the naive baseline the benchmarks compare against)",
    )
    args = parser.parse_args(argv)

    samples = generate_text(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        seed=args.seed,
        device=args.device,
        use_cache=not args.no_cache,
        num_samples=args.num_samples,
    )
    for i, text in enumerate(samples):
        print(f"\n{'-' * 68}\n[{i + 1}] {args.prompt}{text}")


if __name__ == "__main__":
    main()
