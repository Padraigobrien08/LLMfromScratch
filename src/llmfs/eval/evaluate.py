"""``llmfs-eval`` — measure validation loss and perplexity for a checkpoint.

Reported over the whole validation split rather than a sample, because the number
this produces is the one quoted against the reproduction target and a 50-batch
estimate carries enough variance to move it by more than the tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from tqdm import tqdm

from ..data.loader import ShardDataLoader
from ..train.checkpoint import model_from_checkpoint
from ..utils.device import autocast_context, get_device, resolve_dtype


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str | Path,
    data_dir: str | None = None,
    split: str = "val",
    batch_size: int = 8,
    max_batches: int | None = None,
    device: str = "auto",
) -> dict[str, float]:
    dev = get_device(device)
    model, ckpt = model_from_checkpoint(checkpoint, device=dev)
    cfg = ckpt["config"]
    dtype = resolve_dtype(cfg["runtime"]["dtype"], dev)

    loader = ShardDataLoader(
        data_dir=data_dir or cfg["data"]["data_dir"],
        split=split,
        micro_batch_size=batch_size,
        block_size=cfg["data"]["block_size"],
        device=dev,
    )

    tokens_per_batch = batch_size * cfg["data"]["block_size"]
    total_batches = loader.total_tokens // tokens_per_batch
    if max_batches is not None:
        total_batches = min(total_batches, max_batches)

    total_loss = 0.0
    for _ in tqdm(range(total_batches), desc=f"eval[{split}]", unit="batch"):
        x, y = loader.next_batch()
        with autocast_context(dev, dtype):
            total_loss += model(x, targets=y).loss.item()

    mean_loss = total_loss / max(total_batches, 1)
    return {
        "checkpoint": str(checkpoint),
        "split": split,
        "step": ckpt["step"],
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20)),
        "batches": total_batches,
        "tokens_evaluated": total_batches * tokens_per_batch,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=None, help="write results to a JSON file")
    args = parser.parse_args(argv)

    results = evaluate_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        split=args.split,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        device=args.device,
    )
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
