"""``llmfs-train`` entrypoint.

llmfs-train --config gpt2-124m
llmfs-train --config ablations/norm-rmsnorm.yaml --set optim.lr=3e-4
torchrun --nproc_per_node=8 -m llmfs.train.cli --config gpt2-124m
"""

from __future__ import annotations

import argparse

from ..config import load_config
from .distributed import cleanup_distributed
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a decoder-only language model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="debug",
        help="config name (resolved against configs/) or path to a YAML file",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set optim.lr=3e-4 (repeatable)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="checkpoint path, or 'auto' for the latest in the run directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build everything and print the run plan without training",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    if args.resume:
        cfg.train.resume = args.resume

    trainer = Trainer(cfg)
    try:
        if args.dry_run:
            print("[dry-run] setup complete; exiting without training")
            return
        state = trainer.train()
        seen = state.tokens_seen
        seen_str = f"{seen / 1e9:.2f}B" if seen >= 1e9 else f"{seen / 1e6:.1f}M"
        print(
            f"\ndone: {state.step:,} steps, {seen_str} tokens, "
            f"best val loss {state.best_val_loss:.4f}"
        )
    finally:
        cleanup_distributed(trainer.dist)


if __name__ == "__main__":
    main()
