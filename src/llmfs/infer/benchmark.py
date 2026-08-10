"""``llmfs-spec-bench`` — measure what speculative decoding actually buys.

Reports, per drafter and per prompt: wall-clock tokens/sec against ordinary greedy
decoding, the acceptance rate, and tokens produced per target forward pass. It also
verifies the output is identical to greedy decoding on every run, because a speedup
that changed the output would not be a speedup.

Two things this is designed to expose rather than hide:

* **Acceptance rate and speedup are different questions.** A drafter can be accurate
  and still lose, if proposing costs as much as the token it saves. The
  ``tokens_per_target_forward`` column is the theoretical ceiling; the wall-clock
  column is what you actually get.
* **The drafter has to be much cheaper than the target.** Using a same-sized model as
  the drafter is measured here precisely to show that it cannot win, which is the
  clearest way to make the point that speculation is a compute-for-latency trade and
  the draft must be small.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..data.tokenizer import load_tokenizer
from ..train.checkpoint import model_from_checkpoint
from ..utils.device import get_device
from ..utils.provenance import capture
from .speculative import (
    Drafter,
    ModelDrafter,
    PromptLookupDrafter,
    greedy_generate,
    speculative_generate,
)

# The middle prompt is deliberately repetitive: prompt-lookup drafting can only work
# where the context predicts itself, and a benchmark that used only free-form prose
# would report it as useless rather than as conditional.
PROMPTS = {
    "prose": "The history of the printing press begins in",
    "repetitive": (
        "Item 1: apples. Item 2: oranges. Item 3: pears. Item 1: apples. Item 2: oranges. Item 3:"
    ),
    "code-ish": "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
}


@dataclass
class SpecBenchResult:
    prompt: str
    drafter: str
    k: int
    tokens_per_sec: float
    speedup: float
    acceptance_rate: float | None
    tokens_per_target_forward: float | None
    target_forwards: int | None
    drafter_forwards: int | None
    output_matches_greedy: bool | None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def run(
    checkpoint: str | Path,
    draft_checkpoint: str | Path | None = None,
    device_str: str = "auto",
    max_new_tokens: int = 96,
    ks: tuple[int, ...] = (2, 4, 8),
) -> tuple[list[SpecBenchResult], dict[str, Any]]:
    device = get_device(device_str)
    target, ckpt = model_from_checkpoint(checkpoint, device=device)
    tokenizer = load_tokenizer(ckpt["config"]["data"]["tokenizer"])

    drafters: list[tuple[str, Drafter]] = [("prompt-lookup", PromptLookupDrafter(max_ngram=3))]
    if draft_checkpoint:
        draft_model, _ = model_from_checkpoint(draft_checkpoint, device=device)
        drafters.append(("model-draft", ModelDrafter(draft_model, name="model-draft")))

    results: list[SpecBenchResult] = []

    for label, text in PROMPTS.items():
        prompt = torch.tensor([tokenizer.encode(text)], device=device)
        print(f"\n=== {label} ===")

        # Baseline, and the reference output every speculative run must reproduce.
        greedy_generate(target, prompt, max_new_tokens=4)  # warm up
        _sync(device)
        start = time.perf_counter()
        reference = greedy_generate(target, prompt, max_new_tokens=max_new_tokens)
        _sync(device)
        baseline_tps = max_new_tokens / (time.perf_counter() - start)
        print(f"  greedy baseline        {baseline_tps:>8.1f} tok/s")
        results.append(
            SpecBenchResult(
                label, "greedy (baseline)", 0, baseline_tps, 1.0, None, None, None, None, None
            )
        )

        for name, drafter in drafters:
            for k in ks:
                _sync(device)
                start = time.perf_counter()
                out, stats = speculative_generate(
                    target, drafter, prompt, max_new_tokens=max_new_tokens, k=k
                )
                _sync(device)
                tps = max_new_tokens / (time.perf_counter() - start)
                matches = torch.equal(out, reference)
                results.append(
                    SpecBenchResult(
                        prompt=label,
                        drafter=name,
                        k=k,
                        tokens_per_sec=tps,
                        speedup=tps / baseline_tps,
                        acceptance_rate=stats.acceptance_rate,
                        tokens_per_target_forward=stats.tokens_per_target_forward,
                        target_forwards=stats.target_forwards,
                        drafter_forwards=stats.drafter_forwards,
                        output_matches_greedy=matches,
                    )
                )
                flag = "" if matches else "  ** OUTPUT DIVERGED **"
                print(
                    f"  {name:<14} k={k:<2} {tps:>8.1f} tok/s  "
                    f"{tps / baseline_tps:>5.2f}x  accept {stats.acceptance_rate:>5.1%}  "
                    f"tok/fwd {stats.tokens_per_target_forward:>4.2f}{flag}"
                )

    meta = {
        "checkpoint": str(checkpoint),
        "draft_checkpoint": str(draft_checkpoint) if draft_checkpoint else None,
        "max_new_tokens": max_new_tokens,
        "device": str(device),
        "provenance": capture(device, measure=False),
    }
    return results, meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark speculative decoding.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--draft-checkpoint",
        type=str,
        default=None,
        help="a smaller model of the same vocabulary to draft with",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--out", type=str, default="results/speculative.json")
    args = parser.parse_args(argv)

    results, meta = run(
        args.checkpoint,
        draft_checkpoint=args.draft_checkpoint,
        device_str=args.device,
        max_new_tokens=args.max_new_tokens,
    )

    diverged = [r for r in results if r.output_matches_greedy is False]
    if diverged:
        print(f"\n** {len(diverged)} run(s) diverged from greedy decoding — that is a bug **")
    else:
        print("\nevery speculative run reproduced greedy decoding exactly")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"meta": meta, "results": [asdict(r) for r in results]}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
