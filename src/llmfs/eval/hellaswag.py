"""Zero-shot HellaSwag, the downstream check on the reproduction.

Validation loss is a per-token number measured on a corpus of our own choosing. It
can look correct while the tokenizer, the validation split or the evaluation harness
quietly differ from whatever produced the published figure being compared against —
and every one of those mistakes moves the loss without moving anything a reader
would recognise as wrong. A downstream task is far less forgiving: get the setup
wrong and accuracy collapses toward the 25% floor.

Method (the standard one, so the number is comparable):
each example has a context and four candidate endings, exactly one correct. Every
ending is scored by the model's summed log-likelihood over *the ending tokens only*,
and the highest-scoring candidate is the prediction. Two figures are reported:

``acc``       raw summed log-likelihood — biased toward short endings
``acc_norm``  divided by ending token count — the figure usually quoted

Reference point: OpenAI's GPT-2 124M scores ≈0.2955 ``acc_norm``. Chance is 0.25, so
a 124M model clears the floor by only a few points; a result near 0.25 means
something is broken, not that the model is merely small.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..data.tokenizer import Tokenizer, load_tokenizer
from ..model import Transformer
from ..train.checkpoint import model_from_checkpoint
from ..utils.device import autocast_context, get_device, resolve_dtype
from ..utils.provenance import capture

HELLASWAG_VAL_URL = (
    "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
)
GPT2_124M_ACC_NORM = 0.2955
CHANCE = 0.25


def download(cache_dir: str | Path = "data/hellaswag") -> Path:
    """Fetch the validation split, cached. ~10k examples, a few MB."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "hellaswag_val.jsonl"
    if not path.exists():
        print(f"downloading HellaSwag validation set -> {path}")
        try:
            with urllib.request.urlopen(HELLASWAG_VAL_URL) as response:  # noqa: S310
                # Written via a temporary file: an interrupted download must not leave
                # a truncated cache that every later run silently trusts.
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(response.read())
                tmp.replace(path)
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"could not download HellaSwag: {exc}\n"
                f"  On a python.org macOS build this is usually missing CA certificates.\n"
                f"  Fetch it manually instead:\n"
                f"    curl -sSL {HELLASWAG_VAL_URL} -o {path}"
            ) from exc
    return path


def load_examples(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def render(example: dict[str, Any], tokenizer: Tokenizer) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build the 4 candidate sequences and a mask over the ending tokens.

    Returns ``(tokens, mask, label)`` where tokens is ``(4, T)`` right-padded and mask
    marks the positions that belong to the ending. Only masked positions are scored —
    the shared context would otherwise contribute the same constant to all four and
    dilute the comparison.
    """
    context_tokens = tokenizer.encode(example["ctx"])
    rows, masks = [], []
    for ending in example["endings"]:
        # The leading space belongs to the ending: byte-level BPE encodes it into the
        # first token, and dropping it changes the tokenisation of that token.
        ending_tokens = tokenizer.encode(" " + ending)
        rows.append(context_tokens + ending_tokens)
        masks.append([0] * len(context_tokens) + [1] * len(ending_tokens))

    width = max(len(r) for r in rows)
    tokens = torch.zeros(4, width, dtype=torch.long)
    mask = torch.zeros(4, width, dtype=torch.long)
    for i, (row, m) in enumerate(zip(rows, masks)):
        tokens[i, : len(row)] = torch.tensor(row)
        mask[i, : len(m)] = torch.tensor(m)
    return tokens, mask, int(example["label"])


@torch.no_grad()
def score_example(
    model: Transformer, tokens: torch.Tensor, mask: torch.Tensor, dtype: torch.dtype
) -> tuple[int, int]:
    """Return ``(argmin_sum_loss, argmin_mean_loss)`` over the four candidates."""
    device = next(model.parameters()).device
    tokens, mask = tokens.to(device), mask.to(device)

    with autocast_context(device, dtype):
        logits = model(tokens, targets=tokens).logits

    # Shift: position i predicts token i+1, so the loss for target token j lives at
    # logit position j-1.
    shift_logits = logits[:, :-1, :].float()
    shift_targets = tokens[:, 1:]
    shift_mask = mask[:, 1:]

    losses = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_targets.reshape(-1),
        reduction="none",
    ).view(shift_targets.shape)

    masked = losses * shift_mask
    summed = masked.sum(dim=1)
    counts = shift_mask.sum(dim=1).clamp(min=1)
    return int(summed.argmin().item()), int((summed / counts).argmin().item())


def evaluate(
    model: Transformer,
    tokenizer: Tokenizer,
    examples: list[dict[str, Any]],
    dtype: torch.dtype,
    limit: int | None = None,
) -> dict[str, Any]:
    model.eval()
    examples = examples[:limit] if limit else examples

    correct = correct_norm = total = skipped = 0
    for example in tqdm(examples, desc="hellaswag", unit="ex"):
        tokens, mask, label = render(example, tokenizer)
        if tokens.shape[1] > model.cfg.block_size:
            # Truncating would change what the model sees relative to the reference
            # setup; skipping and reporting the count is the honest option.
            skipped += 1
            continue
        pred, pred_norm = score_example(model, tokens, mask, dtype)
        correct += pred == label
        correct_norm += pred_norm == label
        total += 1

    return {
        "acc": correct / total if total else 0.0,
        "acc_norm": correct_norm / total if total else 0.0,
        "n_evaluated": total,
        "n_skipped_too_long": skipped,
        "chance": CHANCE,
        "gpt2_124m_reference_acc_norm": GPT2_124M_ACC_NORM,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Zero-shot HellaSwag evaluation.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default="data/hellaswag")
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    device = get_device(args.device)
    model, ckpt = model_from_checkpoint(args.checkpoint, device=device)
    tokenizer = load_tokenizer(ckpt["config"]["data"]["tokenizer"])
    dtype = resolve_dtype(ckpt["config"]["runtime"]["dtype"], device)

    examples = load_examples(download(args.data_dir))
    results = evaluate(model, tokenizer, examples, dtype, limit=args.limit)
    results.update(
        {
            "checkpoint": str(args.checkpoint),
            "step": ckpt["step"],
            "provenance": capture(device, measure=False),
        }
    )

    print(json.dumps({k: v for k, v in results.items() if k != "provenance"}, indent=2))
    verdict = (
        "above the GPT-2 124M reference"
        if results["acc_norm"] >= GPT2_124M_ACC_NORM
        else "below the GPT-2 124M reference"
    )
    print(
        f"\nacc_norm {results['acc_norm']:.4f} vs {GPT2_124M_ACC_NORM} — {verdict}. "
        f"(chance is {CHANCE}; near it means something is broken, not merely small.)"
    )

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
