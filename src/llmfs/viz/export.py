"""``llmfs-viz`` — build a self-contained attention explorer from a checkpoint.

The output is a single HTML file with the attention data embedded in it. No build
step, no CDN, no backend: it opens from the filesystem and hosts on GitHub Pages
unchanged. That matters more than it sounds — the point of this tool is a URL a
reviewer can click, and anything with a server attached is a URL that will be down
the day someone looks at it.

Live exploration of arbitrary text needs a model in the loop; that is
``llmfs-viz-serve``, which reuses the same front end.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from .attention import attention_for_prompt, load_model_and_tokenizer

TEMPLATE = Path(__file__).parent / "template.html"

#: The latin subset of Source Serif 4 variable, inlined into the template as a data:
#: URI so the page keeps the site's masthead without fetching anything. Adobe's, under
#: the SIL Open Font License 1.1; the notice travels in the template's own comment.
FONT = Path(__file__).parent / "source-serif-4-latin.woff2"

DEFAULT_PROMPTS = [
    "Dorothy lived in the midst of the great Kansas prairies.",
    "The Scarecrow said to the Tin Woodman: the Tin Woodman said nothing.",
    "one two three four five six seven eight",
    "The capital of France is Paris. The capital of Italy is Rome. The capital of Spain is",
]


def render_html(payload: dict[str, Any]) -> str:
    """Inject the payload and the webfont into the template.

    ``<`` is escaped to ``\\u003c`` throughout. The JSON sits inside a ``<script>``
    element, where the HTML parser looks for a literal ``</script`` before the
    JavaScript parser ever sees the content — so a prompt containing that string
    would otherwise terminate the block early and break the page. Escaping the
    character is valid JSON and costs nothing.

    The font is base64'd in rather than linked, for the same reason everything else
    here is: the page has to open from a filesystem with no network. It adds about
    160KB to a file that already carries every attention weight of every prompt.
    """
    if not TEMPLATE.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(f"front-end template missing: {TEMPLATE}")
    if not FONT.exists():  # pragma: no cover - packaging guard
        raise FileNotFoundError(f"webfont missing: {FONT}")

    encoded = json.dumps(payload).replace("<", "\\u003c")
    html = TEMPLATE.read_text(encoding="utf-8")
    for placeholder in ("/*__LLMFS_DATA__*/", "/*__LLMFS_FONT__*/"):
        if placeholder not in html:  # pragma: no cover - packaging guard
            raise ValueError(f"template has no {placeholder} placeholder")

    font = base64.b64encode(FONT.read_bytes()).decode("ascii")
    return html.replace("/*__LLMFS_FONT__*/", font).replace("/*__LLMFS_DATA__*/", encoded)


def build_payload(
    checkpoint: str | Path,
    prompts: list[str],
    max_tokens: int = 64,
    device: str = "auto",
) -> dict[str, Any]:
    """Run every prompt and assemble the front end's boot payload."""
    model, tokenizer, info = load_model_and_tokenizer(checkpoint, device=device)

    views: dict[str, Any] = {}
    for prompt in prompts:
        view = attention_for_prompt(
            model, tokenizer, prompt, max_tokens=max_tokens, device=info["device"]
        )
        # Keyed by prompt text so the selector reads as the prompts themselves.
        views[prompt] = view.to_payload()

    return {
        "mode": "static",
        "model": {
            **info,
            "n_layer": model.cfg.n_layer,
            "n_head": model.cfg.n_head,
            "n_kv_head": model.cfg.n_kv_head,
            "norm": model.cfg.norm,
            "pos_emb": model.cfg.pos_emb,
            "mlp": model.cfg.mlp,
            "params": model.num_params(),
        },
        "views": views,
    }


def export(
    checkpoint: str | Path,
    out: str | Path,
    prompts: list[str] | None = None,
    max_tokens: int = 64,
    device: str = "auto",
) -> Path:
    payload = build_payload(
        checkpoint, prompts or DEFAULT_PROMPTS, max_tokens=max_tokens, device=device
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")

    size_kb = out.stat().st_size / 1024
    print(
        f"wrote {out} ({size_kb:,.0f} KB) — "
        f"{len(payload['views'])} prompts, "
        f"{payload['model']['n_layer']}x{payload['model']['n_head']} heads"
    )
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained attention explorer from a checkpoint."
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--out", type=str, default="site/attention.html")
    parser.add_argument(
        "--prompt",
        action="append",
        dest="prompts",
        help="prompt to precompute (repeatable; defaults to a built-in set)",
    )
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="file with one prompt per line, used instead of --prompt",
    )
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    prompts = args.prompts
    if args.prompts_file:
        lines = Path(args.prompts_file).read_text(encoding="utf-8").splitlines()
        prompts = [ln.strip() for ln in lines if ln.strip()]

    export(
        checkpoint=args.checkpoint,
        out=args.out,
        prompts=prompts,
        max_tokens=args.max_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
