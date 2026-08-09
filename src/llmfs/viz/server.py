"""``llmfs-viz-serve`` — the attention explorer with a model behind it.

Serves the same front end as the static export, plus a ``/api/attention`` endpoint
so arbitrary text can be run through the model on demand. Use this locally; ship the
static export for anything public.

Deliberately built on ``http.server`` rather than FastAPI. This is a single-user
local tool that runs one forward pass per request, so an async framework buys
nothing, and keeping it in the standard library means the visualisation has no
dependency that can rot.
"""

from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .attention import attention_for_prompt, load_model_and_tokenizer
from .export import DEFAULT_PROMPTS, render_html

MAX_PROMPT_CHARS = 4000


class AttentionService:
    """Holds the model and serialises access to it."""

    def __init__(self, checkpoint: str | Path, max_tokens: int = 64, device: str = "auto") -> None:
        self.model, self.tokenizer, self.info = load_model_and_tokenizer(checkpoint, device=device)
        self.max_tokens = max_tokens
        # One model, shared across handler threads. A forward pass mutates no model
        # state, but running several concurrently on one device just contends for it,
        # so requests are serialised rather than parallelised.
        self._lock = threading.Lock()

    def attention(self, prompt: str) -> dict[str, Any]:
        with self._lock:
            view = attention_for_prompt(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.max_tokens,
                device=self.info["device"],
            )
        return view.to_payload()

    def boot_payload(self) -> dict[str, Any]:
        cfg = self.model.cfg
        return {
            "mode": "live",
            "default_prompt": DEFAULT_PROMPTS[0],
            "model": {
                **self.info,
                "n_layer": cfg.n_layer,
                "n_head": cfg.n_head,
                "n_kv_head": cfg.n_kv_head,
                "norm": cfg.norm,
                "pos_emb": cfg.pos_emb,
                "mlp": cfg.mlp,
                "params": self.model.num_params(),
            },
            "views": {DEFAULT_PROMPTS[0]: self.attention(DEFAULT_PROMPTS[0])},
        }


def make_handler(service: AttentionService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "llmfs-viz"

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802 - http.server's interface
            if self.path in ("/", "/index.html"):
                html = render_html(service.boot_payload())
                self._send(200, html.encode(), "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._json(200, {"ok": True, **service.info})
            else:
                self._json(404, {"detail": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - http.server's interface
            if self.path.rstrip("/") != "/api/attention":
                return self._json(404, {"detail": "not found"})

            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_PROMPT_CHARS * 4:
                return self._json(413, {"detail": "request too large"})

            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                prompt = str(body.get("prompt", "")).strip()
                if not prompt:
                    return self._json(400, {"detail": "prompt is required"})
                if len(prompt) > MAX_PROMPT_CHARS:
                    return self._json(400, {"detail": f"prompt exceeds {MAX_PROMPT_CHARS} chars"})
                self._json(200, service.attention(prompt))
            except json.JSONDecodeError:
                self._json(400, {"detail": "body must be JSON"})
            except ValueError as exc:
                self._json(400, {"detail": str(exc)})

        def log_message(self, fmt: str, *args) -> None:
            print(f"[viz] {fmt % args}")

    return Handler


def serve(
    checkpoint: str | Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_tokens: int = 64,
    device: str = "auto",
) -> None:
    service = AttentionService(checkpoint, max_tokens=max_tokens, device=device)
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    print(
        f"attention explorer on http://{host}:{port}\n"
        f"  {service.info['run_name']} @ step {service.info['step']:,} "
        f"({service.model.cfg.n_layer}x{service.model.cfg.n_head} heads, "
        f"{service.info['device']})\n"
        f"  Ctrl-C to stop"
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        httpd.server_close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Serve the interactive attention explorer.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)
    serve(
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        max_tokens=args.max_tokens,
        device=args.device,
    )


if __name__ == "__main__":
    main()
