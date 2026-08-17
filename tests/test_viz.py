"""Attention visualisation: extraction, head statistics, and export.

The statistics tests build attention patterns whose answers are known by
construction — a pure previous-token head, a pure sink head, uniform attention —
rather than checking a real model's output against itself. A statistic that is
subtly wrong still produces plausible numbers on real weights, and the whole point
of the panel is to be trusted when it says a head does something.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from conftest import gpt2_tokenizer, tiny_config
from llmfs.config import load_config
from llmfs.export.web import ROOT
from llmfs.model import Transformer
from llmfs.train.checkpoint import save_checkpoint
from llmfs.viz.attention import attention_for_prompt, compute_head_stats, display_token
from llmfs.viz.export import TEMPLATE, build_payload, export, render_html


def stats_for(matrix: np.ndarray):
    """Wrap a single (T, T) attention matrix as a one-layer, one-head array."""
    return compute_head_stats(matrix[None, None].astype(np.float32))[0]


def causal_uniform(T: int) -> np.ndarray:
    m = np.tril(np.ones((T, T)))
    return m / m.sum(axis=-1, keepdims=True)


# ------------------------------------------------------------------ statistics


def test_identity_head_is_maximally_focused() -> None:
    """Each query attends only to itself: zero entropy, zero distance."""
    s = stats_for(np.eye(8))
    assert s.entropy == pytest.approx(0.0, abs=1e-6)
    assert s.mean_distance == pytest.approx(0.0, abs=1e-6)
    assert s.prev_token_fraction == pytest.approx(0.0, abs=1e-6)
    assert s.sink_fraction == pytest.approx(0.0, abs=1e-6)


def test_previous_token_head_is_identified() -> None:
    T = 8
    m = np.zeros((T, T))
    m[0, 0] = 1.0  # position 0 has no predecessor and must attend to itself
    for i in range(1, T):
        m[i, i - 1] = 1.0

    s = stats_for(m)
    assert s.prev_token_fraction == pytest.approx(1.0)
    assert s.entropy == pytest.approx(0.0, abs=1e-6)
    # Query 0 contributes distance 0; the other T-1 queries each contribute 1.
    assert s.mean_distance == pytest.approx((T - 1) / T)


def test_sink_head_is_identified() -> None:
    T = 8
    m = np.zeros((T, T))
    m[:, 0] = 1.0
    s = stats_for(m)
    assert s.sink_fraction == pytest.approx(1.0)

    # The two statistics genuinely overlap at exactly one position: for query 1,
    # position 0 *is* the previous token. So a pure sink head scores 1/(T-1) on
    # prev-token, and the overlap is proportionally larger on short prompts. Worth
    # pinning, because it means the two columns should be read together rather than
    # either one alone.
    assert s.prev_token_fraction == pytest.approx(1.0 / (T - 1))


def test_uniform_attention_reaches_maximum_entropy() -> None:
    """Normalised entropy is 1 exactly when a query spreads over everything it can see.

    Without normalising by log(position+1) early tokens would score low purely
    because they have fewer positions available, and every head would look focused
    at the start of a sequence.
    """
    T = 16
    s = stats_for(causal_uniform(T))
    assert s.entropy == pytest.approx((T - 1) / T, abs=1e-3)


def test_entropy_ordering_is_meaningful() -> None:
    focused, diffuse = stats_for(np.eye(16)), stats_for(causal_uniform(16))
    assert focused.entropy < diffuse.entropy


def test_position_zero_never_inflates_the_fractions() -> None:
    """Position 0 trivially attends to itself with weight 1.

    Counting it would add 1/T to every head's sink score regardless of behaviour,
    making a genuine sink head indistinguishable from an inert one on short prompts.
    """
    T = 4
    m = np.eye(T)
    assert stats_for(m).sink_fraction == pytest.approx(0.0, abs=1e-6)


def test_stats_cover_every_head() -> None:
    weights = np.random.default_rng(0).random((3, 5, 6, 6)).astype(np.float32)
    weights /= weights.sum(-1, keepdims=True)
    stats = compute_head_stats(weights)
    assert len(stats) == 15
    assert {(s.layer, s.head) for s in stats} == {(x, y) for x in range(3) for y in range(5)}


# ------------------------------------------------------------------ extraction


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    cfg = load_config("debug")
    cfg.model = tiny_config(vocab_size=50304, n_layer=2, n_head=4, n_embd=64, block_size=64)
    cfg.data.block_size = 64
    model = Transformer(cfg.model).eval()
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, model, None, step=7, config=cfg)
    return path


@pytest.fixture
def model_and_tokenizer():
    cfg = tiny_config(vocab_size=50304, n_layer=2, n_head=4, n_embd=64, block_size=64)
    return Transformer(cfg).eval(), gpt2_tokenizer()


def test_extracted_weights_are_causal_and_normalised(model_and_tokenizer) -> None:
    """The visualisation must never show a query attending to its own future."""
    model, tokenizer = model_and_tokenizer
    view = attention_for_prompt(model, tokenizer, "The wizard of Oz stood there.")

    assert view.weights.shape == (2, 4, view.n_tokens, view.n_tokens)
    np.testing.assert_allclose(view.weights.sum(-1), 1.0, atol=1e-5)

    upper = np.triu(np.ones((view.n_tokens, view.n_tokens), dtype=bool), k=1)
    assert view.weights[:, :, upper].max() == 0.0


def test_token_count_matches_the_tokenizer(model_and_tokenizer) -> None:
    model, tokenizer = model_and_tokenizer
    prompt = "The wizard of Oz"
    view = attention_for_prompt(model, tokenizer, prompt)
    assert view.token_ids == tokenizer.encode(prompt)
    assert len(view.tokens) == len(view.token_ids)


def test_long_prompts_are_truncated_from_the_left(model_and_tokenizer) -> None:
    """Truncation keeps the most recent context, and says that it happened."""
    model, tokenizer = model_and_tokenizer
    view = attention_for_prompt(model, tokenizer, "word " * 200, max_tokens=16)
    assert view.n_tokens == 16
    assert view.meta["truncated"] is True
    assert view.token_ids == tokenizer.encode("word " * 200)[-16:]


def test_empty_prompt_is_rejected(model_and_tokenizer) -> None:
    model, tokenizer = model_and_tokenizer
    with pytest.raises(ValueError, match="empty"):
        attention_for_prompt(model, tokenizer, "   ")


def test_display_token_makes_whitespace_visible(model_and_tokenizer) -> None:
    """Byte-level BPE puts the leading space inside the token, so " the" and "the"
    are different tokens; the display must not hide that."""
    _, tokenizer = model_and_tokenizer
    with_space = tokenizer.encode(" the")[0]
    assert display_token(tokenizer, with_space).startswith("␣")


# --------------------------------------------------------------------- export


def test_payload_quantisation_round_trips_within_tolerance(model_and_tokenizer) -> None:
    """uint8 quantisation must not move a weight by more than half a step."""
    import base64

    model, tokenizer = model_and_tokenizer
    view = attention_for_prompt(model, tokenizer, "The wizard of Oz stood there.")
    payload = view.to_payload()

    decoded = np.frombuffer(base64.b64decode(payload["weights_b64"]), dtype=np.uint8)
    decoded = decoded.reshape(payload["weights_shape"]).astype(np.float32) / 255.0
    assert np.abs(decoded - view.weights).max() <= 1.0 / 255.0 / 2 + 1e-6


def test_render_html_escapes_script_terminators() -> None:
    """A prompt containing </script> must not be able to break out of the payload.

    The HTML parser scans for the literal string before any JavaScript runs, so this
    is a page-breaking bug — and, with attacker-controlled text, an injection.
    """
    hostile = "</script><img src=x onerror=alert(1)>"
    payload = {"mode": "static", "model": {}, "views": {hostile: {}}}
    html = render_html(payload)

    body = html.split('id="payload"')[1].split("</script>")[0]
    assert "</script" not in body
    assert "\\u003c/script" in body
    assert "\\u003cimg" in body


def test_export_is_self_contained(checkpoint: Path, tmp_path: Path) -> None:
    """No external resource may be referenced: the page has to work offline and on a
    static host with no network access."""
    out = export(checkpoint, tmp_path / "attention.html", prompts=["hello world"], device="cpu")
    html = out.read_text()

    assert "<script src=" not in html
    assert 'rel="stylesheet"' not in html
    assert "@import" not in html
    # url(...) may only be a data: URI generated at runtime, never a remote fetch.
    assert not re.search(r"url\(\s*['\"]?https?://", html)
    # No element may *fetch* from the network. Anchor hrefs are exempt — a link the
    # reader can click is not a resource the page loads.
    fetched = re.findall(r'src="(https?://[^"]+)"', html)
    fetched += re.findall(r'<link[^>]+href="(https?://[^"]+)"', html)
    assert not fetched, f"external resources referenced: {fetched}"


def test_export_embeds_a_usable_payload(checkpoint: Path, tmp_path: Path) -> None:
    out = export(
        checkpoint, tmp_path / "a.html", prompts=["hello world", "second one"], device="cpu"
    )
    html = out.read_text()

    raw = html.split('id="payload" type="application/json">')[1].split("</script>")[0]
    payload = json.loads(raw)

    assert payload["mode"] == "static"
    assert set(payload["views"]) == {"hello world", "second one"}
    assert payload["model"]["n_layer"] == 2 and payload["model"]["n_head"] == 4
    assert payload["model"]["step"] == 7

    view = payload["views"]["hello world"]
    L, H, T_q, T_k = view["weights_shape"]
    assert (L, H) == (2, 4)
    assert T_q == T_k == view["n_tokens"]
    assert len(view["stats"]) == L * H


def test_build_payload_reports_the_architecture(checkpoint: Path) -> None:
    """The header chips come from the checkpoint, so they cannot describe a
    different model than the one whose weights are shown."""
    payload = build_payload(checkpoint, ["hello"], device="cpu")
    model = payload["model"]
    assert model["norm"] in ("layernorm", "rmsnorm")
    assert model["params"] > 0
    assert model["step"] == 7


def test_viz_weights_match_a_direct_model_call(model_and_tokenizer) -> None:
    """The visualisation must show the weights the model actually computed."""
    model, tokenizer = model_and_tokenizer
    prompt = "The wizard of Oz"
    view = attention_for_prompt(model, tokenizer, prompt)

    idx = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)
    with torch.no_grad():
        direct = model(idx, need_weights=True).attentions
    expected = torch.stack([a[0] for a in direct]).numpy()

    np.testing.assert_allclose(view.weights, expected, atol=1e-6)


def test_masthead_matches_the_site_it_belongs_to() -> None:
    """The restated masthead must say what the site's masthead says.

    ``/attention/`` is a single self-contained file and cannot import the site's
    component, so its masthead is restated in the template — and a restatement is a
    copy, which drifts. It did: the site's strapline changed and this page went on
    calling the project "A laboratory notebook" for a deploy afterwards, on the one
    page a reader reaches by clicking the site's own nav.

    Pinning it here is cheaper than noticing. The template is deliberately not shared,
    so this asserts the two agree rather than trying to make them one thing.
    """
    sub = re.compile(r'class(?:Name)?="wordmark-sub">([^<]+)<')

    template = sub.search(TEMPLATE.read_text(encoding="utf-8"))
    site = sub.search((ROOT / "web/src/components/Masthead.tsx").read_text(encoding="utf-8"))
    assert template is not None, "the attention template has no wordmark-sub"
    assert site is not None, "the site's Masthead component has no wordmark-sub"

    assert template.group(1) == site.group(1), (
        f"the attention page says {template.group(1)!r} where the site says "
        f"{site.group(1)!r} — edit src/llmfs/viz/template.html to match"
    )
