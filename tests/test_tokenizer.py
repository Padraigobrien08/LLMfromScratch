"""Tokenizer backends.

Kept hermetic: the ``file:`` backend uses the tokenizer already committed to this
repo, and the GPT-2 test is skipped when its vocabulary cannot be fetched, so CI
never depends on the network being up.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llmfs.data.tokenizer import load_tokenizer

LOCAL_TOKENIZER = Path(__file__).resolve().parents[1] / "data" / "bpe_tokenizer.json"


@pytest.fixture
def local_tokenizer():
    if not LOCAL_TOKENIZER.exists():
        pytest.skip(f"{LOCAL_TOKENIZER} not present")
    return load_tokenizer(f"file:{LOCAL_TOKENIZER}")


def test_round_trip(local_tokenizer) -> None:
    text = "The Wizard of Oz stood before Dorothy."
    assert local_tokenizer.decode(local_tokenizer.encode(text)) == text


def test_vocab_size_is_reported(local_tokenizer) -> None:
    assert local_tokenizer.vocab_size > 0


def test_add_eot_prefixes_a_document_boundary(local_tokenizer) -> None:
    plain = local_tokenizer.encode("hello")
    with_eot = local_tokenizer.encode("hello", add_eot=True)
    assert with_eot == [local_tokenizer.eot_token, *plain]


def test_unknown_spec_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown tokenizer spec"):
        load_tokenizer("sentencepiece")


def test_missing_file_is_reported() -> None:
    with pytest.raises(FileNotFoundError):
        load_tokenizer("file:/nonexistent/tokenizer.json")


def test_gpt2_vocabulary_matches_the_published_size() -> None:
    """50257 is the number the reproduction target's loss was measured against."""
    try:
        tokenizer = load_tokenizer("gpt2")
    except Exception as exc:  # noqa: BLE001 - offline CI must not fail here
        pytest.skip(f"gpt2 vocabulary unavailable: {exc}")

    assert tokenizer.vocab_size == 50257
    assert tokenizer.eot_token == 50256
    assert tokenizer.decode(tokenizer.encode("hello world")) == "hello world"
