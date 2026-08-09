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


def test_round_trip_recovers_the_tokens(local_tokenizer) -> None:
    """Decoding recovers the content, though not byte-for-byte.

    This tokenizer's decoder re-joins tokens with spaces, so punctuation comes back
    detached ("Dorothy ." rather than "Dorothy."). That is a property of the
    tokenizer, not of the pipeline — asserted here so the difference from GPT-2's
    lossless byte-level BPE is explicit rather than a surprise later.
    """
    text = "The Wizard of Oz stood before Dorothy."
    decoded = local_tokenizer.decode(local_tokenizer.encode(text))
    assert decoded.split() == text.replace(".", " .").split()


def test_encoding_is_stable_under_re_encoding(local_tokenizer) -> None:
    """Once through the round trip, further passes are fixed points."""
    once = local_tokenizer.decode(local_tokenizer.encode("The Wizard of Oz."))
    assert local_tokenizer.decode(local_tokenizer.encode(once)) == once


def test_vocab_size_is_reported(local_tokenizer) -> None:
    assert local_tokenizer.vocab_size > 0


def test_large_vocab_is_rejected_before_tokenisation_starts(local_tokenizer, tmp_path) -> None:
    """Shards are uint16, so a vocabulary above 65,536 cannot be stored.

    This repo's own tokenizer has a 150k vocabulary and therefore cannot prepare
    data. The check has to be up front: a per-token guard would only trip when a
    document happened to contain a high id, which over a 10B-token corpus could be
    hours in.
    """
    from llmfs.data.prepare import check_vocab_fits_shards, prepare_text_file

    assert local_tokenizer.vocab_size > 2**16

    with pytest.raises(ValueError, match="does not fit the uint16 shard format"):
        check_vocab_fits_shards(f"file:{LOCAL_TOKENIZER}")

    source = tmp_path / "corpus.txt"
    source.write_text("The Wizard of Oz stood before Dorothy.")
    with pytest.raises(ValueError, match="does not fit the uint16 shard format"):
        prepare_text_file(source, tmp_path / "out", tokenizer_spec=f"file:{LOCAL_TOKENIZER}")
    assert not (tmp_path / "out").exists(), "failed before writing anything"


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
