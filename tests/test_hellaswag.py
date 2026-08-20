"""The HellaSwag eval set is a pinned input, and the pin is enforced, not aspirational.

The set is fetched from a commit-addressed URL and checksum-verified — including the
*cached* copy on every use, because a truncated or substituted cache used to be trusted
forever once any file existed at the path.
"""

from __future__ import annotations

import hashlib

import pytest

from llmfs.eval import hellaswag


def _install_fake_download(monkeypatch, payload: bytes) -> None:
    class FakeResponse:
        def read(self) -> bytes:
            return payload

        def __enter__(self):
            return self

        def __exit__(self, *exc) -> bool:
            return False

    monkeypatch.setattr(hellaswag.urllib.request, "urlopen", lambda url: FakeResponse())


def test_download_rejects_a_body_that_does_not_match_the_pin(tmp_path, monkeypatch) -> None:
    _install_fake_download(monkeypatch, b"not the eval set")
    with pytest.raises(RuntimeError, match="does not match the pinned"):
        hellaswag.download(tmp_path)
    # Nothing may be cached under the real name — that is how a bad copy becomes trusted.
    assert not (tmp_path / "hellaswag_val.jsonl").exists()


def test_cached_copy_is_reverified_on_every_use(tmp_path) -> None:
    (tmp_path / "hellaswag_val.jsonl").write_bytes(b"truncated by a hand-run curl")
    with pytest.raises(RuntimeError, match="delete it and re-run"):
        hellaswag.download(tmp_path)


def test_download_accepts_and_caches_a_matching_body(tmp_path, monkeypatch) -> None:
    payload = b"the pinned eval set"
    monkeypatch.setattr(hellaswag, "HELLASWAG_VAL_SHA256", hashlib.sha256(payload).hexdigest())
    _install_fake_download(monkeypatch, payload)

    path = hellaswag.download(tmp_path)
    assert path.read_bytes() == payload
    # Second call takes the cache path and still passes verification.
    assert hellaswag.download(tmp_path) == path


def test_url_is_pinned_to_a_commit_not_a_branch() -> None:
    assert hellaswag.HELLASWAG_COMMIT in hellaswag.HELLASWAG_VAL_URL
    assert "/master/" not in hellaswag.HELLASWAG_VAL_URL
