"""The committed web fixtures must be what their generators produce.

`measured.ts` earned this guarantee first (`test_web_export.py`); these two fixtures
had none — `modelsize.test.ts` asserts the TS calculator against `model-sizes.json`,
so a change to `src/llmfs/model/` that nobody re-dumped left the web suite green
against a stale fixture, which is the exact failure `llmfs-export-web --check` was
built to close.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from conftest import gpt2_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_sizes_fixture_is_fresh() -> None:
    module = _load("dump_model_sizes")
    assert module.OUT.read_text() == module.build(), (
        "web/src/data/model-sizes.json is stale — run `python scripts/dump_model_sizes.py` "
        "and commit it. The TS size calculator is asserted against this fixture."
    )


def test_bigram_fixture_is_fresh() -> None:
    gpt2_tokenizer()  # the dump tokenises the corpus; skip/fail per LLMFS_REQUIRE_VOCAB
    module = _load("dump_bigram")
    assert module.OUT.read_text() == module.build(), (
        "web/src/data/bigram.json is stale — run `python scripts/dump_bigram.py` and commit it."
    )
