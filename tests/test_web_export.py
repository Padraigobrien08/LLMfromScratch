"""The site's generated figures must still be what the generator produces.

``test_documented_results.py`` pins the Markdown to ``results/*.json``. This file pins
the *site* to the same artifacts, across the language boundary, which is where the
guarantee had a hole: ``projectState.ts`` claimed 223 Python tests and 69 browser tests
for weeks after both numbers moved, and nothing failed, because a figure retyped into
TypeScript is outside every check written in Python.

The mechanism is a generated module rather than a rule about care. If the committed
``measured.ts`` is not byte-identical to a fresh export, that is a stale figure on a
page, and it fails here.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from conftest import ARCH_VARIANTS
from llmfs.export.web import (
    ARCHITECTURE_OUT,
    OUT,
    ROOT,
    SHOWCASE_OUT,
    TARGET_LOSS,
    build,
    build_architecture,
    build_showcase,
    collect_python_tests,
    committed_browser_tests,
)

PROJECT_STATE = ROOT / "web" / "src" / "content" / "projectState.ts"

# The skip exists for a checkout without the site; in CI it would be a hole — a repo
# layout change that moved web/ would turn every staleness guarantee in this file into
# a skip that reads as a pass. LLMFS_REQUIRE_WEB=1 (set in every CI job that runs this
# suite, like LLMFS_REQUIRE_VOCAB in conftest) converts that silence into a failure.
if os.environ.get("LLMFS_REQUIRE_WEB") == "1" and not OUT.parent.is_dir():
    raise AssertionError(f"LLMFS_REQUIRE_WEB=1 but {OUT.parent} is missing")

pytestmark = pytest.mark.skipif(not OUT.parent.is_dir(), reason="web/ not present")


@pytest.fixture(scope="module")
def committed() -> str:
    return OUT.read_text()


def test_committed_measured_module_is_fresh(committed: str) -> None:
    """Regenerate and compare, which is the whole guarantee in one assertion.

    The browser count is taken from the committed file rather than measured, because
    enumerating the vitest suite needs a Node toolchain this job does not have. It is
    checked where that toolchain always exists — ``npm run check:counts``, in CI's site
    job — so between the two every field is covered.

    This collects the Python suite in a subprocess, which costs a few seconds. That is
    the price of the count being pytest's own answer rather than a second, wrong
    implementation of its collection rules.
    """
    browser = committed_browser_tests()
    assert browser is not None, "the committed module must carry a browser test count"

    fresh = build(python_tests=collect_python_tests()[0], browser_tests=browser)
    assert committed == fresh, (
        "web/src/content/measured.ts is stale — run `llmfs-export-web` and commit it. "
        "Adding or removing a test changes the count the site prints."
    )


def test_measured_module_is_a_json_literal(committed: str) -> None:
    """The renderer relies on JSON being a subset of TypeScript. If that ever stops
    holding — a NaN, an inf, a non-finite float from some future artifact — the site
    would fail to build with a syntax error rather than a legible one. Catch it here."""
    body = re.search(r"export const MEASURED = (\{.*\}) as const;\n\Z", committed, re.S)
    assert body is not None, "measured.ts is not the shape the generator emits"
    payload = json.loads(body.group(1))
    assert payload["tests"]["python"] > 0
    assert payload["reproduction"]["loss"] < payload["reproduction"]["targetLoss"]


def test_target_loss_matches_where_it_was_pre_registered() -> None:
    """The 3.29 target is the one figure in the export that is not a measurement.

    It is a commitment made before the run, so it lives in no results file, and the
    reason the reproduction means anything is that it was fixed in advance. That makes
    it the single most valuable number to be unable to quietly edit: a target adjusted
    afterwards to match the result would leave every other check green.
    """
    # Anchored to how a *target* is stated — "≤ 3.29", "<= 3.29" or "3.29 target" —
    # rather than a bare substring. docs/reproduction.md also contains "$3.29/hr",
    # the H100 hourly rate, and against a bare `in` every real statement of the
    # target in that file could be edited away while the price kept the test green.
    target = re.escape(str(TARGET_LOSS))
    stated = re.compile(rf"(?:≤|<=)\s*{target}|{target} target")
    for name in ("configs/gpt2-124m.yaml", "docs/reproduction.md", "README.md"):
        text = (ROOT / name).read_text()
        assert stated.search(text), f"{name} no longer states the pre-registered target"


def test_readme_status_table_states_the_live_test_count() -> None:
    """The README is the site's ceiling, so a stale README drags the site down with it.

    The site may not claim more than the README does. It used to hold that line by
    mirroring the table row for row in ``content/status.ts``; the front page now prints
    one strip and reads its figures from ``measured.ts``, so the mirror is gone and the
    ceiling is enforced here instead. Either way the site cannot fix this by mirroring
    harder — the count has to be true at the source.
    """
    readme = (ROOT / "README.md").read_text()
    count = collect_python_tests()[0]
    assert f"{count} tests green" in readme, (
        f"README.md does not say '{count} tests green' — the suite has {count} tests"
    )
    # The repository tree further down restates the count, and it had gone stale twice
    # over while the status row above it stayed current. One check, both places.
    assert f"tests/        {count} tests" in readme, (
        f"the repository tree in README.md does not say '{count} tests'"
    )


def test_site_reports_the_real_number_of_architecture_variants() -> None:
    """``archVariants`` is the one figure the site states that no artifact holds.

    Every property test — causality, the KV cache, GQA equivalence — runs against all of
    them, so the number is a claim about coverage. It stays hand-written because it
    comes from a test fixture rather than a run, and it is pinned here instead.
    """
    match = re.search(r"archVariants:\s*(\d+)", PROJECT_STATE.read_text())
    assert match is not None, "projectState.ts no longer states archVariants"
    assert int(match.group(1)) == len(ARCH_VARIANTS)


def test_the_site_does_not_retype_the_reproduction_loss() -> None:
    """A regression guard on the habit, not on a value.

    Before the generator existed, ``projectState.ts`` held `loss: 3.05` as a literal.
    Re-typing it is what this whole phase exists to stop, and it is an easy thing to do
    again when a page wants one figure and importing feels like ceremony.

    Comments are stripped first, deliberately. A comment may quote a figure to explain
    where it came from or why a page stopped printing it, and prose about a number is
    not a claim of one. Only code can lie to a reader.
    """
    for path in sorted((ROOT / "web" / "src" / "content").glob("*.ts")):
        if path.name == "measured.ts":
            continue
        code = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
        assert "3.0503" not in code and "3.05," not in code, (
            f"{path.name} restates the reproduction loss — import it from measured.ts"
        )


def test_committed_test_showcase_is_fresh() -> None:
    """The `#/tests` page's rows come from collection, so they cannot outlive the tests.

    Renaming a showcased test, deleting it, or editing what it claims to pin all change
    what the plugin emits. Without this the page would keep advertising a guarantee the
    suite no longer provides, which is a worse failure than having no page: it is a
    specific, checkable claim that has quietly stopped being true.
    """
    _, rows = collect_python_tests()
    assert rows, "no tests carry @pytest.mark.showcase — the page would be empty"
    assert SHOWCASE_OUT.read_text() == build_showcase(rows), (
        "web/src/content/testShowcase.ts is stale — run `llmfs-export-web` and commit it"
    )


def test_every_showcased_test_says_what_it_pins_and_why() -> None:
    """A row with no `why` is a directory listing entry, which is what this page is not."""
    _, rows = collect_python_tests()
    for row in rows:
        assert row["pins"].strip(), f"{row['name']} is marked but says nothing about what it pins"
        assert row["why"].strip(), f"{row['name']} is marked but gives no reason to exist"
        assert row["cases"] >= 1


def test_committed_architecture_module_is_fresh() -> None:
    """The architecture page's config values, resolved by the repository's own loader.

    `llama-124m.yaml` never states `n_layer` and `gpt2-124m.yaml` never states
    `n_kv_head`; the first inherits through `_base_` and the second is filled in by
    `ModelConfig.__post_init__`. A page that read the YAML directly would show blanks or
    guesses for exactly the fields a reader would check.
    """
    assert ARCHITECTURE_OUT.read_text() == build_architecture(), (
        "web/src/content/architecture.ts is stale — run `llmfs-export-web` and commit it"
    )


def test_the_architecture_export_resolves_what_the_yaml_leaves_unsaid() -> None:
    """The property that makes going through the loader necessary rather than tidy."""
    llama = (ROOT / "configs" / "llama-124m.yaml").read_text()
    gpt2 = (ROOT / "configs" / "gpt2-124m.yaml").read_text()
    assert "n_layer" not in llama, "the inheritance this test guards no longer applies"
    assert "n_kv_head" not in gpt2, "the defaulting this test guards no longer applies"

    module = ARCHITECTURE_OUT.read_text()
    body = re.search(r"export const ARCHITECTURES = (\{.*\}) as const;\n\Z", module, re.S)
    assert body is not None
    resolved = json.loads(body.group(1))
    assert resolved["llama"]["config"]["nLayer"] == resolved["gpt2"]["config"]["nLayer"]
    assert resolved["gpt2"]["config"]["nKvHead"] == resolved["gpt2"]["config"]["nHead"]
    assert resolved["llama"]["config"]["nKvHead"] == 4


def test_every_test_the_architecture_page_names_actually_exists() -> None:
    """The architecture page's whole argument is *this is pinned*, so its claims are the
    worst possible place to be wrong.

    This is the mechanical half: every test named in `blocks.ts` must exist. It cannot
    check that the named test asserts what the sentence beside it says — only reading
    does that, and one sentence here was already wrong on the first pass
    (`test_gpt2_124m_parameter_count` uses vocab_size 50257, counts *non-embedding*
    parameters, and asserts a range, so it does not pin the exact figures the page
    prints; the real pin is the browser-side fixture test, which is what it now names).
    What this does catch is the failure that arrives later: a test renamed or deleted
    while the page goes on citing it.
    """
    blocks = (ROOT / "web" / "src" / "content" / "blocks.ts").read_text()
    named = re.findall(r'test:\s*"([^"]+)"', blocks)
    assert named, "blocks.ts names no tests at all"

    collected = {node.split("[")[0] for node in _collected_node_ids()}
    for name in named:
        if "::" in name:
            assert f"tests/{name}" in collected, f"blocks.ts cites {name}, which no longer exists"
        else:
            assert (ROOT / name).exists(), f"blocks.ts cites {name}, which is not a file"


def _collected_node_ids() -> list[str]:
    import subprocess
    import sys

    # `-o addopts=` clears the ini's own `-q`; without it this inherits `-qq`, which
    # switches the reporter to per-file counts and lists no node ids at all.
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if "::" in line]


def test_committed_axes_module_is_fresh() -> None:
    """Same guarantee as the other three modules, for the ablation registry."""
    from llmfs.export.web import AXES_OUT, build_axes

    assert AXES_OUT.read_text() == build_axes(), (
        "web/src/content/ablationAxes.ts is stale — run `llmfs-export-web` and commit it."
    )


def test_accumulation_micro_step_constant_matches_the_config() -> None:
    """The one hand-typed factor in a site figure, pinned to what it encodes.

    The comm-accum artifacts record only `grad_accum_steps`, so the exporter multiplies
    by the micro-step's token count — gpt2-124m's micro_batch × block at measurement
    time. If the config's batching ever changes this becomes a wrong tokens-per-step
    column on the scaling plate, and this test is what makes that loud. (New scaling
    artifacts record `tokens_per_step` directly; the committed ones predate the field.)
    """
    from llmfs.config import load_config
    from llmfs.export.web import COMM_SWEEP_TOKENS_PER_MICRO_STEP

    cfg = load_config("gpt2-124m")
    assert cfg.data.micro_batch_size * cfg.data.block_size == COMM_SWEEP_TOKENS_PER_MICRO_STEP
