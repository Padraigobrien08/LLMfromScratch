"""Every headline number in the docs must trace to an artifact in ``results/``.

The README opens with "No results are reported below that have not been measured." That is
the repository's central claim, and until this file existed it was enforced by my care alone
— which is exactly the kind of guarantee that decays. Numbers get copied between documents,
a re-run shifts a figure by 0.2%, a table is edited and its prose is not.

So each check here reads a results file, formats the figure the way the documents do, and
asserts the string appears where it is cited. A drifted number fails CI.

What this does *not* do is check prose reasoning, or that the right figure was chosen. It
checks that the numbers presented are the numbers measured. That is a narrow guarantee, and
it is the one that was silently at risk.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"
DOCS = Path(__file__).resolve().parents[1]


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


def must_appear(needle: str, *files: str, why: str = "") -> None:
    """Assert ``needle`` appears in *every* named file.

    The first version of this helper concatenated the documents and checked the joined
    text. That was far too weak, and mutation testing caught it: a figure only had to
    survive in one document, so changing 1,414,340 in the README while docs/scaling.md
    still held it passed cleanly. Three of four deliberate corruptions went undetected.

    Each file is therefore checked on its own, and every claim names the documents that
    actually present it.
    """
    for name in files:
        text = (DOCS / name).read_text()
        assert needle in text, f"{name} does not contain {needle!r}" + (f" ({why})" if why else "")


def point(report: dict, world_size: int) -> dict:
    return next(p for p in report["points"] if p["world_size"] == world_size)


# --------------------------------------------------------------------------- scaling


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_scaling_throughput_matches_artifact(world_size: int) -> None:
    p = point(load("scaling-5090x8.json"), world_size)
    # Total throughput is tabled in both; per-GPU only in the full report, except world
    # size 1 where the two coincide.
    must_appear(f"{p['tokens_per_sec']:,.0f}", "README.md", "docs/scaling.md")
    must_appear(f"{p['tokens_per_sec_per_gpu']:,.0f}", "docs/scaling.md")
    must_appear(f"{p['step_time_ms']:,.1f}", "docs/scaling.md")


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_scaling_efficiency_matches_artifact(world_size: int) -> None:
    p = point(load("scaling-5090x8.json"), world_size)
    must_appear(f"{p['efficiency'] * 100:.1f}%", "README.md", "docs/scaling.md")


def test_scaling_loss_equivalence_quoted_verbatim() -> None:
    """The loss-at-step-1 values are quoted to full precision because the point being made
    is that they are *identical*. Rounding them would destroy the evidence."""
    for p in load("scaling-5090x8.json")["points"]:
        must_appear(
            repr(p["loss_first"]),
            "README.md",
            "docs/scaling.md",
            why="quoted to full precision because the claim is that they are identical",
        )


def test_scaling_reports_no_nvlink_because_the_artifact_says_so() -> None:
    topology = load("scaling-5090x8.json")["topology"]
    assert topology["has_nvlink"] is False
    assert topology["interconnect"] == "PCIe"
    must_appear("no NVLink", "README.md", "docs/scaling.md")


# ------------------------------------------------------------- accumulation sweep


@pytest.mark.parametrize("accum", [1, 2, 4, 8])
def test_accumulation_sweep_matches_artifacts(accum: int) -> None:
    report = load(f"comm-accum{accum}.json")
    p8, p1 = point(report, 8), point(report, 1)
    assert p8["grad_accum_steps"] == accum, "file name must match what was actually run"
    must_appear(f"{p8['tokens_per_sec']:,.0f}", "README.md", "docs/scaling.md")
    must_appear(f"{p8['efficiency'] * 100:.1f}%", "README.md", "docs/scaling.md")
    must_appear(f"{p1['grad_accum_steps'] * 16 * 1024:,}", "README.md", "docs/scaling.md")


def test_accumulation_control_reproduces_the_scaling_run() -> None:
    """The accum=4 point and the scaling run's 8-GPU point are the same configuration,
    measured a day apart. The doc claims they agree to 0.24% and 0.15 points; if a re-run
    ever changes that, the claim must fail rather than quietly become false."""
    a = point(load("comm-accum4.json"), 8)
    b = point(load("scaling-5090x8.json"), 8)
    drift = abs(a["tokens_per_sec"] / b["tokens_per_sec"] - 1) * 100
    points = abs(a["efficiency"] - b["efficiency"]) * 100
    assert drift < 0.5, f"control drifted {drift:.2f}% — the doc says 0.24%"
    assert points < 0.5, f"efficiency drifted {points:.2f} pts — the doc says 0.15"
    must_appear("0.24%", "docs/scaling.md", "README.md")


def test_amortisation_fit_is_the_one_quoted() -> None:
    """The a and b of `loss = a + b/accum` are quoted in both documents, and they must come
    from fitting the accum 8 and 4 points only — the whole argument is that the other two
    were predicted, not fitted."""
    from llmfs.bench.scaling import fit_amortisation

    losses = {
        accum: (1 - point(load(f"comm-accum{accum}.json"), 8)["efficiency"]) * 100
        for accum in (1, 2, 4, 8)
    }
    fit = fit_amortisation(losses, using=(8, 4))
    assert fit is not None
    a, b = fit
    must_appear(f"{a:.3f}", "README.md", "docs/scaling.md")
    must_appear(f"{b:.3f}", "README.md", "docs/scaling.md")

    # And the out-of-sample predictions, which only the full report tables.
    for accum in (2, 1):
        must_appear(f"{100 - (a + b / accum):.2f}%", "docs/scaling.md")


# ------------------------------------------------------------------ quantization


def test_quantization_figures_match_artifact() -> None:
    schemes = {r["name"]: r for r in load("quantization-cuda.json")["results"]}
    base = schemes["fp32 baseline"]["perplexity"]
    for name in ("fp32 baseline", "int8 g128", "int4 g128", "int4 per-tensor"):
        r = schemes[name]
        must_appear(f"{r['perplexity']:.3f}", "README.md", "docs/efficiency.md", why=name)
        must_appear(f"{r['memory_mib']:,.0f} MiB", "README.md", "docs/efficiency.md", why=name)
        if name != "fp32 baseline":
            must_appear(
                f"{r['perplexity'] - base:+.3f}",
                "README.md",
                "docs/efficiency.md",
                why=f"{name} delta perplexity",
            )


# ------------------------------------------------------- speculative decoding


def test_speculative_is_lossless_and_says_how_many_runs() -> None:
    results = load("speculative-cuda.json")["results"]
    lossless = [r for r in results if r["output_matches_greedy"] is True]
    diverged = [r for r in results if r["output_matches_greedy"] is False]
    assert not diverged, (
        "a diverged run would make speculation an approximation, not an optimisation"
    )
    must_appear(f"All {len(lossless)} benchmark runs", "README.md", "docs/efficiency.md")


def test_speculative_best_row_matches_artifact() -> None:
    results = [r for r in load("speculative-cuda.json")["results"] if r["speedup"] != 1.0]
    best = max(results, key=lambda r: r["speedup"])
    must_appear(f"{best['speedup']:.2f}×", "README.md", "docs/efficiency.md")
    must_appear(f"{best['acceptance_rate']:.1%}", "README.md", "docs/efficiency.md")
    must_appear(f"{best['tokens_per_target_forward']:.2f}", "README.md", "docs/efficiency.md")


# ------------------------------------------------------------------ reproduction


def test_reproduction_headline_matches_its_artifacts() -> None:
    """The trust anchor of the whole repository: it appears in the README's first sentence."""
    repro, hella = load("reproduction.json"), load("hellaswag.json")

    assert repro["split"] == "val", "the headline loss must be the held-out split"
    must_appear(f"{repro['loss']:.4f}", "README.md", "docs/reproduction.md")
    must_appear(f"{repro['perplexity']:.2f}", "README.md", "docs/reproduction.md")
    # The eval must have covered the whole 100M-token split, not a sample of it — the
    # README says "full 100M-token split", and a truncated eval would flatter the number.
    assert repro["tokens_evaluated"] > 99_000_000

    must_appear(f"{hella['acc_norm']:.4f}", "README.md", "docs/reproduction.md")
    # And it must beat both chance and the published GPT-2 124M figure, which is what makes
    # the validation loss trustworthy rather than merely self-consistent.
    assert hella["acc_norm"] > hella["chance"]
    assert hella["acc_norm"] > hella["gpt2_124m_reference_acc_norm"]
    must_appear(f"{hella['gpt2_124m_reference_acc_norm']}", "README.md", "docs/reproduction.md")
    assert hella["n_evaluated"] == 10042, "the full HellaSwag validation set"


# ------------------------------------------------------- tables, cell by cell
#
# Substring checks above prove a figure appears *somewhere* in a document, which catches the
# realistic failure — an artifact is re-measured and the docs go stale, so the new number
# appears nowhere. It cannot catch one stale cell among several correct copies of the same
# number. Mutation testing showed exactly that gap. So the two headline tables are parsed and
# compared cell by cell, which closes it for the numbers that matter most.


def markdown_rows(text: str, first_cell: str) -> list[list[str]]:
    """Every table row whose first cell equals ``first_cell``, as stripped cells."""
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip().replace("**", "") for c in line.strip("|").split("|")]
        if cells and cells[0] == first_cell:
            rows.append(cells)
    return rows


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_readme_scaling_table_cells(world_size: int) -> None:
    p = point(load("scaling-5090x8.json"), world_size)
    rows = markdown_rows((DOCS / "README.md").read_text(), str(world_size))
    row = next((r for r in rows if r[1] == str(p["grad_accum_steps"])), None)
    assert row is not None, f"no README scaling row for {world_size} GPUs"
    assert row[2] == f"{p['tokens_per_sec']:,.0f}", f"tokens/sec cell for {world_size} GPUs"
    expected_eff = "—" if world_size == 1 else f"{p['efficiency'] * 100:.1f}%"
    assert row[3] == expected_eff, f"efficiency cell for {world_size} GPUs"


@pytest.mark.parametrize("accum", [1, 2, 4, 8])
def test_readme_accumulation_table_cells(accum: int) -> None:
    p8 = point(load(f"comm-accum{accum}.json"), 8)
    rows = markdown_rows((DOCS / "README.md").read_text(), str(accum))
    row = next((r for r in rows if r[1] == f"{accum * 131072:,}"), None)
    assert row is not None, f"no README accumulation row for accum {accum}"
    assert row[2] == f"{p8['tokens_per_sec']:,.0f}", f"tokens/sec cell for accum {accum}"
    assert row[3].startswith(f"{p8['efficiency'] * 100:.1f}%"), f"efficiency cell for accum {accum}"
