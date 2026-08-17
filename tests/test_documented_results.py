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
import math
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
    for name in ("fp32 baseline", "int8 g128", "int4 g128", "int4 per-channel"):
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


def test_the_embedding_ceiling_is_the_fraction_the_model_actually_has() -> None:
    """ "33% of this model — 147 MiB of the 471" disagreed with itself and with the model.

    147 ÷ 471 is 31.2%, the model's own arithmetic gives 31.0%, and the total is 475 MiB,
    not 471 — three numbers in one clause, two of them wrong, on the claim that explains
    the whole compression ceiling. The site had 31% right the entire time.
    """
    import yaml

    from llmfs.model.config import ModelConfig
    from llmfs.model.transformer import Transformer
    from llmfs.quant.quantize import model_memory_bytes

    cfg = yaml.safe_load((DOCS / "configs/gpt2-124m.yaml").read_text())
    model = Transformer(ModelConfig(**cfg["model"]))
    total = model_memory_bytes(model)
    embedding = model.tok_emb.weight.numel() * 4

    # The same total the fp32 baseline row was measured at, so doc and artifact agree.
    baseline = next(
        r for r in load("quantization-cuda.json")["results"] if r["name"] == "fp32 baseline"
    )
    assert total / 2**20 == pytest.approx(baseline["memory_mib"])

    must_appear(
        f"{embedding / total * 100:.0f}% of this model", "docs/efficiency.md", "docs/roadmap.md"
    )
    must_appear(f"{embedding / 2**20:.0f} MiB of", "docs/efficiency.md")
    must_appear(f"{total / 2**20:,.0f} MiB", "docs/efficiency.md")


def test_quantization_throughput_loss_ranges_match_both_artifacts() -> None:
    """The MPS-versus-CUDA contrast, recomputed from the two files it compares.

    The document said quantization "lost 74–85% of throughput on MPS"; the artifact says
    81–86%. A range is as much a claim as a number, and this one was wrong at both ends
    while sitting in the sentence that draws the section's conclusion.
    """
    ranges = (("quantization.json", "81–86%"), ("quantization-cuda.json", "25–53%"))
    for artifact, doc_range in ranges:
        rows = load(artifact)["results"]
        base = next(r for r in rows if r["name"] == "fp32 baseline")["decode_tok_s"]
        losses = [
            (1 - r["decode_tok_s"] / base) * 100 for r in rows if r["name"] != "fp32 baseline"
        ]
        lo, hi = math.floor(min(losses)), math.ceil(max(losses))
        assert f"{lo}–{hi}%" == doc_range, f"{artifact}: measured {lo}–{hi}%, doc says {doc_range}"
        must_appear(doc_range, "docs/efficiency.md")


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


# ------------------------------------------------------------------ docs index


def test_docs_index_headlines_match_their_artifacts() -> None:
    """``docs/README.md`` restates one headline per document so a reader can choose which to
    open. Restating a figure is exactly how drift starts, so the index is pinned like any
    other document rather than trusted because it is only an index.

    The checks above cannot cover it: they are parametrized over every world size and every
    quantization scheme, and the index quotes one row from each. So it gets its own.
    """
    repro, hella = load("reproduction.json"), load("hellaswag.json")
    must_appear(f"{repro['loss']:.4f}", "docs/README.md")
    must_appear(f"{hella['acc_norm']:.4f}", "docs/README.md")
    must_appear(f"{hella['gpt2_124m_reference_acc_norm']}", "docs/README.md")

    p8 = point(load("scaling-5090x8.json"), 8)
    must_appear(f"{p8['efficiency'] * 100:.1f}%", "docs/README.md")
    must_appear("no NVLink", "docs/README.md")


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


# -------------------------------------------------------------------- ablations


def test_ablation_scale_is_stated_as_it_was_run() -> None:
    """The README described the sweep as "~1B tokens"; every one of its 39 runs was 524M.

    That is the size of the whole study, stated in the document most people read first,
    and it matched no unit in any artifact — not the per-run budget, not the total, not
    the config. It is pinned to the artifact now, in every document that quotes it.
    """
    arms = load("ablations.json")["arms"]
    tokens = {a["tokens"] for a in arms}
    assert len(tokens) == 1, f"arms disagree on token budget: {sorted(tokens)}"
    must_appear(
        f"{tokens.pop() / 1e6:.0f}M tokens", "README.md", "docs/ablations.md", "docs/gpu-runbook.md"
    )
    must_appear(f"{len(arms)} runs", "README.md", "docs/ablations.md")
    assert len({a["seed"] for a in arms}) == 3


# ------------------------------------------------------------------ utilisation


def flops_per_token() -> float:
    """The model's own FLOP/token, which every ``achieved TFLOP/s`` figure is derived from."""
    import yaml

    from llmfs.model.config import ModelConfig
    from llmfs.model.transformer import Transformer

    cfg = yaml.safe_load((DOCS / "configs/gpt2-124m.yaml").read_text())
    return Transformer(ModelConfig(**cfg["model"])).flops_per_token()


def test_scaling_achieved_tflops_column_matches_artifact() -> None:
    """`achieved` is throughput × the model's FLOP/token, and nothing else."""
    fpt = flops_per_token()
    text = (DOCS / "docs/scaling.md").read_text()
    assert f"{fpt / 1e9:.3f} GFLOP/token" in text
    for p in load("scaling-5090x8.json")["points"]:
        achieved = p["tokens_per_sec"] * fpt / 1e12
        must_appear(f"{achieved:,.0f} TFLOP/s", "docs/scaling.md")


def test_of_measured_ceiling_column_matches_artifact() -> None:
    """The per-GPU utilisation table, against the 5090's own measured matmul ceiling.

    The ceiling is quoted from the doc rather than an artifact because the probe that
    produced it was not captured to ``results/`` — the doc says so. What is checked here is
    that every percentage in the table follows from that one number and the measured
    throughputs, so the column cannot drift away from the run it describes.
    """
    text = (DOCS / "docs/scaling.md").read_text()
    ceiling = 234.7
    assert f"**{ceiling} TFLOP/s measured.**" in text, "the stated ceiling moved"
    fpt = flops_per_token()
    for p in load("scaling-5090x8.json")["points"]:
        achieved = p["tokens_per_sec_per_gpu"] * fpt / 1e12
        assert f"{achieved / ceiling * 100:.1f}%" in text, f"{p['world_size']} GPUs"


def test_4090_cross_check_is_the_same_arithmetic_on_the_4090_artifact() -> None:
    """ "The 4090 measured the same way" must actually be the same way.

    It was not: the figure in the doc traced to no artifact, and recomputing it moved the
    number. Same model, same formula, the 4090's compiled training throughput (the scaling
    config compiles too) over the 4090's own measured ceiling — from one file.
    """
    bench = load("benchmarks-cuda.json")
    compiled = next(
        r for r in bench["results"] if r["suite"] == "training" and r["variant"] == "compile"
    )
    achieved = compiled["tokens_per_sec"] * flops_per_token() / 1e12
    ceiling = bench["provenance"]["measured_bf16_tflops"]
    must_appear(f"{achieved / ceiling * 100:.1f}%", "docs/scaling.md")
    must_appear(f"{ceiling} TFLOP/s", "docs/scaling.md", "README.md", "docs/efficiency.md")


BATCHING_ROWS = {
    "naive (no cache), batch 1": "naive (no cache) b1",
    "kv-cache, batch 1": "kv-cache b1",
    "kv-cache, batch 4": "kv-cache b4",
    "kv-cache, batch 16": "kv-cache b16",
}


@pytest.mark.parametrize("label,variant", sorted(BATCHING_ROWS.items()))
def test_efficiency_batching_table_cells(label: str, variant: str) -> None:
    """The batching table in docs/efficiency.md, cell by cell against benchmarks-cuda.json.

    This table is why the check exists. Its tokens/sec column came from the artifact measured
    after the attention-mask fix and its time-to-first-token column from the one measured
    before it — one table quoting two commits, undetectable by a substring check because both
    files are in ``results/`` and each number was true of *a* run. Every column is pinned now,
    and to the same file.
    """
    bench = {r["variant"]: r for r in load("benchmarks-cuda.json")["results"]}[variant]
    rows = markdown_rows((DOCS / "docs/efficiency.md").read_text(), label)
    assert len(rows) == 1, f"expected exactly one batching row for {label!r}"
    row = rows[0]

    assert row[1] == f"{bench['tokens_per_sec']:,.0f}", f"tokens/sec cell for {label}"
    assert row[2] == f"{bench['extra']['time_to_first_token_ms']:.1f} ms", f"ttft cell for {label}"
    mib = bench["extra"]["kv_cache_mib"]
    expected_cache = "—" if mib == 0 else f"{mib:g} MiB"
    assert row[3] == expected_cache, f"kv cache cell for {label}"


@pytest.mark.parametrize("accum", [1, 2, 4, 8])
def test_readme_accumulation_table_cells(accum: int) -> None:
    p8 = point(load(f"comm-accum{accum}.json"), 8)
    rows = markdown_rows((DOCS / "README.md").read_text(), str(accum))
    row = next((r for r in rows if r[1] == f"{accum * 131072:,}"), None)
    assert row is not None, f"no README accumulation row for accum {accum}"
    assert row[2] == f"{p8['tokens_per_sec']:,.0f}", f"tokens/sec cell for accum {accum}"
    assert row[3].startswith(f"{p8['efficiency'] * 100:.1f}%"), f"efficiency cell for accum {accum}"
