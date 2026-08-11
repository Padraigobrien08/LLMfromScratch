"""Export measured results into the site, so it cannot claim more than the repo.

``tests/test_documented_results.py`` solved one half of this: every headline figure in
the Markdown is read from ``results/*.json`` and asserted to appear where it is cited.
The site was outside that guarantee, and drifted exactly as you would expect —
``projectState.ts`` said 223 Python tests and 69 browser tests long after both had
moved, because a number typed into a second language is a number nobody re-derives.

So the figures cross the boundary as generated code:

    results/*.json  ->  llmfs-export-web  ->  web/src/content/measured.ts

``tests/test_web_export.py`` asserts the committed module is what this generator
produces, which turns a stale export into a CI failure rather than a confident wrong
number on a page. The site imports ``MEASURED`` instead of restating it.

This is a repository tool, not library code: it reads the working tree (``results/``,
the test suites) rather than the installed package, and it writes into ``web/``.

    llmfs-export-web            # regenerate
    llmfs-export-web --check    # exit 1 if the committed file is stale

Adding or deleting a test changes the counts, so the generator has to be re-run and the
result committed. That is the intended friction: it is the mechanism by which "328 tests
green" stays true.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "results"
OUT = ROOT / "web" / "src" / "content" / "measured.ts"

# Pre-registered in the header comment of `configs/gpt2-124m.yaml` and in
# `docs/reproduction.md`, before the run. It is a *commitment*, not a measurement, so it
# is not in any results file — it is written here and pinned to both of its declaration
# sites by `test_target_loss_matches_where_it_was_pre_registered`.
TARGET_LOSS = 3.29

HEADER = """\
/**
 * Measured results, generated from `results/*.json`. Do not hand-edit.
 *
 *     llmfs-export-web
 *
 * Every figure here was produced by a run whose artifact is committed in `results/`,
 * and `tests/test_web_export.py` asserts this file is still what the generator emits.
 * A page that imports from here cannot quote a number the repository does not hold;
 * a page that retypes one can, which is why nothing on the site should retype one.
 */
"""


def load(name: str) -> dict[str, Any]:
    return json.loads((RESULTS / name).read_text())


def point(report: dict[str, Any], world_size: int) -> dict[str, Any]:
    return next(p for p in report["points"] if p["world_size"] == world_size)


# --------------------------------------------------------------------------- counting


def count_python_tests(root: Path = ROOT) -> int:
    """Collect the suite and count it, rather than parsing the files ourselves.

    Parametrisation means the number of test *functions* is not the number of test
    *cases*, and a hand-rolled counter would be a second, wrong implementation of
    pytest's collection rules. Collection is cheap and it is the definition.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "--collect-only", "-q", "-p", "no:cacheprovider"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    # The quiet reporter ends with per-file counts, one `path: n` line each.
    counts = re.findall(r"^tests/\S+\.py: (\d+)$", proc.stdout, flags=re.MULTILINE)
    if not counts:
        raise RuntimeError(f"could not parse pytest collection output:\n{proc.stdout[-2000:]}")
    return sum(int(n) for n in counts)


def count_browser_tests(root: Path = ROOT) -> int | None:
    """Ask vitest to enumerate the site's suite, or return ``None`` if it cannot.

    ``None`` is the honest answer on a machine with no Node toolchain — the previous
    value is then carried through rather than silently zeroed. The number is checked
    where the toolchain always exists: `npm run check:counts`, in CI's site job.
    """
    web = root / "web"
    if not (web / "node_modules").is_dir() or shutil.which("npx") is None:
        return None
    proc = subprocess.run(
        ["npx", "vitest", "list", "--json"], cwd=web, capture_output=True, text=True
    )
    if proc.returncode != 0:
        return None
    return len(json.loads(proc.stdout))


def committed_browser_tests(out: Path = OUT) -> int | None:
    """The browser count already in the generated file, for when vitest is unavailable."""
    if not out.exists():
        return None
    match = re.search(r'"browser":\s*(\d+)', out.read_text())
    return int(match.group(1)) if match else None


# ---------------------------------------------------------------------------- payload


def build_payload(*, python_tests: int, browser_tests: int) -> dict[str, Any]:
    """Everything the site is allowed to state as measured, in one object."""
    repro, hella = load("reproduction.json"), load("hellaswag.json")
    scaling = load("scaling-5090x8.json")
    quant, spec = load("quantization-cuda.json"), load("speculative-cuda.json")
    ablations = load("ablations.json")

    return {
        "tests": {"python": python_tests, "browser": browser_tests},
        "reproduction": _reproduction(repro, hella),
        "ablations": _ablations(ablations),
        "scaling": _scaling(scaling),
        "accumulation": _accumulation(),
        "throughput": _throughput(),
        "cache": _cache(),
        "quantization": _quantization(quant),
        "speculative": _speculative(spec),
    }


def _reproduction(repro: dict[str, Any], hella: dict[str, Any]) -> dict[str, Any]:
    curve = load("reproduction-curve.json")
    mfus = [p["mfu"] for p in curve["train"] if p["mfu"] is not None]
    # The first logged step includes compilation and allocator warmup and sits at half
    # the steady-state figure. It belongs in the mean — it is real time the run spent —
    # but quoting it as the *range* would say utilisation swung by twenty-five points
    # when what happened is that it started once and then held.
    settled = mfus[1:]

    return {
        "split": repro["split"],
        "step": repro["step"],
        "loss": repro["loss"],
        "targetLoss": TARGET_LOSS,
        "perplexity": repro["perplexity"],
        "tokensEvaluated": repro["tokens_evaluated"],
        "tokensTrained": curve["tokens"],
        # The mean, because that is the statistic docs/reproduction.md reports and the
        # claim it supports is that MFU was *flat* — a median would hide a run that
        # started well and degraded, which is the failure this number rules out.
        "mfuMean": statistics.fmean(mfus),
        "mfuMin": min(settled),
        "mfuMax": max(settled),
        "mfuWarmup": mfus[0],
        # Where the pre-registered target was first met. The plate's scrubber stops here.
        "crossing": curve["crossing"],
        "hellaswag": {
            "accNorm": hella["acc_norm"],
            "acc": hella["acc"],
            "chance": hella["chance"],
            # The published GPT-2 124M figure. Beating it is what makes the validation
            # loss trustworthy rather than merely self-consistent, so it travels with it.
            "reference": hella["gpt2_124m_reference_acc_norm"],
            "nEvaluated": hella["n_evaluated"],
        },
        "gpu": hella["provenance"]["gpu"],
    }


def _ablations(payload: dict[str, Any]) -> dict[str, Any]:
    from llmfs.ablation.report import baseline_noise

    arms = payload["arms"]
    mean, spread, seeds_counted = baseline_noise(arms)
    return {
        # `meta.arms` includes `_base.yaml`; the study is twelve arms *against* a baseline.
        "arms": len(payload["meta"]["arms"]) - 1,
        "seeds": payload["meta"]["seeds"],
        "runs": len(arms),
        "gpuHours": sum(a["wall_clock_s"] for a in arms if a["wall_clock_s"]) / 3600,
        "baselineLoss": mean,
        # The full range across baseline seeds, not a standard deviation — the same blunt
        # statement `ablation/report.py` makes, so the site cannot report a tighter noise
        # floor than the repository's own report would.
        "noiseFloor": spread,
        "baselineSeeds": seeds_counted,
    }


def _scaling(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": report["label"],
        "config": report["config"],
        "steps": report["steps"],
        "warmup": report["warmup"],
        "hasNvlink": report["topology"]["has_nvlink"],
        "interconnect": report["topology"]["interconnect"],
        "points": [
            {
                "worldSize": p["world_size"],
                "gradAccum": p["grad_accum_steps"],
                "tokensPerSec": p["tokens_per_sec"],
                "tokensPerSecPerGpu": p["tokens_per_sec_per_gpu"],
                "stepTimeMs": p["step_time_ms"],
                "efficiency": p["efficiency"],
                "lossFirst": p["loss_first"],
                "maxLossDeltaVs1Gpu": p["max_loss_delta_vs_1gpu"],
            }
            for p in sorted(report["points"], key=lambda p: p["world_size"])
        ],
    }


def _accumulation() -> dict[str, Any]:
    """The sweep, plus the fit — marking which points the fit was allowed to see.

    ``fittedFrom`` is not decoration. The argument is that `a + b/accum` was fitted to
    accum 8 and 4 and then landed on 2 and 1, which were measured afterwards. A figure
    that draws all four points the same way turns a prediction into a curve through
    data, so the distinction has to survive into the site.
    """
    from llmfs.bench.scaling import fit_amortisation

    reports = {accum: load(f"comm-accum{accum}.json") for accum in (1, 2, 4, 8)}
    losses = {accum: (1 - point(r, 8)["efficiency"]) * 100 for accum, r in reports.items()}
    fitted_from = (8, 4)
    fit = fit_amortisation(losses, using=fitted_from)
    if fit is None:  # pragma: no cover - both points are committed artifacts
        raise RuntimeError("the amortisation fit needs the accum 8 and 4 points")
    a, b = fit

    return {
        "fit": {"a": a, "b": b, "fittedFrom": list(fitted_from)},
        "points": [
            {
                "accum": accum,
                "tokensPerStep": point(reports[accum], 1)["grad_accum_steps"] * 16 * 1024,
                "tokensPerSec": point(reports[accum], 8)["tokens_per_sec"],
                "efficiency": point(reports[accum], 8)["efficiency"],
                "predicted": accum not in fitted_from,
                "predictedEfficiency": (100 - (a + b / accum)) / 100,
            }
            for accum in (8, 4, 2, 1)
        ],
    }


def _throughput() -> dict[str, Any]:
    """Training and batching throughput on both cards.

    Two GPUs rather than one because the contrast is the finding: the same code and the
    same compile speedup read as 41.5% MFU on an H100 and 77.9% on a 4090, and the
    micro-batch that was the *fastest* H100 configuration does not fit in 24 GiB at all.
    A single-card table would have reported the model as inefficient rather than small.
    """
    cards = {"h100": load("benchmarks.json"), "l4090": load("benchmarks-cuda.json")}
    return {
        name: {
            "gpu": payload["provenance"]["gpu"],
            "training": [
                {
                    "variant": r["variant"],
                    "tokensPerSec": r["tokens_per_sec"],
                    "peakMemoryGib": r["peak_memory_gib"],
                    "mfu": r["mfu"],
                    "settings": r["settings"],
                }
                for r in payload["results"]
                if r["suite"] == "training"
            ],
            "inference": [
                {
                    "variant": r["variant"],
                    "tokensPerSec": r["tokens_per_sec"],
                    "batchSize": r["settings"]["batch_size"],
                    "useCache": r["settings"]["use_cache"],
                }
                for r in payload["results"]
                if r["suite"] == "inference"
            ],
        }
        for name, payload in cards.items()
    }


def _cache() -> dict[str, Any]:
    """The cache-versus-recompute sweep, before and after the mask fix.

    Two artifacts, and the second one is the reason this plate exists. `benchmarks-cuda.json`
    is the sweep as it stands; `benchmarks-cuda-before-mask-fix.json` is the same sweep run
    against the code *before* the decode path stopped building a causal mask it did not
    need — restored byte-for-byte from the commit that first published it, so the toggle on
    the page flips between two measurements rather than between a measurement and prose.

    Each file records the commit it was measured at, which is what distinguishes them:
    `6c13dcb1` built the mask, `42ed0a66` did not.
    """
    after, before = load("benchmarks-cuda.json"), load("benchmarks-cuda-before-mask-fix.json")

    def sweep(payload: dict[str, Any]) -> dict[tuple[int, bool], dict[str, Any]]:
        return {
            (r["settings"]["total_len"], r["settings"]["use_cache"]): r
            for r in payload["results"]
            if r["suite"] == "cache-scaling"
        }

    a, b = sweep(after), sweep(before)
    lengths = sorted({length for length, _ in a})

    points = []
    for length in lengths:
        naive, cached, was = a[(length, False)], a[(length, True)], b[(length, True)]
        points.append(
            {
                "totalLen": length,
                "genLen": cached["settings"]["gen_len"],
                "naive": naive["tokens_per_sec"],
                "naiveBefore": b[(length, False)]["tokens_per_sec"],
                "cached": cached["tokens_per_sec"],
                "cachedBefore": was["tokens_per_sec"],
                "advantage": cached["tokens_per_sec"] / naive["tokens_per_sec"],
                "advantageBefore": was["tokens_per_sec"] / b[(length, False)]["tokens_per_sec"],
                "gainFromFix": cached["tokens_per_sec"] / was["tokens_per_sec"],
            }
        )

    return {
        "gpu": after["provenance"]["gpu"],
        "commitBefore": before["provenance"]["git_commit"],
        "commitAfter": after["provenance"]["git_commit"],
        "points": points,
    }


def _quantization(payload: dict[str, Any]) -> dict[str, Any]:
    schemes = {r["name"]: r for r in payload["results"]}
    base = schemes["fp32 baseline"]
    return {
        "device": payload["meta"]["device"],
        "schemes": [
            {
                "name": r["name"],
                "bits": r["bits"],
                "groupSize": r["group_size"],
                "memoryMib": r["memory_mib"],
                "compression": r["compression"],
                "perplexity": r["perplexity"],
                "deltaPerplexity": r["perplexity"] - base["perplexity"],
                "decodeTokS": r["decode_tok_s"],
            }
            for r in payload["results"]
        ],
    }


def _speculative(payload: dict[str, Any]) -> dict[str, Any]:
    results = payload["results"]
    scored = [r for r in results if r["speedup"] != 1.0]
    best = max(scored, key=lambda r: r["speedup"])
    return {
        "device": payload["meta"]["device"],
        # Every run reproduced greedy decoding token for token. An implementation that
        # were merely close would not be a faster decoder, it would be a different model,
        # so the count of verified-lossless runs is part of the claim.
        "losslessRuns": sum(1 for r in results if r["output_matches_greedy"] is True),
        "divergedRuns": sum(1 for r in results if r["output_matches_greedy"] is False),
        "best": {
            "prompt": best["prompt"],
            "drafter": best["drafter"],
            "k": best["k"],
            "speedup": best["speedup"],
            "acceptanceRate": best["acceptance_rate"],
            "tokensPerTargetForward": best["tokens_per_target_forward"],
        },
        "rows": [
            {
                "prompt": r["prompt"],
                "drafter": r["drafter"],
                "k": r["k"],
                "tokensPerSec": r["tokens_per_sec"],
                "speedup": r["speedup"],
                "acceptanceRate": r["acceptance_rate"],
                "tokensPerTargetForward": r["tokens_per_target_forward"],
            }
            for r in scored
        ],
    }


# ----------------------------------------------------------------------------- render


def render(payload: dict[str, Any]) -> str:
    """JSON is a subset of TypeScript, so the literal needs no separate serialiser.

    ``as const`` is what makes it worth generating a module rather than a data file: the
    site gets literal types, so a page that reads `MEASURED.scaling.points[0].efficiency`
    is checked at build time against what the generator actually emitted.
    """
    return f"{HEADER}export const MEASURED = {json.dumps(payload, indent=2)} as const;\n"


def build(*, python_tests: int, browser_tests: int) -> str:
    return render(build_payload(python_tests=python_tests, browser_tests=browser_tests))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write; exit 1 if the committed module is stale",
    )
    args = parser.parse_args(argv)

    python_tests = count_python_tests()
    browser_tests = count_browser_tests()
    if browser_tests is None:
        browser_tests = committed_browser_tests()
        if browser_tests is None:
            raise SystemExit(
                "no browser test count available: run `npm ci --prefix web` so vitest can "
                "enumerate the suite, or generate this file on a machine that has it"
            )
        print(f"vitest unavailable — carrying the committed browser count ({browser_tests})")

    module = build(python_tests=python_tests, browser_tests=browser_tests)

    if args.check:
        current = OUT.read_text() if OUT.exists() else ""
        if current != module:
            raise SystemExit(f"{OUT.relative_to(ROOT)} is stale — run llmfs-export-web")
        print(f"{OUT.relative_to(ROOT)} is up to date")
        return

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(module)
    print(f"wrote {OUT.relative_to(ROOT)}  ({python_tests} python, {browser_tests} browser tests)")


if __name__ == "__main__":
    main()
