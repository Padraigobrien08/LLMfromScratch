"""The two artifacts the results plates added, and the properties their pages assert.

`reproduction-curve.json` and `benchmarks-cuda-before-mask-fix.json` are not measured by
anything in this suite — one came out of a rented H100's log, the other out of the commit
before a bug was fixed. What *can* be checked is that they still say what the site says
they say, because each page is built around a specific claim about their shape rather
than around a number they contain.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text())


# ------------------------------------------------------------------ the loss curve


def test_the_curve_is_the_run_the_scalar_artifact_describes() -> None:
    curve, repro = load("reproduction-curve.json"), load("reproduction.json")
    assert curve["finalStep"] >= repro["step"], "the curve must cover the evaluated step"
    assert curve["train"], "no training points"
    assert curve["val"], "no validation points"
    # The periodic evaluations run on fewer batches than the final full-split one, so
    # they are close rather than equal — asserting equality would fail for a correct run.
    assert abs(curve["val"][-1]["loss"] - repro["loss"]) < 0.1


def test_the_target_was_met_early_enough_for_the_claim_the_plate_makes() -> None:
    """The plate says the run met its target a third of the way in and kept improving.

    Both halves matter and both are checkable. A target met on the final step would be a
    target chosen to be hit; a run that met it and then got worse would make the final
    number a matter of when someone stopped looking.
    """
    curve = load("reproduction-curve.json")
    crossing = curve["crossing"]
    assert crossing is not None, "the run never met its target"
    assert crossing["loss"] <= curve["targetLoss"]
    assert crossing["fractionOfRun"] < 0.5, "met the target more than halfway through"

    earlier = [p for p in curve["val"] if p["step"] < crossing["step"]]
    assert all(p["loss"] > curve["targetLoss"] for p in earlier), "not the first crossing"
    assert curve["val"][-1]["loss"] < crossing["loss"], "the run did not keep improving"


def test_utilisation_held_flat_after_the_first_logged_step() -> None:
    """The claim the reproduction plate makes about MFU, which is about *flatness*.

    A mean alone cannot support it: a run that halves its throughput partway through can
    have the same mean as one that never moves, and the two say completely different
    things about whether the input pipeline kept up.
    """
    mfus = [p["mfu"] for p in load("reproduction-curve.json")["train"] if p["mfu"] is not None]
    settled = mfus[1:]
    assert max(settled) - min(settled) < 0.05, "utilisation drifted by more than five points"
    assert mfus[0] < min(settled), "the first step is supposed to be the warmup outlier"


# --------------------------------------------------------------- the mask-bug pair


@pytest.fixture(scope="module")
def sweeps() -> tuple[dict, dict]:
    def sweep(name: str) -> dict:
        payload = load(name)
        return {
            # Keyed by (total length, cached?), which is the only pair of settings the
            # cache-scaling suite varies.
            "points": {
                (r["settings"]["total_len"], r["settings"]["use_cache"]): r["tokens_per_sec"]
                for r in payload["results"]
                if r["suite"] == "cache-scaling"
            },
            "commit": payload["provenance"]["git_commit"],
            "gpu": payload["provenance"]["gpu"],
        }

    return sweep("benchmarks-cuda-before-mask-fix.json"), sweep("benchmarks-cuda.json")


def test_the_two_cache_sweeps_are_the_same_experiment_at_two_commits(sweeps) -> None:
    before, after = sweeps
    assert before["commit"] != after["commit"], (
        "the whole figure is a before-and-after; identical commits mean one of the files "
        "was overwritten with the other"
    )
    assert before["gpu"] == after["gpu"], "a different card would confound the comparison"
    lengths = {length for length, _ in before["points"]}
    assert lengths == {length for length, _ in after["points"]}
    assert lengths, "no cache-scaling rows in either file"


def test_the_mask_bug_lost_at_every_length_and_the_fix_wins_at_the_longest(sweeps) -> None:
    """The efficiency plate's toggle, stated as the assertion it makes.

    If a re-measurement ever removed the inversion, the page would keep flipping between
    two curves while the sentence beside it — "the cache loses at every length" — quietly
    became false. This is that sentence.
    """
    before, after = sweeps
    lengths = sorted(length for length, cached in before["points"] if cached)

    for length in lengths:
        assert before["points"][(length, True)] < before["points"][(length, False)], (
            f"with the mask, the cache is supposed to lose at {length}"
        )

    longest = lengths[-1]
    assert after["points"][(longest, True)] > after["points"][(longest, False)], (
        f"without the mask, the cache is supposed to win at {longest}"
    )


def test_the_recompute_path_is_the_untouched_control(sweeps) -> None:
    """What makes the pair a measurement rather than a coincidence.

    The fix touched only the cached path. If the recompute numbers had also moved, the
    two runs would differ by something other than the change under test — a driver, a
    thermal state, a different machine — and the whole comparison would be worthless.
    """
    before, after = sweeps
    for length in sorted(length for length, cached in before["points"] if not cached):
        ratio = after["points"][(length, False)] / before["points"][(length, False)]
        assert 0.95 < ratio < 1.05, f"the control moved {ratio:.2f}× at {length}"


def test_the_reproduction_wall_clock_is_the_stage_timing_that_was_recorded() -> None:
    """The "7.1 h of training" figure was in neither the log nor the stepping arithmetic.

    The pipeline recorded 25,167 s for the `repro` stage — 6.99 h — and 19,073 steps at
    1,305 ms/step is 6.91 h of stepping, so 7.1 was between two real numbers and equal to
    neither. The run log itself is 1.4 MB of progress redraws and is not committed, so the
    stage timings are, and the document is checked against them.
    """
    import re

    stages = (RESULTS / "run-stages.log").read_text()
    seconds = int(re.search(r"stage 'repro': done in (\d+)s", stages).group(1))
    hours = seconds / 3600

    doc = (RESULTS.parent / "docs" / "reproduction.md").read_text()
    assert f"{hours:.2f} h" in doc, f"the repro stage took {hours:.2f} h"
    assert f"{seconds:,} s" in doc
