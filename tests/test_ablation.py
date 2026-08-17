"""Ablation sweep and reporting.

The tests that matter here are about the noise floor. An ablation table that
reports a 0.01 improvement as an improvement, when two runs of the *same* config
differ by 0.08, is worse than no table — it is a confident recommendation to make
a change that does nothing. So the rule "a delta smaller than the baseline seed
spread is not a result" is pinned from several directions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmfs.ablation.report import baseline_noise, compare, load_results, render_csv, render_markdown
from llmfs.ablation.sweep import DEFAULT_ARMS, ArmResult, _arm_name, _load_existing, _write
from llmfs.config import CONFIG_ROOT


def arm(name: str, seed: int = 1337, loss: float | None = 1.0, status: str = "completed") -> dict:
    return {
        "name": name,
        "config": f"{name}.yaml",
        "seed": seed,
        "status": status,
        "val_loss": loss,
        "perplexity": None,
        "steps": 100,
        "tokens": 100_000,
        "wall_clock_s": 10.0,
        "tokens_per_sec": 10_000.0,
        "params": 51_000_000,
        "run_dir": f"out/{name}",
        "error": None,
        "history": [{"step": 50, "val_loss": loss}, {"step": 100, "val_loss": loss}],
    }


def payload(arms: list[dict]) -> dict:
    return {"meta": {}, "arms": arms}


# ------------------------------------------------------------------ noise floor


def test_noise_floor_is_the_full_range_across_baseline_seeds() -> None:
    mean, spread, n = baseline_noise(
        [arm("baseline", 1, 2.00), arm("baseline", 2, 2.06), arm("baseline", 3, 2.03)]
    )
    assert mean == pytest.approx(2.03)
    assert spread == pytest.approx(0.06)
    assert n == 3


def test_single_baseline_seed_reports_no_spread() -> None:
    mean, spread, n = baseline_noise([arm("baseline", 1, 2.0)])
    assert (mean, spread, n) == (2.0, 0.0, 1)


def test_diverged_baseline_runs_are_excluded_from_the_floor() -> None:
    mean, spread, n = baseline_noise(
        [arm("baseline", 1, 2.0), arm("baseline", 2, None, "diverged")]
    )
    assert n == 1 and mean == 2.0


def test_delta_inside_the_noise_floor_is_not_a_result() -> None:
    """The central rule. A 0.02 gain against a 0.10 seed spread means nothing."""
    rows, stats = compare(
        payload(
            [
                arm("baseline", 1, 2.00),
                arm("baseline", 2, 2.10),
                arm("baseline", 3, 2.05),
                arm("tiny-gain", 1337, 2.03),  # 0.02 better than the 2.05 mean
            ]
        )
    )
    assert stats["baseline_spread"] == pytest.approx(0.10)

    tiny = next(r for r in rows if r.name == "tiny-gain")
    assert tiny.delta == pytest.approx(-0.02)
    assert tiny.significant is False
    assert tiny.verdict == "within noise"


def test_delta_beyond_the_noise_floor_counts() -> None:
    rows, _ = compare(
        payload(
            [
                arm("baseline", 1, 2.00),
                arm("baseline", 2, 2.02),
                arm("real-gain", 1337, 1.80),
                arm("real-loss", 1337, 2.30),
            ]
        )
    )
    by = {r.name: r for r in rows}
    assert by["real-gain"].significant and by["real-gain"].verdict == "better"
    assert by["real-loss"].significant and by["real-loss"].verdict == "worse"


def test_diverged_arm_is_reported_not_dropped() -> None:
    """A diverged arm is a finding about that hyperparameter, not a missing row."""
    rows, _ = compare(payload([arm("baseline", 1, 2.0), arm("lr-3e-3", 1337, None, "diverged")]))
    diverged = next(r for r in rows if r.name == "lr-3e-3")
    assert diverged.verdict == "diverged"
    assert diverged.val_loss is None


def test_baseline_row_has_no_delta_against_itself() -> None:
    rows, _ = compare(payload([arm("baseline", 1, 2.0), arm("baseline", 2, 2.1)]))
    base = next(r for r in rows if r.name == "baseline")
    assert base.delta is None and base.verdict == "baseline"
    assert base.val_loss == pytest.approx(2.05)  # the mean over seeds


# --------------------------------------------------------------------- report


def test_markdown_states_the_noise_floor_and_labels_weak_arms() -> None:
    rows, stats = compare(
        payload(
            [
                arm("baseline", 1, 2.00),
                arm("baseline", 2, 2.10),
                arm("weak", 1337, 2.04),
                arm("strong", 1337, 1.70),
            ]
        )
    )
    md = render_markdown(rows, stats)
    assert "Seed noise floor" in md
    assert "within noise" in md
    assert "`strong`" in md
    assert "Caveats" in md


def test_seed_caveat_describes_the_seeds_the_arms_actually_ran() -> None:
    """The caveat was a hardcoded string, and the published report was wrong because of it.

    ``results/ablations.md`` told its readers "each non-baseline arm is a single seed"
    while the run behind it used three per arm, 39 runs in all — and the same file's own
    header described the paired multi-seed design. A generated document asserting a
    methodology it did not use is worse than no caveat, because it reads as verified.
    """
    seeds = (1337, 1338, 1339)
    base = [arm("baseline", s, 2.00 + i * 0.05) for i, s in enumerate(seeds)]
    wide = [arm("wide", s, 1.90 + i * 0.05) for i, s in enumerate(seeds)]

    rows, stats = compare(payload(base + wide))
    md = render_markdown(rows, stats)
    assert "ran all 3 seeds" in md
    assert "is a single seed" not in md
    assert [r.n_seeds for r in rows if r.delta is not None] == [3]

    # One seed per arm: the original caveat, now earned rather than assumed.
    single = render_markdown(*compare(payload(base + [arm("wide", 1337, 1.90)])))
    assert "Each non-baseline arm is a single seed" in single

    # And a mixed sweep names the arms that are thin instead of generalising either way.
    mixed = render_markdown(*compare(payload(base + wide + [arm("thin", 1337, 1.95)])))
    assert "Seed coverage is uneven" in mixed
    assert "`thin`" in mixed and "`wide`" not in mixed.split("Seed coverage is uneven")[1]


def test_markdown_warns_when_no_noise_floor_was_measured() -> None:
    """A single-seed baseline cannot support any claim, and must say so."""
    rows, stats = compare(payload([arm("baseline", 1, 2.0), arm("other", 1337, 1.9)]))
    md = render_markdown(rows, stats)
    assert "No noise floor was measured" in md
    assert "--seeds" in md


# ------------------------------------------------------------------ paired design


def paired(base: list[float], other: list[float], seeds=(1, 2, 3)) -> dict:
    """Baseline and one arm, run at the same seeds."""
    arms = [arm("baseline", s, v) for s, v in zip(seeds, base)]
    arms += [arm("armX", s, v) for s, v in zip(seeds, other)]
    return payload(arms)


def test_pairing_resolves_an_effect_smaller_than_the_raw_noise() -> None:
    """The entire reason for running every arm at the same seeds.

    The baseline swings 0.20 across seeds, so an unpaired comparison could never
    claim a 0.03 effect. But the arm is better by ~0.03 *in every seed*, so the
    paired differences agree and the effect is real.
    """
    rows, stats = compare(paired([2.00, 2.10, 2.20], [1.97, 2.07, 2.17]))
    assert stats["baseline_spread"] == pytest.approx(0.20)

    x = next(r for r in rows if r.name == "armX")
    assert x.paired is True and x.n_seeds == 3
    assert x.delta == pytest.approx(-0.03)
    assert x.half_range == pytest.approx(0.0, abs=1e-9)
    assert x.significant is True, "consistent per-seed improvement must count"
    assert x.verdict == "better"
    # Unpaired, this same data would have been dismissed.
    assert abs(x.delta) < stats["baseline_spread"]


def test_inconsistent_sign_is_not_a_result() -> None:
    """If the seeds disagree about the direction, there is no effect to report."""
    rows, _ = compare(paired([2.00, 2.10, 2.20], [1.95, 2.15, 2.18]))
    x = next(r for r in rows if r.name == "armX")
    assert sorted(round(d, 2) for d in x.deltas) == [-0.05, -0.02, 0.05]
    assert x.significant is False
    assert x.verdict == "within noise"


def test_half_range_is_the_reported_error_bar() -> None:
    rows, _ = compare(paired([2.00, 2.00, 2.00], [1.90, 1.94, 1.98]))
    x = next(r for r in rows if r.name == "armX")
    assert x.delta == pytest.approx(-0.06)
    assert x.half_range == pytest.approx(0.04)
    assert x.significant is True  # range [-0.10, -0.02] stays below zero


def test_markdown_shows_paired_error_bars() -> None:
    rows, stats = compare(paired([2.00, 2.10, 2.20], [1.97, 2.07, 2.17]))
    md = render_markdown(rows, stats)
    assert "paired" in md.lower()
    assert "±" in md
    assert "does not straddle zero" in md


def test_arm_missing_a_seed_still_pairs_on_what_it_shares() -> None:
    """A crashed run must not invalidate the arm; it pairs on the seeds it has."""
    arms = [arm("baseline", s, v) for s, v in zip((1, 2, 3), (2.00, 2.10, 2.20))]
    arms += [arm("armX", 1, 1.97), arm("armX", 3, 2.17)]
    rows, _ = compare(payload(arms))
    x = next(r for r in rows if r.name == "armX")
    assert x.n_seeds == 2 and x.paired is True
    assert x.delta == pytest.approx(-0.03)


def test_unpaired_falls_back_when_seeds_do_not_overlap() -> None:
    """Arms run at a different seed than the baseline cannot be paired."""
    arms = [arm("baseline", s, v) for s, v in zip((1, 2, 3), (2.00, 2.10, 2.20))]
    arms += [arm("armX", 99, 1.50)]
    rows, _ = compare(payload(arms))
    x = next(r for r in rows if r.name == "armX")
    assert x.paired is False
    assert x.delta == pytest.approx(1.50 - 2.10)
    assert x.significant is True  # 0.60 clears the 0.20 unpaired spread


def test_markdown_handles_no_baseline_at_all() -> None:
    rows, stats = compare(payload([arm("other", 1337, 1.9)]))
    assert "nothing to compare against" in render_markdown(rows, stats)


def test_csv_has_a_row_per_arm() -> None:
    rows, _ = compare(payload([arm("baseline", 1, 2.0), arm("x", 1337, 1.9)]))
    lines = render_csv(rows).splitlines()
    assert lines[0].startswith("arm,axis,status")
    assert len(lines) == 3


# ---------------------------------------------------------------------- sweep


def test_results_round_trip_through_disk(tmp_path: Path) -> None:
    """A sweep killed mid-way must be able to read back what it finished."""
    results = {
        "baseline@seed1": ArmResult(name="baseline", config="c", seed=1, status="completed"),
        "x@seed1": ArmResult(name="x", config="c", seed=1, status="diverged"),
    }
    path = tmp_path / "r.json"
    _write(path, results, {"note": "meta survives"})

    restored = _load_existing(path)
    assert set(restored) == {"baseline@seed1", "x@seed1"}
    assert restored["x@seed1"].status == "diverged"
    assert json.loads(path.read_text())["meta"]["note"] == "meta survives"


def test_missing_results_file_is_an_empty_start(tmp_path: Path) -> None:
    assert _load_existing(tmp_path / "absent.json") == {}


def test_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Written to a temp path and renamed into place — observed, not inferred.

    Asserting only "the file exists and no .tmp is left" is equally true of a plain
    in-place write, which is the write this forbids: the sweep rewrites this file after
    every arm, and a crash during one of those rewrites would take the previous 38 runs
    with it.
    """
    arm = {"a@seed1": ArmResult(name="a", config="c", seed=1, status="completed")}
    path = tmp_path / "r.json"

    renames: list[tuple[Path, Path]] = []
    real_replace = Path.replace

    def replace_spy(self, target):
        renames.append((Path(self), Path(target)))
        return real_replace(self, target)

    monkeypatch.setattr(Path, "replace", replace_spy)
    _write(path, arm, {})
    monkeypatch.undo()

    assert len(renames) == 1, f"expected one rename into place, got {renames}"
    src, dst = renames[0]
    assert src != path and dst == path
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))


def test_a_crash_mid_write_leaves_the_previous_results_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arm = {"a@seed1": ArmResult(name="a", config="c", seed=1, status="completed")}
    path = tmp_path / "r.json"
    _write(path, arm, {"round": 1})
    good = path.read_text()

    real_write_text = Path.write_text

    def dies_after_writing(self, *args, **kwargs):
        real_write_text(self, *args, **kwargs)
        raise RuntimeError("pod terminated mid-sweep")

    monkeypatch.setattr(Path, "write_text", dies_after_writing)
    with pytest.raises(RuntimeError, match="terminated"):
        _write(path, arm, {"round": 2})
    monkeypatch.undo()

    assert path.read_text() == good
    assert json.loads(path.read_text())["meta"] == {"round": 1}


def test_base_config_is_named_baseline() -> None:
    assert _arm_name(Path("configs/ablations/_base.yaml")) == "baseline"
    assert _arm_name(Path("configs/ablations/gqa-2.yaml")) == "gqa-2"


def test_every_default_arm_config_exists() -> None:
    """The sweep must not die on arm 9 of 13 because a filename drifted."""
    for name in DEFAULT_ARMS:
        assert (CONFIG_ROOT / "ablations" / name).exists(), name


def test_default_arms_cover_every_shipped_ablation_config() -> None:
    """A new config in configs/ablations/ should not be silently left out of the sweep."""
    on_disk = {p.name for p in (CONFIG_ROOT / "ablations").glob("*.yaml")}
    assert on_disk == set(DEFAULT_ARMS)


def test_load_results_rejects_a_foreign_file(tmp_path: Path) -> None:
    path = tmp_path / "nope.json"
    path.write_text(json.dumps({"something": "else"}))
    with pytest.raises(ValueError, match="does not look like sweep results"):
        load_results(path)
