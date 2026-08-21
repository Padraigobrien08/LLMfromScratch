"""Turn sweep results into a table, plots, and a write-up scaffold.

The rule this module enforces: **an inconsistent difference is not a result.** Every
arm runs at the same seeds as the baseline, each is differenced against the baseline
run that saw its data in the same order, and an arm only counts when those per-seed
differences all agree in sign. Ablation tables without such a check are how a study
ends up confidently recommending a change that does nothing.

Pairing is what makes this affordable. Comparing means alone, an effect would have to
exceed the entire baseline seed spread — larger than most architecture changes
produce. Differencing within a seed cancels the batch-ordering variance both runs
share, so a much smaller effect becomes visible without longer runs.

Where an arm cannot be paired (no shared seed), the report falls back to the unpaired
test against the baseline spread and says so. Neither test is a p-value; with three
seeds nothing stronger would be honest.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# What each arm is meant to isolate, for the report's "axis" column. Keeping this
# next to the presentation rather than in the config means a reader of the table
# does not have to open thirteen YAML files to know what varied.
AXIS = {
    "baseline": "—",
    "norm-rmsnorm": "LayerNorm → RMSNorm",
    "pos-rope": "learned positions → RoPE",
    "pos-none": "learned positions → none",
    "mlp-swiglu": "GELU → SwiGLU (param-matched)",
    "untied-embeddings": "tied → untied embeddings",
    "no-bias": "bias → no bias",
    "gqa-2": "8 KV heads → 2 (GQA)",
    "sched-wsd": "cosine → WSD schedule",
    "wd-zero": "weight decay 0.1 → 0",
    "lr-3e-4": "lr 1e-3 → 3e-4",
    "lr-3e-3": "lr 1e-3 → 3e-3",
    "modern-stack": "all modern components",
}


@dataclass
class Comparison:
    name: str
    axis: str
    status: str
    val_loss: float | None
    perplexity: float | None
    delta: float | None
    """Mean signed change vs the baseline. Negative is better (lower loss)."""
    significant: bool
    tokens_per_sec: float
    wall_clock_s: float
    params: int
    deltas: list[float] = field(default_factory=list)
    """Per-seed paired differences, when the arm and baseline share seeds."""
    paired: bool = False
    n_seeds: int = 0

    @property
    def half_range(self) -> float:
        """Half the spread of the paired deltas — the error bar on ``delta``."""
        return (max(self.deltas) - min(self.deltas)) / 2 if len(self.deltas) > 1 else 0.0

    @property
    def verdict(self) -> str:
        if self.status == "diverged":
            return "diverged"
        if self.status != "completed":
            return "failed"
        if self.delta is None:
            return "baseline"
        if not self.significant:
            return "within noise"
        return "better" if self.delta < 0 else "worse"


def load_results(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if "arms" not in payload:
        raise ValueError(f"{path} does not look like sweep results (no 'arms' key)")
    return payload


def baseline_noise(arms: list[dict]) -> tuple[float | None, float, int]:
    """Return ``(mean, spread, n)`` over the completed baseline seeds.

    ``spread`` is the full range (max − min), not a standard deviation. With three
    or four seeds a standard deviation is barely estimable and reads as more precise
    than it is; the range is a blunt, honest statement of how much these runs move.
    """
    losses = [
        a["val_loss"]
        for a in arms
        if a["name"] == "baseline" and a["status"] == "completed" and a["val_loss"] is not None
    ]
    if not losses:
        return None, 0.0, 0
    if len(losses) == 1:
        return losses[0], 0.0, 1
    return statistics.fmean(losses), max(losses) - min(losses), len(losses)


def _is_significant(delta: float, deltas: list[float], paired: bool, spread: float) -> bool:
    """Does this arm's difference from the baseline count as a result?

    Paired: yes only if the range of per-seed differences excludes zero — every seed
    agreed on the direction, and the disagreement between them is smaller than the
    effect itself.

    Unpaired (a single shared seed, so nothing to pair against): fall back to
    requiring the difference to clear the baseline's whole seed spread, which is a
    much higher bar and the reason pairing is worth its compute.
    """
    if paired:
        return min(deltas) > 0 or max(deltas) < 0
    return abs(delta) > spread


def compare(payload: dict[str, Any]) -> tuple[list[Comparison], dict[str, Any]]:
    """Compare each arm against the baseline, paired by seed where possible.

    When an arm and the baseline share seeds, the comparison is done *within* each
    seed and the per-seed differences are averaged. That cancels the run-to-run
    variation both share — principally the order the batches arrived in — and leaves
    the effect of the design change. An unpaired comparison of means has to clear the
    entire seed spread before it can claim anything; a paired one only has to produce
    differences that agree with each other.

    Significance rule, stated plainly because it is a heuristic and not a p-value:
    **an arm counts only if the range of its per-seed deltas does not straddle zero.**
    With three seeds that is a weak test in the formal sense, but it is honest, it is
    what the error bars in the plot show, and it cannot manufacture a result out of
    an effect whose sign is not even consistent.
    """
    arms = payload["arms"]
    mean, spread, n_seeds = baseline_noise(arms)

    def completed_by_seed(runs: list[dict]) -> dict[int, float]:
        return {
            r["seed"]: r["val_loss"]
            for r in runs
            if r["status"] == "completed" and r["val_loss"] is not None
        }

    by_name: dict[str, list[dict]] = {}
    for arm in arms:
        by_name.setdefault(arm["name"], []).append(arm)

    baseline_by_seed = completed_by_seed(by_name.get("baseline", []))

    rows: list[Comparison] = []
    for name, runs in by_name.items():
        arm_by_seed = completed_by_seed(runs)
        completed = [r for r in runs if r["status"] == "completed" and r["val_loss"] is not None]
        representative = completed[0] if completed else runs[0]
        loss = statistics.fmean(arm_by_seed.values()) if arm_by_seed else None

        deltas: list[float] = []
        paired = False
        if name != "baseline" and arm_by_seed and baseline_by_seed:
            shared = sorted(set(arm_by_seed) & set(baseline_by_seed))
            if shared:
                deltas = [arm_by_seed[s] - baseline_by_seed[s] for s in shared]
                paired = len(shared) > 1

        if deltas:
            delta = statistics.fmean(deltas)
            significant = _is_significant(delta, deltas, paired, spread)
        elif name == "baseline" or loss is None or mean is None:
            delta, significant = None, False
        else:
            delta = loss - mean
            significant = abs(delta) > spread

        rows.append(
            Comparison(
                name=name,
                axis=AXIS.get(name, "—"),
                status="completed" if completed else representative["status"],
                val_loss=loss,
                perplexity=math.exp(min(loss, 20)) if loss is not None else None,
                delta=delta,
                significant=significant,
                tokens_per_sec=statistics.fmean([r["tokens_per_sec"] for r in runs]),
                wall_clock_s=statistics.fmean([r["wall_clock_s"] for r in runs]),
                params=representative.get("params", 0),
                deltas=deltas,
                paired=paired,
                n_seeds=len(arm_by_seed),
            )
        )

    rows.sort(key=lambda r: (r.name != "baseline", r.delta if r.delta is not None else 0.0))
    stats = {
        "baseline_mean": mean,
        "baseline_spread": spread,
        "baseline_seeds": n_seeds,
        "paired": any(r.paired for r in rows),
        "meta": payload.get("meta", {}),
    }
    return rows, stats


def seed_caveat(rows: list[Comparison]) -> str:
    """The seed caveat the arms actually earn.

    Three cases, because they carry genuinely different warnings: single-seed arms are
    judged against someone else's spread, multi-seed arms carry their own error bar, and a
    mixed sweep has to say which arms are which rather than pick the flattering half.
    """
    arms = [r for r in rows if r.delta is not None and r.status == "completed"]
    counts = {r.n_seeds for r in arms}
    if not arms or counts <= {0, 1}:
        return (
            "- Each non-baseline arm is a single seed. Its delta is judged against the "
            "baseline's spread, which assumes the arms have comparable seed sensitivity, "
            "reasonable for architecture changes, less so for learning-rate arms near the "
            "stability boundary."
        )
    if min(counts) > 1:
        n = min(counts)
        same = f"all {n}" if len(counts) == 1 else f"at least {n}"
        return (
            f"- Every non-baseline arm ran {same} seeds, so each delta carries its own "
            "measured spread rather than borrowing the baseline's. That is what makes the "
            "paired comparison above legitimate; it does not make the arms independent of "
            "the seeds they share with the baseline."
        )
    single = sorted(r.name for r in arms if r.n_seeds <= 1)
    return (
        "- Seed coverage is uneven: " + ", ".join(f"`{n}`" for n in single) + " ran a "
        "single seed and so have no spread of their own, while the remaining arms do. "
        "The single-seed deltas are judged against the baseline's spread instead, which "
        "assumes comparable seed sensitivity."
    )


def render_markdown(rows: list[Comparison], stats: dict[str, Any]) -> str:
    mean, spread, n = stats["baseline_mean"], stats["baseline_spread"], stats["baseline_seeds"]

    lines = ["# Ablation results", ""]

    if n == 0:
        lines += ["No completed baseline runs; nothing to compare against.", ""]
        return "\n".join(lines)

    lines += [
        f"Baseline: **{mean:.4f}** validation loss, "
        f"{'over ' + str(n) + ' seeds' if n > 1 else 'single seed'}."
    ]
    paired = stats.get("paired", False)

    if n > 1:
        lines += [
            "",
            f"**Seed noise floor: {spread:.4f}**, the full range across {n} baseline "
            f"runs differing only in seed. It is the scale against which any *unpaired* "
            f"comparison would have to be judged.",
        ]
    else:
        lines += [
            "",
            "**No noise floor was measured**: the baseline ran with a single seed, so "
            "no delta below can be distinguished from run-to-run variation. Re-run with "
            "`--seeds 3` before drawing conclusions.",
        ]

    if paired:
        lines += [
            "",
            "Comparisons below are **paired**: every arm ran at the same seeds as the "
            "baseline, and each is differenced against the baseline run that saw its "
            "data in the same order. That cancels the batch-ordering variance the two "
            "share, so an effect well below the raw noise floor above can still be "
            "resolved. The ± is the half-range of the per-seed differences.",
            "",
            "An arm counts as a result only when **the range of its per-seed deltas "
            "does not straddle zero**, meaning every seed agreed on the direction. This is a "
            "deliberately blunt rule, not a p-value; with three seeds nothing stronger "
            "would be honest.",
        ]

    lines += [
        "",
        "| Arm | What varied | Val loss | Δ vs baseline | Verdict | Tokens/s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        loss = f"{r.val_loss:.4f}" if r.val_loss is not None else "—"
        if r.delta is None:
            delta = "—"
        elif r.paired:
            delta = f"{r.delta:+.4f} ± {r.half_range:.4f}"
        else:
            delta = f"{r.delta:+.4f}"
        verdict = r.verdict
        if verdict == "better":
            verdict = "**better**"
        elif verdict == "diverged":
            verdict = "diverged ⚠"
        lines.append(
            f"| `{r.name}` | {r.axis} | {loss} | {delta} | {verdict} | {r.tokens_per_sec:,.0f} |"
        )

    significant = [r for r in rows if r.delta is not None and r.significant]
    noise = [
        r for r in rows if r.delta is not None and not r.significant and r.status == "completed"
    ]
    diverged = [r for r in rows if r.status == "diverged"]

    # The justification differs between the two designs, and saying "beyond the noise
    # floor" about a paired result would misdescribe how it was established.
    basis = "consistently across every seed" if paired else "beyond the seed noise floor"

    lines += ["", "## What this shows", ""]
    if significant:
        better = [r for r in significant if r.delta < 0]
        worse = [r for r in significant if r.delta > 0]
        if better:
            lines.append(
                f"Improved {basis}: "
                + ", ".join(f"`{r.name}` ({r.delta:+.4f})" for r in better)
                + "."
            )
        if worse:
            lines.append(
                f"Hurt {basis}: " + ", ".join(f"`{r.name}` ({r.delta:+.4f})" for r in worse) + "."
            )
    else:
        lines.append(
            "No arm changed validation loss "
            + ("consistently across seeds." if paired else "beyond the seed noise floor.")
        )

    if noise:
        lines += [
            "",
            "Indistinguishable from the baseline at this scale: "
            + ", ".join(f"`{r.name}`" for r in noise)
            + ". That is a finding in its own right — a component that costs nothing "
            "and changes nothing is still worth keeping if it buys something else, "
            "which is why `gqa-2` should be read against the inference benchmarks "
            "rather than against loss alone.",
        ]

    if diverged:
        lines += [
            "",
            "Diverged: " + ", ".join(f"`{r.name}`" for r in diverged) + ".",
        ]

    lines += [
        "",
        "## Caveats",
        "",
        f"- Run at ablation scale ({rows[0].params / 1e6:.0f}M parameters), not at the "
        "124M reproduction scale. Conclusions transfer in direction, not in magnitude.",
        # This caveat used to be emitted unconditionally, and the published report
        # therefore told its readers that every arm was a single seed while the run it
        # described used three — a false statement about methodology, in a generated file,
        # contradicted by that file's own header. What the arms actually ran now decides
        # which caveat is printed.
        seed_caveat(rows),
        "- Validation loss is the only metric here. A change that leaves loss alone but "
        "shrinks the KV cache or speeds up a step is measured elsewhere.",
        "",
        "*Generated by `llmfs-ablate-report`.*",
    ]
    return "\n".join(lines)


def render_csv(rows: list[Comparison]) -> str:
    header = (
        "arm,axis,status,val_loss,perplexity,delta,significant,tokens_per_sec,wall_clock_s,params"
    )
    lines = [header]
    for r in rows:
        lines.append(
            f'{r.name},"{r.axis}",{r.status},'
            f"{'' if r.val_loss is None else f'{r.val_loss:.6f}'},"
            f"{'' if r.perplexity is None else f'{r.perplexity:.4f}'},"
            f"{'' if r.delta is None else f'{r.delta:.6f}'},"
            f"{r.significant},{r.tokens_per_sec:.1f},{r.wall_clock_s:.1f},{r.params}"
        )
    return "\n".join(lines)


def plot(payload: dict[str, Any], rows: list[Comparison], stats: dict[str, Any], out_dir: Path):
    """Loss curves and a delta bar chart. Skipped if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skipping plots (pip install -e '.[bench]')")
        return []

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []

    # --- loss curves ---
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for arm in payload["arms"]:
        history = arm.get("history") or []
        if not history:
            continue
        steps = [h["step"] for h in history]
        losses = [h["val_loss"] for h in history]
        baseline = arm["name"] == "baseline"
        ax.plot(
            steps,
            losses,
            label=f"{arm['name']}" if not baseline else f"baseline (seed {arm['seed']})",
            linewidth=2.2 if baseline else 1.4,
            color="black" if baseline else None,
            alpha=1.0 if baseline else 0.85,
            zorder=3 if baseline else 2,
        )
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title("Ablation arms: validation loss")
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = out_dir / "ablation_curves.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(path)

    # --- deltas ---
    deltas = [r for r in rows if r.delta is not None and r.status == "completed"]
    if deltas:
        fig, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(deltas))))
        names = [r.name for r in deltas]
        values = [r.delta for r in deltas]
        colours = [
            ("#2f5fe0" if r.delta < 0 else "#d1425c") if r.significant else "#b9bec9"
            for r in deltas
        ]
        # Error bars are the half-range of the per-seed paired differences. Whether a
        # bar crosses zero *is* the significance test, so it has to be visible rather
        # than only asserted in the table.
        errors = [r.half_range for r in deltas]
        ax.barh(
            names,
            values,
            color=colours,
            xerr=errors if any(errors) else None,
            error_kw={"ecolor": "#5b6270", "elinewidth": 1.1, "capsize": 3},
        )
        spread = stats["baseline_spread"]
        if spread > 0 and not stats.get("paired"):
            # Only meaningful unpaired; with pairing the error bars are the relevant
            # scale and this band would overstate the uncertainty.
            ax.axvspan(-spread, spread, color="#b9bec9", alpha=0.28, zorder=0)
            ax.text(0, len(names) - 0.4, f"  seed noise ±{spread:.4f}", fontsize=8, color="#5b6270")
        ax.axvline(0, color="black", linewidth=1)
        ax.set_xlabel("Δ validation loss vs baseline  (negative = better)")
        ax.set_title(
            "Ablation deltas — paired by seed; grey bars straddle zero"
            if stats.get("paired")
            else "Ablation deltas — grey bars are within seed noise"
        )
        ax.grid(axis="x", alpha=0.25, linewidth=0.6)
        fig.tight_layout()
        path = out_dir / "ablation_deltas.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        written.append(path)

    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report on ablation sweep results.")
    parser.add_argument("--results", type=str, default="results/ablations.json")
    parser.add_argument("--out-dir", type=str, default="results")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)

    payload = load_results(args.results)
    rows, stats = compare(payload)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    markdown = render_markdown(rows, stats)
    (out_dir / "ablations.md").write_text(markdown)
    (out_dir / "ablations.csv").write_text(render_csv(rows))

    written = [] if args.no_plots else plot(payload, rows, stats, out_dir)

    print(markdown)
    print(f"\nwrote {out_dir / 'ablations.md'}, {out_dir / 'ablations.csv'}")
    for path in written:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
