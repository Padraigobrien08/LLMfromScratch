"""Run the ablation sweep.

The sweep exists to answer "does this design choice help?", and the honest version
of that question needs a second one answered first: **how big a difference is big
enough to mean anything?**

Two identical runs differing only in seed will not reach the same validation loss.
If that spread is 0.02 and an arm beats the baseline by 0.01, the arm has measured
nothing. So the baseline is run with several seeds and the spread between them
becomes the noise floor that every reported delta is judged against. Without it a
sweep produces a table of numbers with no way to tell which are real — which is
worse than no table, because it looks authoritative.

Other properties the runner needs, all learned from the fact that this is a
multi-hour job on rented hardware:

* **Resumable at arm granularity.** A completed arm is never re-run, so a sweep
  killed at hour six restarts at hour six.
* **Divergence is a result, not a crash.** The high-learning-rate arm is *expected*
  to blow up. That gets recorded and the sweep continues.
* **Every arm's outcome is written as it finishes**, so a sweep that dies still
  leaves the arms that completed.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ..config import CONFIG_ROOT, load_config
from ..train.trainer import Trainer, TrainingDiverged

DEFAULT_ARMS = [
    "_base.yaml",
    "norm-rmsnorm.yaml",
    "pos-rope.yaml",
    "pos-none.yaml",
    "mlp-swiglu.yaml",
    "untied-embeddings.yaml",
    "no-bias.yaml",
    "gqa-2.yaml",
    "sched-wsd.yaml",
    "wd-zero.yaml",
    "lr-3e-4.yaml",
    "lr-3e-3.yaml",
    "modern-stack.yaml",
]


@dataclass
class ArmResult:
    name: str
    config: str
    seed: int
    status: str  # completed | diverged | failed
    val_loss: float | None = None
    perplexity: float | None = None
    steps: int = 0
    tokens: int = 0
    wall_clock_s: float = 0.0
    tokens_per_sec: float = 0.0
    params: int = 0
    run_dir: str = ""
    error: str | None = None
    history: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.name}@seed{self.seed}"


def _arm_name(config_path: Path) -> str:
    return "baseline" if config_path.stem == "_base" else config_path.stem


def run_arm(config_path: Path, seed: int, out_dir: Path, extra: list[str]) -> ArmResult:
    """Train one arm and return its outcome, whatever that outcome is."""
    name = _arm_name(config_path)
    run_name = f"{name}-seed{seed}"

    overrides = [f"runtime.seed={seed}", f"log.run_name={run_name}", f"log.out_dir={out_dir}"]
    cfg = load_config(config_path, overrides + list(extra))

    result = ArmResult(
        name=name,
        config=str(config_path),
        seed=seed,
        status="failed",
        run_dir=str(Path(out_dir) / run_name),
    )

    print(f"\n{'=' * 68}\narm: {name}  seed {seed}\n{'=' * 68}")
    started = time.perf_counter()
    try:
        trainer = Trainer(cfg)
        result.params = sum(p.numel() for p in trainer.model.parameters())
        state = trainer.train()

        result.status = "completed"
        result.val_loss = state.best_val_loss
        result.perplexity = math.exp(min(state.best_val_loss, 20))
        result.steps = state.step
        result.tokens = state.tokens_seen
        result.history = state.history
    except TrainingDiverged as exc:
        # Expected for the high-learning-rate arm. A real finding, reported as one.
        result.status = "diverged"
        result.error = str(exc)
        print(f"[diverged] {name} seed {seed}: {exc}")
    except Exception as exc:  # noqa: BLE001 - one broken arm must not end the sweep
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()

    result.wall_clock_s = time.perf_counter() - started
    if result.wall_clock_s > 0 and result.tokens:
        result.tokens_per_sec = result.tokens / result.wall_clock_s
    return result


def _load_existing(results_path: Path) -> dict[str, ArmResult]:
    if not results_path.exists():
        return {}
    records = json.loads(results_path.read_text()).get("arms", [])
    out = {}
    for record in records:
        record.setdefault("history", [])
        arm = ArmResult(**record)
        out[arm.key] = arm
    return out


def _write(results_path: Path, arms: dict[str, ArmResult], meta: dict) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, "arms": [asdict(a) for a in arms.values()]}
    tmp = results_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(results_path)


def run_sweep(
    arms: list[str],
    out_dir: Path,
    results_path: Path,
    seed: int = 1337,
    baseline_seeds: int = 3,
    extra: list[str] | None = None,
    resume: bool = True,
) -> dict[str, ArmResult]:
    """Run every arm once, and the baseline ``baseline_seeds`` times.

    The repeated baseline is the whole basis for interpreting the results: it is the
    only way to know how much of a delta is the design change and how much is the
    seed.
    """
    extra = extra or []
    existing = _load_existing(results_path) if resume else {}
    results: dict[str, ArmResult] = dict(existing)

    # (config, seed) pairs. The baseline repeats; every other arm runs once, at the
    # same seed as the baseline's first run so the comparison is like-for-like.
    plan: list[tuple[Path, int]] = []
    for arm in arms:
        path = (CONFIG_ROOT / "ablations" / arm).resolve()
        if not path.exists():
            raise FileNotFoundError(f"ablation config not found: {path}")
        seeds = [seed + i for i in range(baseline_seeds)] if path.stem == "_base" else [seed]
        plan.extend((path, s) for s in seeds)

    meta = {
        "arms": arms,
        "seed": seed,
        "baseline_seeds": baseline_seeds,
        "overrides": extra,
        "out_dir": str(out_dir),
    }

    print(f"sweep: {len(plan)} runs ({len(arms)} arms, baseline x{baseline_seeds})")
    for index, (path, arm_seed) in enumerate(plan, start=1):
        key = f"{_arm_name(path)}@seed{arm_seed}"
        if key in results and results[key].status in ("completed", "diverged"):
            print(f"[{index}/{len(plan)}] {key}: already done ({results[key].status}), skipping")
            continue

        print(f"[{index}/{len(plan)}] {key}")
        result = run_arm(path, arm_seed, out_dir, extra)
        results[key] = result
        # Written after every arm, so a sweep killed mid-way keeps what it finished.
        _write(results_path, results, meta)

    # Counted over this invocation's plan, not every arm ever recorded in the file —
    # a resumed sweep of 2 arms should not report "15/2".
    planned = [f"{_arm_name(p)}@seed{s}" for p, s in plan]
    done = sum(1 for k in planned if k in results and results[k].status == "completed")
    diverged = sum(1 for k in planned if k in results and results[k].status == "diverged")
    print(
        f"\nsweep finished: {done}/{len(plan)} completed"
        + (f", {diverged} diverged" if diverged else "")
        + f" -> {results_path}"
    )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the ablation sweep.",
        epilog="Each arm varies exactly one design choice; the baseline repeats "
        "across seeds so deltas can be judged against run-to-run noise.",
    )
    parser.add_argument(
        "--arms",
        nargs="*",
        default=DEFAULT_ARMS,
        help="ablation config filenames under configs/ablations/",
    )
    parser.add_argument("--out-dir", type=str, default="out/ablations")
    parser.add_argument("--results", type=str, default="results/ablations.json")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--baseline-seeds",
        type=int,
        default=3,
        help="how many seeds to run the baseline with, to establish the noise floor",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="config override applied to every arm (repeatable)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="re-run arms even if they already have a recorded result",
    )
    args = parser.parse_args(argv)

    run_sweep(
        arms=args.arms,
        out_dir=Path(args.out_dir),
        results_path=Path(args.results),
        seed=args.seed,
        baseline_seeds=args.baseline_seeds,
        extra=args.overrides,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
