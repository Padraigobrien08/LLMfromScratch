""":command:`llmfs-scaling` — measure what adding GPUs actually buys.

Runs the **real trainer** at each world size rather than a synthetic loop. That choice is
the point of this module: a hand-written benchmark loop would measure a program nobody
trains with, and would quietly omit the two things most likely to spoil scaling — the
gradient all-reduce and the optimiser step. Here each data point is
``torchrun --nproc_per_node=N -m llmfs.train.cli`` against the same config the
reproduction used, so what is reported is what a real run would get.

Two questions, and they are separate:

* **Does throughput scale?** Reported as scaling efficiency, ``(tokens_sec_N / N) /
  tokens_sec_1``. Perfect scaling is 1.0; the interesting part is where it falls off and
  why.
* **Is it still the same optimisation?** ``tokens_per_step`` is fixed in tokens, and
  gradient accumulation is derived from it and the world size, so eight GPUs should take
  the *same optimisation steps* as one — only faster. That is a claim about correctness,
  not speed, and it is checked by comparing loss curves across world sizes rather than
  asserted in prose.

The second question is the one that would embarrass a scaling report if it were wrong,
which is why it is measured first and reported alongside every throughput number.
"""

from __future__ import annotations

import argparse
import json
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# Checkpointing is disabled for every scaling run. Nothing here needs a checkpoint, and an
# earlier sweep filled a 100GB disk with per-run milestones and killed the job that
# followed it — so this is an explicit override rather than a default to be trusted.
_NO_CHECKPOINTS = [
    "log.milestone_fracs=[]",
    "log.keep_last_n=0",
    "log.checkpoint_interval=100000000",
    "log.tensorboard=false",
]


def base_overrides(run_name: str, out_dir: Path | str, steps: int) -> list[str]:
    """Config overrides every scaling run shares.

    ``eval_interval`` is on ``log``, not ``train`` — putting it on the wrong section is
    not a silent no-op, it raises at config load, which is how the first version of this
    harness managed to fail every single run before the trainer even started.
    """
    return [
        f"train.max_steps={steps}",
        f"log.run_name={run_name}",
        f"log.out_dir={out_dir}",
        "log.log_interval=1",  # one record per step; a 20-step run needs the resolution
        f"log.eval_interval={steps + 1}",  # no eval pauses inside the timed region
        *_NO_CHECKPOINTS,
    ]


def validate_overrides(config: str, world_sizes: tuple[int, ...], extra: list[str]) -> None:
    """Load the config once, locally, before anything is launched.

    A mistyped override key is fatal at config load. Discovering that from the *first*
    torchrun means every rank fails instantly, the harness records "run produced no
    metrics", and — if stdout happens to be redirected — the explanation sits in a buffer
    while the rental keeps billing. So the config is resolved here first, where the
    traceback is immediate and free.
    """
    from ..config import load_config

    probe = base_overrides("scaling-validate", "out/scaling", max(2, len(world_sizes) + 1))
    try:
        cfg = load_config(config, probe + extra)
    except Exception as exc:  # noqa: BLE001 - re-raised as a clean message
        raise SystemExit(f"config '{config}' rejected the scaling overrides: {exc}") from exc

    check_divisibility(cfg, world_sizes)


def check_divisibility(cfg: Any, world_sizes: tuple[int, ...]) -> None:
    """Every requested world size must divide the effective batch exactly.

    ``grad_accum_steps`` demands that ``tokens_per_step`` be divisible by
    ``micro_batch x block_size x world_size``, and a world size that fails it raises
    inside the trainer — one rank at a time, after torchrun has launched, on a rented
    machine. Non-powers-of-two are where this bites: 524,288 tokens is 512 sequences, and
    512 has no factor of 7, so a 7-GPU box cannot run the default batch at all and no
    choice of micro_batch changes that.

    So it is checked here, before anything launches, with the arithmetic needed to fix it.
    """
    from math import lcm

    bad = []
    for ws in world_sizes:
        try:
            cfg.grad_accum_steps(ws)
        except ValueError:
            bad.append(ws)
    if not bad:
        return

    unit = cfg.data.micro_batch_size * cfg.data.block_size * lcm(*world_sizes)
    workable = (cfg.train.tokens_per_step // unit) * unit
    # Underscore separators, because the suggestion is meant to be pasted onto a command
    # line where a comma would be read as an argument separator.
    detail = (
        f"try --set train.tokens_per_step={workable:_} "
        f"({workable // cfg.data.block_size} sequences), which divides evenly for every "
        f"requested world size"
        if workable
        else "no smaller multiple exists; drop the offending world sizes or lower micro_batch"
    )

    raise SystemExit(
        f"world size(s) {bad} cannot divide the effective batch: tokens_per_step="
        f"{cfg.train.tokens_per_step:,} is not divisible by micro_batch "
        f"({cfg.data.micro_batch_size}) x block_size ({cfg.data.block_size}) x world_size. "
        f"{detail}. Hold whatever you choose constant across every point, or the "
        f"comparison measures two different optimisations."
    )


@dataclass
class ScalingPoint:
    world_size: int
    tokens_per_sec: float
    """Global throughput: tokens across all ranks, not per GPU."""
    tokens_per_sec_per_gpu: float
    step_time_ms: float
    mfu: float | None
    grad_accum_steps: int | None
    samples: int
    """How many logged steps went into the medians, after discarding warmup."""
    loss_first: float | None
    loss_last: float | None
    max_loss_delta_vs_1gpu: float | None = None
    """Largest absolute difference against the 1-GPU loss curve, step for step."""
    speedup: float = 1.0
    efficiency: float = 1.0
    error: str | None = None
    seconds: float = 0.0
    tokens_per_step: int | None = None
    """The effective batch in tokens, so the artifact states it rather than implying it
    through grad_accum_steps times a micro-step size someone has to know."""


@dataclass
class ScalingReport:
    config: str
    steps: int
    warmup: int
    label: str = ""
    """Names this measurement, e.g. "a100x8-nvlink". Two runs on different hardware are
    the whole point of comparing interconnects, and they must not overwrite each other."""
    points: list[ScalingPoint] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)
    topology: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def capture_topology() -> dict[str, Any]:
    """Record how the GPUs are actually wired together.

    Scaling efficiency is largely an interconnect story, so "8 GPUs" is not a sufficient
    description of the hardware — 8 cards on NVLink and 8 on PCIe are different machines
    for this purpose. ``nvidia-smi topo -m`` is the evidence, kept raw so the claim can be
    checked rather than taken on trust, plus a derived flag for the summary table.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    if proc.returncode != 0:
        return {}

    matrix = proc.stdout
    # NV1/NV2/... in the matrix means a direct NVLink between that pair. "PIX"/"PHB"/"SYS"
    # are PCIe routes of decreasing quality.
    has_nvlink = any(f"NV{n}" in matrix for n in range(1, 19))
    return {
        "nvidia_smi_topo_m": matrix,
        "has_nvlink": has_nvlink,
        "interconnect": "NVLink" if has_nvlink else "PCIe",
    }


def read_metrics(path: Path) -> list[dict[str, Any]]:
    """Parse a run's ``metrics.jsonl``, keeping only training-step records."""
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Eval records carry no throughput, and including them would drag the medians.
        if "perf/tokens_per_sec" in record:
            records.append(record)
    return records


def summarise(
    records: list[dict[str, Any]], world_size: int, warmup: int
) -> tuple[dict[str, Any], list[float]]:
    """Reduce a run's records to steady-state medians, plus its loss curve.

    Medians rather than means: the first steps after warmup still catch the occasional
    allocator or autotune hiccup, and one 3x outlier moves a mean of twenty samples
    enough to invent a scaling cliff that is not there.
    """
    steady = records[warmup:] if len(records) > warmup else records
    if not steady:
        return {}, []

    throughput = statistics.median(r["perf/tokens_per_sec"] for r in steady)
    step_ms = statistics.median(r["perf/step_time_ms"] for r in steady)
    mfus = [r["perf/mfu"] for r in steady if r.get("perf/mfu") is not None]
    losses = [r["train/loss"] for r in records if "train/loss" in r]

    return (
        {
            "tokens_per_sec": throughput,
            "tokens_per_sec_per_gpu": throughput / world_size,
            "step_time_ms": step_ms,
            "mfu": statistics.median(mfus) if mfus else None,
            "samples": len(steady),
            "loss_first": losses[0] if losses else None,
            "loss_last": losses[-1] if losses else None,
        },
        losses,
    )


def max_loss_delta(baseline: list[float], other: list[float]) -> float | None:
    """Largest step-for-step absolute loss difference over the overlapping prefix.

    Not bitwise equality: summing gradients in a different order across a different
    number of ranks perturbs the last bits, and that perturbation compounds a little
    over steps. What would signal a real bug is a *drift* — a delta that grows with the
    step count — rather than a small constant one.
    """
    n = min(len(baseline), len(other))
    if n == 0:
        return None
    return max(abs(a - b) for a, b in zip(baseline[:n], other[:n], strict=True))


def _free_port() -> int:
    """Ask the OS for a port that is free right now.

    Racy in principle — the port could be taken between closing this socket and torchrun
    binding it — but the alternative is a fixed port, which collides with any previous run
    still shutting down and fails the same way every time instead of almost never.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_one(
    world_size: int,
    config: str,
    steps: int,
    out_dir: Path,
    extra_overrides: list[str],
    python: str = sys.executable,
) -> tuple[list[dict[str, Any]], str | None]:
    """Train ``steps`` optimiser steps at ``world_size`` and return its metric records."""
    run_name = f"scaling-ws{world_size}"
    run_dir = out_dir / run_name

    # metrics.jsonl is opened in append mode by the trainer, so a stale directory from an
    # earlier attempt would silently blend two runs' throughputs into one median.
    if run_dir.exists():
        shutil.rmtree(run_dir)

    overrides = base_overrides(run_name, out_dir, steps) + extra_overrides

    # Explicit IPv4 rendezvous rather than --standalone. --standalone resolves "localhost",
    # and on a machine whose IPv6 reverse lookup for the loopback address fails (macOS,
    # commonly) c10d retries that resolution forever: torchrun hangs before the trainer
    # starts, printing only gai errors. Pinning 127.0.0.1 with a port we know is free
    # skips name resolution altogether, and costs nothing on a Linux GPU box.
    cmd = [
        python,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc_per_node={world_size}",
        "--master_addr=127.0.0.1",
        f"--master_port={_free_port()}",
        "-m",
        "llmfs.train.cli",
        "--config",
        config,
    ]
    for override in overrides:
        cmd += ["--set", override]

    print(f"\n=== world_size={world_size} ===", flush=True)
    print("  " + " ".join(cmd[:8]) + " ...", flush=True)
    completed = subprocess.run(cmd, capture_output=True, text=True)

    records = read_metrics(run_dir / "metrics.jsonl")
    # A non-zero exit is a failure even when rank 0 got some metrics down first. The
    # earlier form required *both* a bad exit and an empty log, so a torchrun that
    # aborted at step 12 of 30 — one rank OOMing, a NCCL timeout — published a median
    # over whatever rank 0 managed to log, indistinguishable in the artifact from a
    # clean run except by a `samples` field nobody asserted. Only rank 0 writes
    # metrics.jsonl at all, so partial records say nothing about the other ranks.
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip().splitlines()[-6:]
        return [], (
            f"torchrun exited {completed.returncode} "
            f"({len(records)} step records logged): " + " | ".join(tail)
        )
    if not records:
        return [], "run produced no metrics"
    # And a clean exit must still account for every step: base_overrides pins
    # log_interval=1, so the last record's step number is the number of steps that ran.
    if records[-1].get("step") != steps:
        return [], (
            f"run logged through step {records[-1].get('step')} of {steps} — incomplete, "
            f"so its median would describe a shorter run than the artifact claims"
        )
    return records, None


def tokens_per_step_for(config: str, extra_overrides: list[str]) -> int | None:
    """The effective batch in tokens, recorded so the artifact needs no outside help.

    The comm-accum artifacts record only ``grad_accum_steps``, which left the exporter
    multiplying by a hand-typed micro-step size to state tokens/step on the site. New
    artifacts carry the number itself.
    """
    try:
        from ..config import load_config

        return load_config(config, extra_overrides).train.tokens_per_step
    except Exception:  # noqa: BLE001 - a missing config must not sink the benchmark
        return None


def grad_accum_for(config: str, world_size: int, extra_overrides: list[str]) -> int | None:
    """Ask the config system what accumulation this world size implies.

    Reported because it is the mechanism behind the equivalence claim: the product
    ``micro_batch x grad_accum x world_size`` is held constant, so it is worth showing
    that the middle term really did change.
    """
    try:
        from ..config import load_config

        cfg = load_config(config, extra_overrides)
        return cfg.grad_accum_steps(world_size)
    except Exception:  # noqa: BLE001 - a missing config must not sink the benchmark
        return None


def run(
    config: str = "gpt2-124m",
    world_sizes: tuple[int, ...] = (1, 2, 4, 8),
    steps: int = 30,
    warmup: int = 10,
    out_dir: Path = Path("out/scaling"),
    extra_overrides: list[str] | None = None,
    label: str = "",
) -> ScalingReport:
    extra_overrides = extra_overrides or []
    report = ScalingReport(config=config, steps=steps, warmup=warmup, label=label)
    report.topology = capture_topology()

    # capture() takes a device and calls torch.device() on it, so passing None raised a
    # TypeError that an over-broad `except` then turned into an empty provenance dict. The
    # 8x 5090 run shipped with "provenance": {} — no commit, no torch version, no GPU name
    # — in a repository whose central claim is that every result traces to a commit. A
    # failure here is now recorded and printed rather than swallowed.
    try:
        import torch

        from ..utils.provenance import capture

        device = "cuda" if torch.cuda.is_available() else "cpu"
        report.provenance = capture(device, measure=False)
        report.provenance["gpu_count"] = torch.cuda.device_count() if device == "cuda" else 0
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        report.provenance = {"error": f"{type(exc).__name__}: {exc}"}
        print(f"[warn] provenance capture FAILED: {exc}", flush=True)

    baseline_losses: list[float] = []
    baseline_throughput: float | None = None

    for world_size in world_sizes:
        started = time.perf_counter()
        records, error = run_one(world_size, config, steps, out_dir, extra_overrides)
        elapsed = time.perf_counter() - started

        if error:
            print(f"  FAILED: {error}", flush=True)
            report.points.append(
                ScalingPoint(
                    world_size=world_size,
                    tokens_per_sec=0.0,
                    tokens_per_sec_per_gpu=0.0,
                    step_time_ms=0.0,
                    mfu=None,
                    grad_accum_steps=grad_accum_for(config, world_size, extra_overrides),
                    samples=0,
                    loss_first=None,
                    loss_last=None,
                    error=error,
                    seconds=elapsed,
                )
            )
            continue

        stats, losses = summarise(records, world_size, warmup)
        point = ScalingPoint(
            world_size=world_size,
            grad_accum_steps=grad_accum_for(config, world_size, extra_overrides),
            tokens_per_step=tokens_per_step_for(config, extra_overrides),
            seconds=elapsed,
            **stats,
        )

        if world_size == min(world_sizes):
            baseline_losses, baseline_throughput = losses, point.tokens_per_sec
        else:
            point.max_loss_delta_vs_1gpu = max_loss_delta(baseline_losses, losses)

        if baseline_throughput:
            base_ws = min(world_sizes)
            point.speedup = point.tokens_per_sec / baseline_throughput
            point.efficiency = point.speedup / (world_size / base_ws)

        report.points.append(point)
        mfu = f", mfu {point.mfu:.1%}" if point.mfu else ""
        print(
            f"  {point.tokens_per_sec:,.0f} tok/s total "
            f"({point.tokens_per_sec_per_gpu:,.0f}/gpu), "
            f"{point.speedup:.2f}x, efficiency {point.efficiency:.1%}{mfu}",
            flush=True,
        )
        if point.max_loss_delta_vs_1gpu is not None:
            print(
                f"  max |loss delta| vs {min(world_sizes)} GPU: {point.max_loss_delta_vs_1gpu:.2e}",
                flush=True,
            )

    return report


def format_table(report: ScalingReport) -> str:
    """Markdown table, ready to paste into the scaling doc."""
    lines = [
        "| GPUs | grad accum | tokens/sec | per GPU | speedup | efficiency | MFU | max Δloss |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for p in report.points:
        if p.error:
            lines.append(f"| {p.world_size} | — | **failed** | — | — | — | — | — |")
            continue
        mfu = f"{p.mfu:.1%}" if p.mfu else "—"
        delta = (
            "baseline" if p.max_loss_delta_vs_1gpu is None else f"{p.max_loss_delta_vs_1gpu:.1e}"
        )
        lines.append(
            f"| {p.world_size} | {p.grad_accum_steps or '—'} | {p.tokens_per_sec:,.0f} | "
            f"{p.tokens_per_sec_per_gpu:,.0f} | {p.speedup:.2f}× | {p.efficiency:.1%} | "
            f"{mfu} | {delta} |"
        )
    return "\n".join(lines)


def plot(report: ScalingReport, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    good = [p for p in report.points if not p.error]
    if len(good) < 2:
        return False

    sizes = [p.world_size for p in good]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(sizes, [p.tokens_per_sec for p in good], "o-", label="measured")
    ideal = [good[0].tokens_per_sec * (s / good[0].world_size) for s in sizes]
    ax1.plot(sizes, ideal, "k--", alpha=0.5, label="linear")
    ax1.set_xlabel("GPUs")
    ax1.set_ylabel("tokens/sec")
    ax1.set_title("Throughput")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(sizes)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(sizes, [p.efficiency * 100 for p in good], "o-", color="tab:orange")
    ax2.axhline(100, color="k", ls="--", alpha=0.5)
    ax2.set_xlabel("GPUs")
    ax2.set_ylabel("scaling efficiency (%)")
    ax2.set_title("Efficiency")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(sizes)
    ax2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.set_ylim(0, 110)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Measure multi-GPU scaling.")
    parser.add_argument("--config", type=str, default="gpt2-124m")
    parser.add_argument(
        "--world-sizes",
        type=str,
        default="1,2,4,8",
        help="comma-separated GPU counts; sizes above the visible device count are skipped",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="logged steps to discard before taking medians (compile and autotune)",
    )
    parser.add_argument("--out-dir", type=str, default="out/scaling")
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help='names this measurement, e.g. "a100x8-nvlink"; keeps two runs from colliding',
    )
    parser.add_argument("--out", type=str, default="results/scaling.json")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra config override passed to every run (repeatable)",
    )
    args = parser.parse_args(argv)

    requested = tuple(int(s) for s in args.world_sizes.split(",") if s.strip())

    try:
        import torch

        available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    except ImportError:
        available = 0

    cpu_only = available == 0
    if cpu_only:
        # Multi-rank on CPU runs over gloo. Throughput means nothing there — the ranks
        # contend for the same cores — but it exercises DDP, no_sync, the rank-aware
        # loader and the world-size-derived accumulation, which is the cheapest possible
        # way to find out the distributed path is broken *before* renting eight GPUs to
        # discover it. So this is allowed rather than skipped, and labelled.
        #
        # Caveat, measured rather than assumed: this works on Linux, and multi-rank gloo
        # does NOT come up on macOS. Rank 0 starts and the group never forms, with or
        # without GLOO_SOCKET_IFNAME=lo0. Single-rank is fine everywhere. So on a Mac,
        # use this for the 1-GPU path and validate multi-rank on the Linux box.
        world_sizes, skipped = requested, []
        print("[cpu] no CUDA visible — running over gloo as a CORRECTNESS check.")
        print("[cpu] throughput and efficiency figures from this run are meaningless.")
        if sys.platform == "darwin" and max(requested) > 1:
            print("[cpu] macOS: multi-rank gloo does not form a process group here.")
            print("[cpu] expect world_size>1 to hang; run the multi-rank check on Linux.")
    else:
        # Ask the machine rather than trusting the flag: a request for 8 GPUs on a 2-GPU
        # box would otherwise fail after already paying for the earlier points.
        world_sizes = tuple(w for w in requested if w <= available)
        skipped = [w for w in requested if w not in world_sizes]
        if skipped:
            print(f"[skip] {skipped} — only {available} device(s) visible")

    if not world_sizes:
        raise SystemExit(f"no runnable world sizes: requested {requested}, {available} available")

    if args.steps <= args.warmup:
        raise SystemExit(
            f"--steps ({args.steps}) must exceed --warmup ({args.warmup}), "
            "or there is nothing left to take a median over"
        )

    validate_overrides(args.config, world_sizes, args.overrides)

    report = run(
        config=args.config,
        world_sizes=world_sizes,
        steps=args.steps,
        warmup=args.warmup,
        out_dir=Path(args.out_dir),
        extra_overrides=args.overrides,
        label=args.label,
    )
    if skipped:
        report.notes.append(f"skipped world sizes {skipped}: only {available} device(s) visible")
    if cpu_only:
        report.notes.append(
            "NO CUDA: ran over gloo on CPU. This validates the distributed code path "
            "(DDP, no_sync, rank-aware loading, world-size-derived accumulation) and the "
            "loss-equivalence claim. Throughput, per-GPU and efficiency figures are NOT "
            "meaningful — the ranks share cores."
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(report), indent=2))

    print("\n" + format_table(report))
    if plot(report, out.with_suffix(".png")):
        print(f"\nwrote {out} and {out.with_suffix('.png')}")
    else:
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()


# --------------------------------------------------------------- communication sweep


def comm_table(reports: list[dict[str, Any]], pivot_world_size: int = 8) -> str:
    """Merge the communication sweep's per-batch reports into one table.

    Each report is a normal scaling report at one ``tokens_per_step``, run at world size 1
    and ``pivot_world_size``. What the sweep is for is the relationship between the
    accumulation count at the pivot world size and the efficiency there — i.e. whether
    amortising the all-reduce over more compute is what buys the efficiency.

    Efficiency comes from each report's own single-GPU baseline, never from another
    report's: single-GPU throughput itself varies a little with the batch, so borrowing a
    baseline across batch sizes would divide by the wrong number.
    """
    rows = []
    for report in reports:
        points = {p["world_size"]: p for p in report.get("points", [])}
        pivot, base = points.get(pivot_world_size), points.get(1)
        if not pivot or pivot.get("error"):
            continue
        accum = pivot.get("grad_accum_steps")
        rows.append(
            {
                "accum": accum,
                "tokens_per_step": (accum or 0) * 131_072 if accum else None,
                "pivot_tps": pivot["tokens_per_sec"],
                "per_gpu": pivot["tokens_per_sec_per_gpu"],
                "base_tps": base["tokens_per_sec"] if base and not base.get("error") else None,
                "efficiency": pivot.get("efficiency"),
                "delta": pivot.get("max_loss_delta_vs_1gpu"),
            }
        )
    rows.sort(key=lambda r: -(r["accum"] or 0))

    lines = [
        f"| accum @ {pivot_world_size} GPUs | tokens/step | 1 GPU tok/s | "
        f"{pivot_world_size} GPU tok/s | per GPU | efficiency | max Δloss |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        base = f"{r['base_tps']:,.0f}" if r["base_tps"] else "—"
        eff = f"{r['efficiency']:.1%}" if r["efficiency"] is not None else "—"
        delta = f"{r['delta']:.1e}" if r["delta"] is not None else "—"
        lines.append(
            f"| {r['accum']} | {r['tokens_per_step']:,} | {base} | "
            f"{r['pivot_tps']:,.0f} | {r['per_gpu']:,.0f} | {eff} | {delta} |"
        )
    return "\n".join(lines)


def main_comm_report(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Merge a communication sweep into a table.")
    parser.add_argument("reports", nargs="+", help="comm-accum*.json files")
    parser.add_argument("--pivot-world-size", type=int, default=8)
    parser.add_argument("--plot", type=str, default=None, help="write a PNG here")
    args = parser.parse_args(argv)

    loaded = [json.loads(Path(p).read_text()) for p in args.reports]
    print(comm_table(loaded, pivot_world_size=args.pivot_world_size))
    if args.plot and plot_comm_sweep(loaded, Path(args.plot), args.pivot_world_size):
        print(f"\nwrote {args.plot}")


def fit_amortisation(
    losses: dict[int, float], using: tuple[int, int] = (8, 4)
) -> tuple[float, float] | None:
    """Fit ``loss = a + b/accum`` to exactly two accumulation points.

    Deliberately two, not least-squares over all of them. The point of the model is that it
    was fitted to the accum 8 and 4 measurements and then *predicted* accum 2 and 1 before
    those existed. Fitting to everything would destroy that: a curve through four points is
    a description, whereas a curve through two that lands on the other two is a test.
    """
    hi, lo = using
    if hi not in losses or lo not in losses:
        return None
    b = (losses[lo] - losses[hi]) / (1 / lo - 1 / hi)
    return losses[hi] - b / hi, b


def plot_comm_sweep(reports: list[dict[str, Any]], path: Path, pivot_world_size: int = 8) -> bool:
    """Efficiency against gradient accumulation, with the fit and what it predicted.

    The left panel is the finding; the right decomposes it. Points used to fit the model are
    drawn differently from the ones it predicted, because which is which is the whole
    argument — anyone can draw a curve through their data afterwards.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    eff: dict[int, float] = {}
    for report in reports:
        pivot = next(
            (p for p in report.get("points", []) if p["world_size"] == pivot_world_size), None
        )
        if pivot and not pivot.get("error") and pivot.get("grad_accum_steps"):
            eff[pivot["grad_accum_steps"]] = pivot["efficiency"] * 100
    if len(eff) < 3:
        return False

    losses = {a: 100 - e for a, e in eff.items()}
    fit = fit_amortisation(losses)
    accums = sorted(eff)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.4))

    fitted_on, predicted = (8, 4), [a for a in accums if a not in (8, 4)]
    ax1.plot(
        [a for a in accums if a in fitted_on],
        [eff[a] for a in accums if a in fitted_on],
        "o",
        ms=9,
        color="tab:blue",
        label="measured (used to fit)",
        zorder=3,
    )
    ax1.plot(
        predicted,
        [eff[a] for a in predicted],
        "s",
        ms=9,
        color="tab:red",
        label="measured (predicted first)",
        zorder=3,
    )
    if fit:
        a0, b = fit
        xs = [accums[0] * (accums[-1] / accums[0]) ** (i / 100) for i in range(101)]
        ax1.plot(
            xs,
            [100 - (a0 + b / x) for x in xs],
            "--",
            color="0.4",
            label=f"$100-(a+b/k)$,  a={a0:.2f}, b={b:.2f}",
        )
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(accums)
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xlabel(
        "gradient accumulation at 8 GPUs\n(all-reduces amortised over this many micro-batches)"
    )
    ax1.set_ylabel("scaling efficiency (%)")
    ax1.set_title("Communication is what accumulation hides")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc="lower right")

    if fit:
        a0, b = fit
        ax2.bar(
            [str(a) for a in accums], [a0] * len(accums), color="0.6", label=f"fixed: {a0:.2f} pts"
        )
        ax2.bar(
            [str(a) for a in accums],
            [losses[a] - a0 for a in accums],
            bottom=[a0] * len(accums),
            color="tab:orange",
            label="per-all-reduce, /accum",
        )
        ax2.set_ylabel("efficiency lost (percentage points)")
        ax2.set_xlabel("gradient accumulation at 8 GPUs")
        ax2.set_title("Where the loss comes from")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True
