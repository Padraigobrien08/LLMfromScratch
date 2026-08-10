"""Tests for the multi-GPU scaling harness.

The measurement itself needs several GPUs, but everything that turns raw records into a
claim — warmup trimming, medians, efficiency, the loss-equivalence delta — is ordinary
arithmetic and is where a scaling report actually goes wrong. A harness that computes
efficiency against the wrong baseline, or takes a mean over the compile step, reports a
scaling cliff that does not exist. So that arithmetic is pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llmfs.bench.scaling import (
    ScalingPoint,
    ScalingReport,
    base_overrides,
    format_table,
    main,
    max_loss_delta,
    read_metrics,
    summarise,
    validate_overrides,
)


def write_metrics(path: Path, records: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def perf(step: int, tps: float, loss: float = 5.0, mfu: float = 0.4) -> dict:
    return {
        "step": step,
        "train/loss": loss,
        "perf/tokens_per_sec": tps,
        "perf/step_time_ms": 1000.0,
        "perf/mfu": mfu,
    }


def test_read_metrics_skips_eval_and_malformed_records(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(perf(1, 100.0)),
                json.dumps({"step": 1, "val/loss": 4.0}),  # eval record: no throughput
                "{not json",
                "",
                json.dumps(perf(2, 200.0)),
            ]
        )
    )
    records = read_metrics(path)
    assert [r["perf/tokens_per_sec"] for r in records] == [100.0, 200.0]


def test_read_metrics_missing_file_is_empty(tmp_path: Path) -> None:
    assert read_metrics(tmp_path / "absent.jsonl") == []


def test_summarise_discards_warmup() -> None:
    # Warmup steps are deliberately slow; including them would understate throughput.
    records = [perf(i, 10.0) for i in range(5)] + [perf(i, 100.0) for i in range(5, 10)]
    stats, _ = summarise(records, world_size=1, warmup=5)
    assert stats["tokens_per_sec"] == 100.0
    assert stats["samples"] == 5


def test_summarise_uses_median_not_mean() -> None:
    """One outlier must not move the number. A mean of these is 220; the median is 100."""
    records = [perf(i, 100.0) for i in range(5)] + [perf(5, 700.0)]
    stats, _ = summarise(records, world_size=1, warmup=0)
    assert stats["tokens_per_sec"] == 100.0


def test_summarise_reports_per_gpu_throughput() -> None:
    stats, _ = summarise([perf(i, 800.0) for i in range(4)], world_size=8, warmup=0)
    assert stats["tokens_per_sec"] == 800.0
    assert stats["tokens_per_sec_per_gpu"] == 100.0


def test_summarise_keeps_all_losses_including_warmup() -> None:
    """Throughput needs steady state; the loss-equivalence check needs every step,
    because a divergence between world sizes would show up first at step 1."""
    records = [perf(i, 10.0, loss=5.0 - i * 0.1) for i in range(10)]
    _, losses = summarise(records, world_size=1, warmup=8)
    assert len(losses) == 10


def test_summarise_survives_missing_mfu() -> None:
    records = [{"step": i, "perf/tokens_per_sec": 10.0, "perf/step_time_ms": 5.0} for i in range(3)]
    stats, losses = summarise(records, world_size=1, warmup=0)
    assert stats["mfu"] is None
    assert losses == []


def test_summarise_empty_is_empty() -> None:
    assert summarise([], world_size=1, warmup=10) == ({}, [])


def test_max_loss_delta_over_overlapping_prefix() -> None:
    assert max_loss_delta([1.0, 2.0, 3.0], [1.0, 2.5, 3.0]) == pytest.approx(0.5)
    # Unequal lengths compare only the prefix they share, rather than raising.
    assert max_loss_delta([1.0, 2.0], [1.0, 2.0, 99.0]) == pytest.approx(0.0)
    assert max_loss_delta([], [1.0]) is None


def test_format_table_marks_failures_without_inventing_numbers() -> None:
    report = ScalingReport(config="c", steps=10, warmup=2)
    report.points = [
        ScalingPoint(1, 100.0, 100.0, 10.0, 0.4, 32, 5, 5.0, 4.0),
        ScalingPoint(2, 0.0, 0.0, 0.0, None, 16, 0, None, None, error="OOM"),
    ]
    table = format_table(report)
    assert "**failed**" in table
    # The failed row must not contribute a speedup or efficiency figure.
    failed_row = [line for line in table.splitlines() if "failed" in line][0]
    assert "0.00×" not in failed_row and "0.0%" not in failed_row


def test_cli_rejects_warmup_that_leaves_no_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    """--steps 10 --warmup 10 would take a median of nothing. Fail before renting time,
    not thirty minutes in."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    with pytest.raises(SystemExit, match="must exceed"):
        main(["--steps", "10", "--warmup", "10", "--world-sizes", "1"])


def _capture_world_sizes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[int]:
    attempted: list[int] = []

    def fake_run(**kwargs):  # type: ignore[no-untyped-def]
        attempted.extend(kwargs["world_sizes"])
        return ScalingReport(config="c", steps=kwargs["steps"], warmup=kwargs["warmup"])

    monkeypatch.setattr("llmfs.bench.scaling.run", fake_run)
    return attempted


def test_cli_skips_world_sizes_the_gpu_box_cannot_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """On a 2-GPU box, the 4- and 8-GPU points must not be attempted — they would fail
    only after the earlier points had already been paid for."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    attempted = _capture_world_sizes(monkeypatch, tmp_path)
    main(
        ["--world-sizes", "1,2,4,8", "--steps", "3", "--warmup", "1",
         "--out", str(tmp_path / "s.json")]
    )
    assert attempted == [1, 2]


def test_cli_allows_cpu_multirank_as_a_correctness_check(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without CUDA, multi-rank must still run: gloo on CPU is how the distributed path
    gets exercised before any GPU is rented. It has to be labelled, though, because the
    throughput numbers from it are worthless."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    attempted = _capture_world_sizes(monkeypatch, tmp_path)
    out = tmp_path / "s.json"
    main(["--world-sizes", "1,2", "--steps", "3", "--warmup", "1", "--out", str(out)])

    assert attempted == [1, 2], "CPU ranks must not be silently dropped"
    notes = " ".join(json.loads(out.read_text())["notes"])
    assert "NO CUDA" in notes and "NOT" in notes, "the run must be labelled correctness-only"


def test_base_overrides_put_eval_interval_on_the_right_section() -> None:
    """`eval_interval` is a LogConfig field. Addressing it as `train.eval_interval` raises
    at config load, which killed every run of the first version of this harness — after
    launch, where the explanation sat in a buffered stdout while the GPU billed."""
    overrides = base_overrides("r", "out/x", steps=10)
    assert "log.eval_interval=11" in overrides
    assert not any(o.startswith("train.eval_interval") for o in overrides)


def test_base_overrides_disable_every_checkpoint_path() -> None:
    """A 13-arm sweep once filled a 100GB disk with milestone checkpoints and killed the
    job after it. Nothing in a throughput measurement needs a checkpoint."""
    overrides = base_overrides("r", "out/x", steps=10)
    assert "log.milestone_fracs=[]" in overrides
    assert "log.keep_last_n=0" in overrides
    assert any(o.startswith("log.checkpoint_interval=") for o in overrides)


def test_validate_overrides_accepts_the_real_config() -> None:
    validate_overrides("debug", (1,), [])


def test_validate_overrides_rejects_a_bad_key_before_launching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(SystemExit, match="rejected the scaling overrides"):
        validate_overrides("debug", (1,), ["train.no_such_field=3"])


def test_cli_validates_before_running_anything(monkeypatch: pytest.MonkeyPatch) -> None:
    """The guard has to fire before the first torchrun, not after it fails."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    launched: list[int] = []
    monkeypatch.setattr(
        "llmfs.bench.scaling.run",
        lambda **kw: launched.append(1) or ScalingReport(config="c", steps=1, warmup=0),
    )
    with pytest.raises(SystemExit, match="rejected the scaling overrides"):
        main(["--config", "debug", "--world-sizes", "1", "--steps", "3", "--warmup", "1",
              "--set", "optim.not_a_field=1"])
    assert launched == [], "nothing may be launched once the config is known to be bad"
