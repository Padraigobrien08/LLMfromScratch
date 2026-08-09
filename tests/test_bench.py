"""Benchmarks and provenance.

These run on CPU in CI, where the absolute numbers are meaningless — what is tested
is that the harness measures the right things and records what produced them.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from conftest import tiny_config
from llmfs.bench.throughput import bench_inference, bench_training, write_results
from llmfs.config import load_config
from llmfs.model import Transformer
from llmfs.utils.provenance import capture


def test_training_bench_covers_every_variant() -> None:
    cfg = load_config("debug")
    cfg.model = tiny_config(vocab_size=512, n_layer=2, n_embd=32, block_size=32)
    cfg.data.block_size = 32
    cfg.data.micro_batch_size = 2

    variants = [
        {"name": "baseline", "compile": False, "grad_checkpointing": False},
        {"name": "grad-checkpoint", "compile": False, "grad_checkpointing": True},
    ]
    results = bench_training(cfg, variants, steps=2, warmup=1, device=torch.device("cpu"))

    assert [r.variant for r in results] == ["baseline", "grad-checkpoint"]
    assert all(r.tokens_per_sec > 0 and r.ms_per_step > 0 for r in results)
    assert all(r.suite == "training" for r in results)
    # The settings are recorded, so a table row can be traced to what produced it.
    assert results[1].settings["grad_checkpointing"] is True


def test_inference_bench_reports_cache_and_naive() -> None:
    model = Transformer(tiny_config(vocab_size=256, block_size=64)).eval()
    results = bench_inference(
        model, prompt_len=8, gen_len=8, batch_sizes=(1,), device=torch.device("cpu")
    )
    variants = {r.variant for r in results}
    assert any("kv-cache" in v for v in variants)
    assert any("naive" in v for v in variants)

    cached = next(r for r in results if "kv-cache" in r.variant)
    assert cached.extra["time_to_first_token_ms"] > 0
    assert cached.extra["kv_cache_mib"] > 0
    assert cached.settings["use_cache"] is True


def test_naive_baseline_is_measured_only_at_batch_one() -> None:
    """Larger naive batches cost benchmark runtime without adding information."""
    model = Transformer(tiny_config(vocab_size=256, block_size=64)).eval()
    results = bench_inference(
        model, prompt_len=8, gen_len=6, batch_sizes=(1, 2), device=torch.device("cpu")
    )
    naive = [r for r in results if "naive" in r.variant]
    assert len(naive) == 1 and naive[0].settings["batch_size"] == 1


def test_provenance_records_what_produced_a_number() -> None:
    info = capture(torch.device("cpu"), measure=False)
    for key in ("timestamp_utc", "torch", "python", "platform", "device", "cpu_count"):
        assert key in info, key
    # git_dirty must be a real boolean: a result from an uncommitted tree cannot
    # honestly claim the recorded commit produced it.
    assert isinstance(info["git_dirty"], bool)


def test_results_file_carries_provenance(tmp_path: Path) -> None:
    model = Transformer(tiny_config(vocab_size=256, block_size=64)).eval()
    results = bench_inference(
        model,
        prompt_len=8,
        gen_len=4,
        batch_sizes=(1,),
        device=torch.device("cpu"),
        include_naive=False,
    )
    out = tmp_path / "b.json"
    write_results(results, out, torch.device("cpu"), extra={"note": "test"})

    payload = json.loads(out.read_text())
    assert payload["meta"]["note"] == "test"
    assert "torch" in payload["provenance"]
    assert len(payload["results"]) == len(results)
