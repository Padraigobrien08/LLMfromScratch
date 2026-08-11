"""Data pipeline: shard writing, sequential reading, and resumption determinism."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from llmfs.data.loader import ShardDataLoader, read_meta
from llmfs.data.prepare import ShardWriter


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Three train shards holding the sequence 0, 1, 2, ... so any token's identity
    reveals its position — which is what makes the ordering assertions below sharp."""
    out = tmp_path / "corpus"
    out.mkdir()
    total = 0
    for i, size in enumerate([500, 500, 300]):
        arr = np.arange(total, total + size, dtype=np.uint16)
        arr.tofile(out / f"train_{i:06d}.bin")
        total += size
    np.arange(50, dtype=np.uint16).tofile(out / "val_000000.bin")
    (out / "meta.json").write_text(
        json.dumps({"tokenizer": "gpt2", "vocab_size": 50257, "tokens": {"train": total}})
    )
    return out


def make_loader(corpus: Path, **kwargs) -> ShardDataLoader:
    params = dict(split="train", micro_batch_size=2, block_size=8)
    params.update(kwargs)
    return ShardDataLoader(corpus, **params)  # type: ignore[arg-type]


def test_reads_the_stream_in_order(corpus: Path) -> None:
    loader = make_loader(corpus)
    x, y = loader.next_batch()
    assert x.shape == (2, 8) and y.shape == (2, 8)
    # First micro-batch is tokens 0..15, laid out row-major.
    torch.testing.assert_close(x.flatten(), torch.arange(16))
    # y is x shifted by one — the next-token targets.
    torch.testing.assert_close(y.flatten(), torch.arange(1, 17))


def test_consecutive_batches_do_not_overlap(corpus: Path) -> None:
    loader = make_loader(corpus)
    first, _ = loader.next_batch()
    second, _ = loader.next_batch()
    assert first.max().item() < second.min().item()
    assert second.flatten()[0].item() == first.flatten()[-1].item() + 1


def test_reads_across_shard_boundaries(corpus: Path) -> None:
    """Shard joins must be invisible: the stream is contiguous across files."""
    loader = ShardDataLoader(corpus, "train", micro_batch_size=1, block_size=20)
    loader.position = 495  # straddles the 500-token boundary between shard 0 and 1
    x, _ = loader.next_batch()
    torch.testing.assert_close(x.flatten(), torch.arange(495, 515))


def test_wraps_at_the_end_of_the_corpus(corpus: Path) -> None:
    loader = ShardDataLoader(corpus, "train", micro_batch_size=1, block_size=10)
    loader.position = loader.total_tokens - 5
    x, _ = loader.next_batch()
    expected = torch.tensor([1295, 1296, 1297, 1298, 1299, 0, 1, 2, 3, 4])
    torch.testing.assert_close(x.flatten(), expected)
    assert loader.epoch == 1


def test_set_step_is_the_only_state_needed_to_resume(corpus: Path) -> None:
    """The resumption contract.

    A loader fast-forwarded with ``set_step`` must yield exactly what an
    uninterrupted loader would have yielded at that step — no data-loader state is
    checkpointed, so there is nothing that can go stale or disagree.
    """
    grad_accum = 3
    uninterrupted = make_loader(corpus)
    for _ in range(5 * grad_accum):
        uninterrupted.next_batch()
    expected, _ = uninterrupted.next_batch()

    resumed = make_loader(corpus)
    resumed.set_step(5, grad_accum_steps=grad_accum)
    actual, _ = resumed.next_batch()

    torch.testing.assert_close(actual, expected)


def test_ranks_read_disjoint_slices(corpus: Path) -> None:
    """No two ranks may train on the same tokens in a step — that would silently
    double-count gradients and make the effective batch smaller than reported."""
    world = 4
    loaders = [make_loader(corpus, rank=r, world_size=world) for r in range(world)]
    batches = [loader.next_batch()[0].flatten().tolist() for loader in loaders]

    seen = [t for batch in batches for t in batch]
    assert len(seen) == len(set(seen)), "ranks overlapped"
    # Their union is the same contiguous span a single-GPU run would have consumed.
    assert sorted(seen) == list(range(world * 2 * 8))


def test_distributed_step_matches_single_gpu_token_span(corpus: Path) -> None:
    """Scaling the world size must not change which tokens a step consumes."""
    world = 2
    loaders = [make_loader(corpus, rank=r, world_size=world) for r in range(world)]
    distributed: list[int] = []
    for _ in range(2):
        for loader in loaders:
            distributed += loader.next_batch()[0].flatten().tolist()

    single = make_loader(corpus)
    solo: list[int] = []
    for _ in range(4):
        solo += single.next_batch()[0].flatten().tolist()

    assert sorted(distributed) == sorted(solo)


def test_missing_shards_raise_a_useful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="llmfs-prepare-data"):
        ShardDataLoader(tmp_path, "train", 2, 8)


def test_corpus_too_small_is_rejected(tmp_path: Path) -> None:
    np.arange(10, dtype=np.uint16).tofile(tmp_path / "train_000000.bin")
    with pytest.raises(ValueError, match="too few"):
        ShardDataLoader(tmp_path, "train", micro_batch_size=4, block_size=64)


def test_read_meta(corpus: Path) -> None:
    assert read_meta(corpus)["tokenizer"] == "gpt2"
    with pytest.raises(FileNotFoundError, match="meta.json"):
        read_meta(corpus.parent)


def test_shard_writer_splits_at_the_configured_size(tmp_path: Path) -> None:
    writer = ShardWriter(tmp_path, shard_tokens=100, val_shards=1)
    writer.add(np.arange(250, dtype=np.uint16))
    writer.close()

    # Shard 0 goes to val, the rest to train; the final shard holds the remainder.
    assert [m["split"] for m in writer.manifest] == ["val", "train", "train"]
    assert [m["tokens"] for m in writer.manifest] == [100, 100, 50]
    assert (tmp_path / "val_000000.bin").exists()
    assert (tmp_path / "train_000001.bin").exists()


def test_shard_writer_round_trips_token_values(tmp_path: Path) -> None:
    tokens = np.random.randint(0, 50257, size=333, dtype=np.uint16)
    writer = ShardWriter(tmp_path, shard_tokens=64, val_shards=0)
    writer.add(tokens)
    writer.close()

    recovered = np.concatenate(
        [np.fromfile(tmp_path / m["path"], dtype=np.uint16) for m in writer.manifest]
    )
    np.testing.assert_array_equal(recovered, tokens)


def test_small_limit_docs_would_have_produced_no_training_data() -> None:
    """The trap this guards: shard 0 is the validation split, so a corpus smaller than one
    shard becomes entirely validation. Measured on a real pod — `--limit-docs 40000` gave
    41.8M tokens in one partial shard and "0 train / 41,834,799 val", after paying ten
    minutes to tokenise it."""
    from llmfs.data.prepare import _assert_trainable

    meta = {"tokens": {"train": 0, "val": 41_834_799}}
    with pytest.raises(SystemExit) as excinfo:
        _assert_trainable(meta, Path("/tmp/x"), shard_tokens=100_000_000)
    message = str(excinfo.value)
    assert "0 training tokens" in message
    # Must carry the remedy, with a shard size that would actually work.
    assert "--shard-tokens" in message
    assert "4,183,479" in message


def test_assert_trainable_passes_when_train_tokens_exist() -> None:
    from llmfs.data.prepare import _assert_trainable

    _assert_trainable({"tokens": {"train": 1, "val": 1}}, Path("/tmp/x"), shard_tokens=10)
