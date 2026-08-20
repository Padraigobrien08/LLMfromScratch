"""Turn a corpus into tokenised ``uint16`` shards.

Two sources:

``fineweb-edu``
    The reproduction corpus. Streams ``HuggingFaceFW/fineweb-edu`` sample-10BT and
    tokenises it across all cores. Shard 0 of the train split is held out as
    validation, matching the convention the target number was measured under.
``text``
    Any local text file. This exists so the pipeline is runnable — and the debug
    config trainable — without a 10B-token download, which keeps the smoke test
    honest rather than skipped.

Output layout::

    data_dir/
      meta.json          tokenizer, vocab size, shard sizes, token total
      train_000000.bin   raw uint16 token ids
      train_000001.bin
      val_000000.bin
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .tokenizer import load_tokenizer

_TOKENIZER_SPEC: str | None = None
_TOKENIZER = None


def _init_worker(spec: str) -> None:
    """Load the tokenizer once per worker process rather than once per document."""
    global _TOKENIZER_SPEC, _TOKENIZER
    _TOKENIZER_SPEC = spec
    _TOKENIZER = load_tokenizer(spec)


def _tokenize_document(text: str) -> np.ndarray:
    assert _TOKENIZER is not None
    # Every document is prefixed with the end-of-text token, so the model gets an
    # explicit document boundary and never learns to continue one document into
    # the next across a shard join.
    ids = _TOKENIZER.encode(text, add_eot=True)
    arr = np.array(ids, dtype=np.uint32)
    if (arr >= 2**16).any():
        raise ValueError(
            f"token id {arr.max()} exceeds the uint16 range: shards store ids as uint16, "
            f"so the tokenizer's vocabulary must be under 65,536. Use a smaller "
            f"vocabulary (the gpt2 tokenizer is 50,257) or widen the shard dtype."
        )
    return arr.astype(np.uint16)


def check_vocab_fits_shards(tokenizer_spec: str) -> None:
    """Reject an oversized vocabulary before any tokenisation starts.

    Shards store ids as uint16. The per-document guard in ``_tokenize_document`` is a
    backstop, but on its own it only fires when a document happens to contain a high
    id — which for a large-vocabulary tokenizer might be hours into a run over 10B
    tokens. Checking the vocabulary up front turns that into an immediate error.
    """
    tokenizer = load_tokenizer(tokenizer_spec)
    if tokenizer.vocab_size > 2**16:
        raise ValueError(
            f"tokenizer {tokenizer_spec!r} has a vocabulary of {tokenizer.vocab_size:,}, "
            f"which does not fit the uint16 shard format (max 65,536). Use a smaller "
            f"vocabulary (the gpt2 tokenizer is 50,257) or widen the shard dtype."
        )


class ShardWriter:
    """Accumulates tokens and flushes fixed-size shards to disk."""

    def __init__(self, out_dir: Path, shard_tokens: int, val_shards: int = 1) -> None:
        self.out_dir = out_dir
        self.shard_tokens = shard_tokens
        self.val_shards = val_shards
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = np.empty(shard_tokens, dtype=np.uint16)
        self.fill = 0
        self.shard_index = 0
        self.manifest: list[dict] = []

    def _split_for(self, index: int) -> str:
        return "val" if index < self.val_shards else "train"

    def _flush(self, size: int) -> None:
        if size == 0:
            return
        split = self._split_for(self.shard_index)
        # Index within the split, so filenames stay contiguous per split.
        within = sum(1 for m in self.manifest if m["split"] == split)
        path = self.out_dir / f"{split}_{within:06d}.bin"
        # Temp-and-rename, the same discipline checkpoints use: a preparation killed
        # mid-write must not leave a truncated shard under a real shard name. The
        # loader would memory-map it without complaint — uint16 length is inferred
        # from file size — and only the meta.json token-count check would notice.
        tmp = path.with_suffix(path.suffix + ".tmp")
        self.buffer[:size].tofile(tmp)
        os.replace(tmp, path)
        self.manifest.append({"path": path.name, "split": split, "tokens": int(size)})
        self.shard_index += 1
        self.fill = 0

    def add(self, tokens: np.ndarray) -> None:
        offset = 0
        while offset < len(tokens):
            space = self.shard_tokens - self.fill
            take = min(space, len(tokens) - offset)
            self.buffer[self.fill : self.fill + take] = tokens[offset : offset + take]
            self.fill += take
            offset += take
            if self.fill == self.shard_tokens:
                self._flush(self.shard_tokens)

    def flush_partial(self) -> None:
        """Close the current shard early, even if it is not full.

        Used to force a split boundary — the val shard must not spill over into
        train tokens.
        """
        self._flush(self.fill)

    def close(self) -> None:
        self._flush(self.fill)


def _write_meta(
    out_dir: Path,
    writer: ShardWriter,
    tokenizer_spec: str,
    source: str,
    extra: dict | None = None,
) -> dict:
    tok = load_tokenizer(tokenizer_spec)
    totals = {"train": 0, "val": 0}
    for entry in writer.manifest:
        totals[entry["split"]] += entry["tokens"]
    meta = {
        "source": source,
        "tokenizer": tokenizer_spec,
        # The name alone pins nothing — "gpt2" is fetched by tiktoken at runtime. The
        # fingerprint is a content hash of the vocabulary that actually tokenised this
        # corpus, so two preparations can be compared instead of trusted.
        "tokenizer_fingerprint": tok.fingerprint(),
        "vocab_size": tok.vocab_size,
        "eot_token": tok.eot_token,
        "dtype": "uint16",
        "shards": writer.manifest,
        "tokens": totals,
    }
    if extra:
        meta.update(extra)
    # Written last and written atomically: meta.json existing and agreeing with the
    # shards is the completion signal — the loader takes its shard list from it, and
    # the remote pipelines' "corpus already present" checks are only as good as it is.
    path = out_dir / "meta.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(meta, indent=2))
    os.replace(tmp, path)
    return meta


def prepare_text_file(
    input_path: str | Path,
    out_dir: str | Path,
    tokenizer_spec: str = "gpt2",
    shard_tokens: int = 1_000_000,
    val_fraction: float = 0.05,
) -> dict:
    """Tokenise a single local text file into train/val shards."""
    check_vocab_fits_shards(tokenizer_spec)
    input_path, out_dir = Path(input_path), Path(out_dir)
    text = input_path.read_text(encoding="utf-8", errors="ignore")

    _init_worker(tokenizer_spec)
    tokens = _tokenize_document(text)

    n_val = max(int(len(tokens) * val_fraction), 1)
    # A contiguous tail, not a random sample: interleaving would put text adjacent
    # to its own validation targets and report a loss that is partly memorisation.
    val_tokens, train_tokens = tokens[:n_val], tokens[n_val:]

    writer = ShardWriter(out_dir, shard_tokens=max(shard_tokens, n_val), val_shards=1)
    writer.add(val_tokens)
    writer.flush_partial()  # close the val shard before any train tokens land in it
    writer.add(train_tokens)
    writer.close()

    meta = _write_meta(out_dir, writer, tokenizer_spec, source=str(input_path))
    print(f"wrote {meta['tokens']['train']:,} train / {meta['tokens']['val']:,} val tokens")
    return meta


def _assert_trainable(meta: dict, out_dir: Path, shard_tokens: int) -> None:
    """Refuse to report success on a corpus that cannot train anything.

    Data preparation that "succeeds" while producing no training tokens is the worst
    shape of failure available here: it costs the full tokenisation time, exits 0, and
    the consequence appears much later as a loader error about missing shards, on a
    machine that bills by the minute.
    """
    train_tokens = meta["tokens"]["train"]
    if train_tokens > 0:
        return
    total = train_tokens + meta["tokens"]["val"]
    raise SystemExit(
        f"FATAL: 0 training tokens written to {out_dir} ({total:,} tokens, all validation).\n"
        f"  Shard 0 is the validation split, and this corpus did not fill even one shard "
        f"of {shard_tokens:,} tokens.\n"
        f"  Re-run with a smaller --shard-tokens (try {max(1_000_000, total // 10):,}) "
        f"or more documents."
    )


def prepare_fineweb_edu(
    out_dir: str | Path,
    tokenizer_spec: str = "gpt2",
    subset: str = "sample-10BT",
    shard_tokens: int = 100_000_000,
    limit_docs: int | None = None,
    num_proc: int | None = None,
) -> dict:
    """Stream and tokenise FineWeb-Edu into shards."""
    check_vocab_fits_shards(tokenizer_spec)
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ImportError(
            "preparing FineWeb-Edu needs the `datasets` package: pip install -e '.[train]'"
        ) from exc

    out_dir = Path(out_dir)

    # Resolve the dataset's current revision and stream *that*, so what meta.json
    # records is guaranteed to be what was tokenised. Without this the load floats on
    # the hub's default branch, and "the split is pinned" means only that its *name*
    # is — a dataset revision bump changes the corpus with nothing in this repository
    # able to say so. Resolution failure degrades to the unpinned behaviour, recorded
    # as such, rather than blocking a prep the way a hard error would.
    dataset_name = "HuggingFaceFW/fineweb-edu"
    try:
        from huggingface_hub import HfApi

        revision: str | None = HfApi().dataset_info(dataset_name).sha
    except Exception as exc:  # noqa: BLE001 - recorded, not hidden
        revision = None
        print(f"[warn] could not resolve the {dataset_name} revision, streaming unpinned: {exc}")

    dataset = load_dataset(
        dataset_name, name=subset, split="train", streaming=True, revision=revision
    )
    if limit_docs is not None:
        dataset = dataset.take(limit_docs)

        # Shard 0 is the validation split, so a corpus smaller than one shard becomes
        # *entirely* validation and leaves zero training tokens. The 100M default is
        # sized for the full 10B sample; with --limit-docs it is usually far too large.
        # Measured on the real thing: --limit-docs 40000 produced 41.8M tokens, one
        # partial shard, "wrote 0 train / 41,834,799 val tokens" — and the failure only
        # surfaced later as an unrelated-looking loader error.
        #
        # ~1,000 tokens per FineWeb-Edu document is a reasonable estimate, so aim for
        # roughly ten shards: one for validation and nine for training.
        estimated_tokens = limit_docs * 1_000
        if shard_tokens > estimated_tokens // 4:
            shard_tokens = max(1_000_000, estimated_tokens // 10)
            print(
                f"--limit-docs {limit_docs:,} is small, so shard size is reduced to "
                f"{shard_tokens:,} tokens; otherwise the whole corpus would land in the "
                f"validation shard and training would have none."
            )

    num_proc = num_proc or max(1, (mp.cpu_count() or 2) - 1)
    writer = ShardWriter(out_dir, shard_tokens=shard_tokens, val_shards=1)

    with mp.Pool(num_proc, initializer=_init_worker, initargs=(tokenizer_spec,)) as pool:
        texts = (record["text"] for record in dataset)
        progress = tqdm(unit="tok", unit_scale=True, desc="tokenising")
        for tokens in pool.imap(_tokenize_document, texts, chunksize=16):
            writer.add(tokens)
            progress.update(len(tokens))
        progress.close()
    writer.close()

    meta = _write_meta(
        out_dir,
        writer,
        tokenizer_spec,
        source=f"fineweb-edu/{subset}",
        extra={"dataset": dataset_name, "subset": subset, "dataset_revision": revision},
    )
    print(f"wrote {meta['tokens']['train']:,} train / {meta['tokens']['val']:,} val tokens")
    _assert_trainable(meta, out_dir, shard_tokens)
    return meta


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Tokenise a corpus into uint16 shards.")
    parser.add_argument(
        "--source",
        choices=["fineweb-edu", "text"],
        default="fineweb-edu",
        help="fineweb-edu streams the reproduction corpus; text tokenises a local file",
    )
    parser.add_argument("--input", type=str, help="input file, for --source text")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="gpt2")
    parser.add_argument("--subset", type=str, default="sample-10BT")
    parser.add_argument("--shard-tokens", type=int, default=None)
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="stop after N documents — useful for a quick end-to-end check",
    )
    parser.add_argument("--num-proc", type=int, default=None)
    args = parser.parse_args(argv)

    if args.source == "text":
        if not args.input:
            parser.error("--source text requires --input")
        prepare_text_file(
            args.input,
            args.out_dir,
            tokenizer_spec=args.tokenizer,
            shard_tokens=args.shard_tokens or 1_000_000,
        )
    else:
        prepare_fineweb_edu(
            args.out_dir,
            tokenizer_spec=args.tokenizer,
            subset=args.subset,
            shard_tokens=args.shard_tokens or 100_000_000,
            limit_docs=args.limit_docs,
            num_proc=args.num_proc,
        )

    _exit_before_teardown()


def _exit_before_teardown() -> None:
    """Exit without running interpreter finalisation, once the work is done and on disk.

    ``tokenizers`` holds Rust thread state that CPython's shutdown can trip over:

        Fatal Python error: PyGILState_Release: auto-releasing thread-state,
        but no thread-state for this thread

    It aborts with a core dump *after* every shard and ``meta.json`` have been written and
    flushed, so the data is complete and correct — but the process exits non-zero, and any
    caller that reasonably trusts an exit code concludes the corpus failed. That happened
    twice on rented GPUs, costing ten minutes of tokenising each time, the second time
    after I had already diagnosed it as harmless and not acted on it.

    There is nothing left to clean up at this point: shards are closed, ``meta.json`` is
    written, and ``_assert_trainable`` has already vetoed a corpus that cannot train. So
    the honest exit status is 0, and the only reliable way to report it is to skip the
    teardown that would crash. Streams are flushed explicitly first, since ``os._exit``
    does not.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
