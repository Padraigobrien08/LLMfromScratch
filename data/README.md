# What is in `data/`, and where it came from

Three things are tracked here. Everything else in this directory — prepared shards,
downloaded corpora, HellaSwag — is regenerable and is gitignored.

## `wizard_of_oz.txt`

*Dorothy and the Wizard in Oz* by L. Frank Baum (1908), from **Project Gutenberg**
([ebook #420](https://www.gutenberg.org/ebooks/420)). The work is in the public domain in
the United States. The Project Gutenberg licence header and trailer have been stripped, so
this file carries no Gutenberg trademark and is not distributed under their licence — it is
the public-domain text alone, which is what their licence section 1.E.9 permits.

It exists so `llmfs-prepare-data --source text` and `llmfs-train --config debug` work
immediately after a clone, on any machine, with no download. It is also the corpus behind
the bigram sampler on the site's explainer page.

## `wizard/meta.json`

The manifest `llmfs-prepare-data` writes for the tokenized corpus. Its sibling `*.bin`
shards are gitignored — they are binary and regenerable — but the manifest is not, on
purpose: it records the tokenizer, vocabulary size and per-shard token counts, and
`Trainer` refuses to start when the manifest disagrees with the model config. Keeping it
tracked means that check has something to check against on a fresh clone, and means the
tokenizer the demo corpus was built with is part of the repository rather than part of
whoever last ran the command.

Regenerate the shards it describes with:

```bash
llmfs-prepare-data --source text --input data/wizard_of_oz.txt --out-dir data/wizard
```

If that ever produces a manifest differing from the committed one, the tokenizer or the
preparation code has changed and the difference is worth reading rather than committing.

## `bpe_tokenizer.json`

**Legacy.** A 10 MB HuggingFace `tokenizers` BPE vocabulary trained during the original
tutorial project, before this repository moved to GPT-2's byte-level BPE via `tiktoken`.
Nothing in `src/llmfs` requires it: the trainer, the reproduction and the site all use
`gpt2`.

It stays tracked for one reason — `tests/test_tokenizer.py` uses it to exercise the
`file:` loader path against a real, non-trivial vocabulary that needs no network. Those
tests skip cleanly when it is absent, so a fork that wants the 10 MB back can delete it
without breaking the suite. It is the largest tracked file in the repository, and it is
here on sufferance rather than because anything depends on it.
