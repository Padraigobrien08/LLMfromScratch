# LLMfromScratch

A from-scratch decoder-only language model: a GPT-2 124M reproduction, extended with
modern architecture components, a controlled ablation study, and efficiency
benchmarks — reproducible from one command.

[![CI](https://github.com/Padraigobrien08/LLMfromScratch/actions/workflows/ci.yml/badge.svg)](https://github.com/Padraigobrien08/LLMfromScratch/actions/workflows/ci.yml)
[![Site](https://github.com/Padraigobrien08/LLMfromScratch/actions/workflows/pages.yml/badge.svg)](https://padraigobrien08.github.io/LLMfromScratch/)

**→ [Open the interactive site](https://padraigobrien08.github.io/LLMfromScratch/)** — an
[explainer that assumes no prior knowledge](https://padraigobrien08.github.io/LLMfromScratch/#/explainer),
built out of things you can poke at: a real tokenizer, a parameter budget you can drag, a sampler
you can turn up. Then the deep ends — [RoPE's defining property holding as you move two
tokens](https://padraigobrien08.github.io/LLMfromScratch/#/rope), an
[ablation playground](https://padraigobrien08.github.io/LLMfromScratch/#/ablations), and
[every attention weight](https://padraigobrien08.github.io/LLMfromScratch/attention/) per layer
and per head.

Every architecture component here — rotary embeddings, RMSNorm, SwiGLU,
grouped-query attention, the KV cache — is written by hand and covered by tests that
assert its defining mathematical property, not just its output shape.

---

## Status

This repository is under active development. The table is the honest state of it:
what is built and verified, and what is designed but not yet run.

| Pillar | Status |
| --- | --- |
| Package, config system, data pipeline, trainer, CI | **Done** — 223 tests green, end-to-end verified |
| Modern architecture (RoPE, RMSNorm, SwiGLU, GQA, KV cache) | **Done** — hand-implemented, property-tested |
| GPT-2 124M reproduction on FineWeb-Edu | Configured; **GPU run pending** |
| Ablation study (12 arms × 3 seeds) | **Done** — [docs/ablations.md](docs/ablations.md), 39 runs, 7.6 GPU-h |
| Efficiency benchmarks (throughput, memory, KV cache) | **Built** — `llmfs-bench`; **GPU run pending** |
| Quantization + speculative decoding | **Not started** |
| Fault-tolerance design doc | **Done** — [docs/fault-tolerance.md](docs/fault-tolerance.md) |
| Multi-GPU scaling report | DDP wired; **scaling run pending** |
| Interactive attention visualization | **Done** — [live](https://padraigobrien08.github.io/LLMfromScratch/attention/), auto-deployed from CI |
| Interactive site (explainer, RoPE explorer, ablation playground) | **Done** — [live](https://padraigobrien08.github.io/LLMfromScratch/); the playground shows a pending state until the sweep publishes |

No results are reported below that have not been measured. Sections describing
pending work say so.

---

## Architecture

One `Transformer` class covers both the GPT-2 baseline and the modern Llama-style
stack; which one you get is decided entirely by config. That is deliberate — an
ablation that swaps LayerNorm for RMSNorm differs from its baseline by one line of
YAML, so the two cannot silently drift apart.

| Component | Baseline (`gpt2-124m`) | Modern (`llama-124m`) |
| --- | --- | --- |
| Normalisation | LayerNorm | **RMSNorm** (fp32 reduction) |
| Position | Learned absolute table | **RoPE** |
| Feed-forward | GELU, 4×d | **SwiGLU**, parameter-matched via 2/3 width scaling |
| Attention | 12 query / 12 KV heads | **GQA**, 12 query / 4 KV heads (3× smaller cache) |
| Bias terms | Yes | No |
| Inference | — | **Static preallocated KV cache** |

```python
from llmfs import ModelConfig, Transformer

model = Transformer(ModelConfig(
    n_layer=12, n_head=12, n_kv_head=4, n_embd=768, block_size=1024,
    norm="rmsnorm", pos_emb="rope", mlp="swiglu", bias=False,
))
```

### What the tests actually assert

Shape checks catch typos; these catch the bugs that would otherwise survive all the
way into a training run and only show up as a mysteriously worse loss:

- **RoPE** — that `⟨R(q, m), R(k, n)⟩` depends only on `m − n`, to 1e-6, which is the
  entire point of rotary embeddings and is not implied by any shape.
- **Causality** — perturbing token *t* leaves every position `< t` bitwise unchanged.
  Run across all 10 architecture variants. An off-by-one mask makes loss look *better*.
- **KV cache** — incremental decoding reproduces a full forward pass at every
  position, for every variant. Training never exercises the cache, so nothing else
  would catch a stale offset or a double-rotated key.
- **GQA** — with `n_kv_head == n_head`, output is numerically identical to plain MHA.
- **Eager vs fused** — the attention-weight export path matches the SDPA kernel, so
  the visualizer cannot show weights the model never used.
- **Mask alignment** — `build_causal_mask` is bottom-right aligned. PyTorch's
  `is_causal=True` is top-left aligned and is silently wrong whenever the query block
  is shorter than the key sequence — i.e. on every decode step.

```bash
pytest tests -q
```

---

## Quickstart

```bash
git clone https://github.com/Padraigobrien08/LLMfromScratch.git && cd LLMfromScratch
uv venv && uv pip install -e ".[dev,train]"
```

Train a small model on the included corpus in a couple of minutes, on CPU, MPS or
CUDA — no download required. This is the smoke test that every code path a real run
takes is exercised before renting a GPU:

```bash
llmfs-prepare-data --source text --input data/wizard_of_oz.txt --out-dir data/wizard
```

```bash
llmfs-train --config debug
```

```bash
llmfs-generate --checkpoint out/debug/best.pt --prompt "Dorothy lived in the"
```

The full reproduction, on a rented GPU:

```bash
llmfs-prepare-data --source fineweb-edu --out-dir data/fineweb-edu-10B
```

```bash
llmfs-train --config gpt2-124m
```

Multi-GPU is the same config — gradient accumulation absorbs the world size, so the
optimisation is unchanged and only throughput moves:

```bash
torchrun --nproc_per_node=8 -m llmfs.train.cli --config gpt2-124m
```

On a rented pod, [`scripts/gpu.sh`](scripts/gpu.sh) drives the whole thing over SSH —
setup, data, detached run, monitoring, and fetching results back. Jobs run inside
tmux so a dropped connection cannot kill something you are paying for.
[docs/gpu-runbook.md](docs/gpu-runbook.md) is the operational guide.

```bash
./scripts/gpu.sh setup && ./scripts/gpu.sh sweep && ./scripts/gpu.sh watch
```

Every run writes its fully-resolved config, `metrics.jsonl`, and atomic checkpoints
to `out/<run_name>/`. Resume with `--resume auto`.

---

## Reproduction

The trust anchor: a documented target, hit within a stated tolerance, rather than a
model that merely emits plausible text.

- **Target**: GPT-2 124M on FineWeb-Edu (sample-10BT), one epoch, 10B tokens.
- **Protocol, provenance of the target number, and tolerance**:
  [docs/reproduction.md](docs/reproduction.md).
- **Status**: not yet run. The config, data pipeline and trainer are complete and
  verified end to end at small scale; what remains is the GPU time.

Results, loss curves, wall-clock, hardware and cost will be published here once the
run completes.

---

## Ablation study

Twelve arms against a shared baseline: eleven vary exactly one design decision, and
`modern-stack` combines them to test whether the individual deltas actually add up.
Run at a smaller scale than the reproduction (8 layers, 512-wide, ~1B tokens) so the
whole sweep is affordable and every arm can share a seed and a token budget.

Axes: LayerNorm vs RMSNorm · learned vs RoPE vs none · GELU vs SwiGLU · tied vs untied
embeddings · bias vs no bias · MHA vs GQA · cosine vs WSD schedule · weight decay ·
learning rate (3e-4 / 1e-3 / 3e-3) · all modern components combined.

The discipline this rests on is enforced by a test: every arm is asserted to differ
from `configs/ablations/_base.yaml` in its own named axis and nothing else. An arm
that drifted would be measuring something other than what it claims.

```bash
llmfs-ablate --seeds 3
```

```bash
llmfs-ablate-report
```

**Every arm runs at the same three seeds, and comparisons are paired.** That is the
point of the design, not a detail. Two runs differing only in seed do not reach the
same loss, so an unpaired comparison has to clear that entire spread before it can
claim anything — and most architecture effects are smaller than it. Differencing each
arm against the baseline run that saw its data *in the same order* cancels the
batch-ordering variance the two share, and resolves effects well below the raw noise
floor.

An arm counts as a result only when the range of its per-seed deltas does not
straddle zero: every seed agreed on the direction. A deliberately blunt rule, not a
p-value — with three seeds nothing stronger would be honest — and it is exactly what
the error bars in the plot show. An ablation table without that check is worse than
no table: it reads as authoritative while recommending changes that do nothing.

The runner is built for a multi-hour job on rented hardware: it skips arms that
already have a result, writes after every arm, and records a diverged arm as a
finding rather than dying on it — the `lr-3e-3` arm is *expected* to blow up.

### Results

**[Full write-up: docs/ablations.md](docs/ablations.md)** — 39 runs, 7.6 GPU-hours on
one H100, ~$25. Baseline 3.9116, seed noise floor **0.0043**.

| Arm | Δ val loss | Δ throughput |
| --- | --- | --- |
| `lr-3e-3` | **−0.1251** | +1.1% |
| `sched-wsd` | **−0.1034** | +0.8% |
| `modern-stack` | **−0.0886** | ±0.0% |
| `pos-rope` | **−0.0886** | −2.3% |
| `mlp-swiglu` | **−0.0341** | −2.2% |
| `norm-rmsnorm` | +0.0007 *(within noise)* | +1.3% |
| `no-bias` | +0.0038 | **+4.2%** |
| `gqa-2` | +0.0311 | +2.0% |
| `lr-3e-4` | +0.4457 | +0.9% |

![Ablation deltas](results/ablation_deltas.png)

Three things worth pulling out:

- **The optimiser dominates the architecture.** Learning rate and schedule move loss
  more than every architecture change combined. RMSNorm vs LayerNorm is worth 0.0007;
  the learning rate is worth 0.125 — 180× more.
- **The components are additive.** Summing the five individual modern-stack parts
  predicts −0.0872; the combined arm measured −0.0886. Within a third of the noise
  floor, so they compose without interacting.
- **Loss is the wrong single metric.** Every change that improves loss costs
  throughput and vice versa. `modern-stack` buys −0.0886 for ±0.0% — which a
  loss-only table cannot show.

The prediction that `lr-3e-3` would diverge was wrong: it won. That means every arm
was measured at a learning rate now known to be suboptimal, which is the study's
largest caveat and is stated as such in the write-up.

---

## Efficiency

The consolidated benchmark — naive → KV cache → quantized → speculative decoding,
with latency, tokens/sec, memory and cost — is the headline deliverable here and is
**not yet built**.

One number is measured so far, and only because it falls out of the cache tests: on a
7M-parameter debug model on MPS, KV-cache decoding runs at 44.5 tok/s against 14.5
tok/s for the naive re-forward baseline (3.1×), producing bitwise-identical output
under greedy decoding. That is a toy model on a laptop, reported as a sanity check
rather than a benchmark.

Already wired and awaiting measurement on real hardware: mixed precision (bf16),
`torch.compile`, gradient checkpointing, fused AdamW, TF32, and MFU logging.

---

## Interactive site

**[padraigobrien08.github.io/LLMfromScratch](https://padraigobrien08.github.io/LLMfromScratch/)**

**"How a language model actually works"** is the on-ramp: eight steps from a sentence
you type to a model that predicts what comes next, assuming no prior knowledge, with
something to poke at rather than take on faith at each one. Type text and watch the
real GPT-2 vocabulary cut it into tokens. Drag `n_layer` and `n_embd` and watch 31% of
a 124M model turn out to be the embedding table before any computation happens. Sample
from a real next-token distribution and see what temperature, top-k and top-p actually
do. Read a validation loss as "how many equally likely options is it choosing between?"

Three things run in the browser that would normally be hand-waved, and each is pinned
to the Python that produced it: the **BPE tokenizer** (the real merges, exact on all 14
fixture cases including emoji, newlines and leading spaces), the **parameter
accounting** (exact against the real `Transformer` for twelve configurations), and the
**sampler** (same temperature → top-k → top-p order as `Transformer.generate`,
including the off-by-one that makes `top_p = 0` sample from nothing).

The sampling distribution is a bigram model counted from `data/wizard_of_oz.txt` — real
statistics rather than invented logits, and the model this project began as. It also
sets up the next step honestly: draw a few tokens and the text wanders, because it can
only see one token back, which is precisely what attention exists to fix.

**The RoPE explorer** puts the property this repository tests for on screen. Drag a
query and a key along a sequence and the attention logit between them does not move,
as long as the distance between them does not — one dial per dimension pair shows
every arrow spinning while every angle between them holds. A panel accumulates the
range the logit actually occupies as you slide, so the page reports its own
assertion: across ~900 samples spanning positions 0–512, the logit moves by about
5e-15.

The rotation it draws is a TypeScript port of `src/llmfs/model/rope.py`, and the two
are pinned together **in both directions**: a fixture generated by the Python
implementation is asserted by the browser tests, and `tests/test_rope.py` asserts
that same committed fixture still reproduces the model. Either side changing alone
fails CI. A visualization that has quietly drifted from the code is worse than none,
because nothing about it looks wrong.

That pairing surfaced one measured detail worth stating: the cos/sin tables are built
in float32 while the port uses float64, so the two agree to ~1e-8 early in a sequence
and ~3e-7 by position 200. That is the same choice HF Llama makes and sits far below
the resolution of the bf16 activations consuming it — but it is asserted as a bound
rather than assumed.

**The ablation playground** turns the sweep into something you can interrogate:
toggle a design decision and get its paired delta, its per-seed values, the error bar,
and a verdict — including "not a result" when the seeds disagree. Toggle two and it
says the combination was never measured, rather than adding the individual deltas and
pretending that is a finding. The analysis mirrors `ablation/report.py`, so the page
cannot report something the repository's own report would refuse to.

```bash
npm install --prefix web && npm run dev --prefix web
```

---

## Attention explorer

**[padraigobrien08.github.io/LLMfromScratch/attention/](https://padraigobrien08.github.io/LLMfromScratch/attention/)**

Every attention weight in the model, per layer and per head, in a page you can click
through. Built by CI from a model CI trains, and deployed to GitHub Pages on every
push to `main` — so the hosted page always reflects the current code rather than a
stale artifact.

```bash
llmfs-viz --checkpoint out/debug/best.pt --out site/attention.html
```

```bash
llmfs-viz-serve --checkpoint out/debug/best.pt   # type your own text
```

Four views, each answering a different question:

- **Which tokens attend to which** — click a token to make it the query; every other
  token is shaded by the attention it received. This is the view the whole tool exists
  for, and it reads as a sentence rather than a matrix.
- **All heads** — one thumbnail per head, so structure (diagonals, sinks, stripes) is
  visible across the whole model at a glance instead of one head at a time.
- **Focused head** — the full token × token heatmap, hover for exact weights. Masked
  cells are left as page background, so "structurally impossible" looks different from
  "attended with weight zero".
- **Head statistics** — entropy, mean attention distance, previous-token fraction and
  sink fraction, per head. These turn a grid of 144 heads into something searchable:
  sort by previous-token fraction and the induction-circuit building blocks come to
  the top.

Two engineering notes. The export is a **single self-contained HTML file** — no build
step, no CDN, no backend — because a visualisation with a server attached is a URL
that will be down the day someone looks at it; a test asserts no external resource is
ever referenced. And the weights are quantised to uint8 and base64-encoded rather than
written as JSON numbers: for a 12×12-head model over 64 tokens that is 590k values,
several megabytes as text, for cells a few pixels wide. Statistics are computed at
full precision *before* quantisation.

The hosted demo runs a deliberately small model, and the page header states its
parameter count, step and validation loss so nobody has to guess. It gets more
interesting the moment the 124M checkpoint exists — pointing `llmfs-viz` at it is the
only change needed.

---

## Reliability

[**docs/fault-tolerance.md**](docs/fault-tolerance.md) — the design doc for running a
24-hour job on hardware that fails: failure taxonomy, checkpointing strategy,
resumption semantics, silent-corruption and straggler detection, and what breaks at
1,000+ GPUs.

Two things it produced that changed the code's direction rather than just describing
it:

- **The checkpoint interval is denominated in the wrong unit.** Applying the
  Young/Daly optimum to this run's real step times shows the configured 1000-step
  default wastes ~16% of a single-GPU spot run — about 3.9 hours — because a *step* is
  not a fixed amount of wall-clock, and the failure rate it guards against is.
- **"Atomic write" was over-claimed.** `os.replace` is atomic against interruption but
  not against power loss, since POSIX does not guarantee the data reached disk before
  the rename became visible. Documented as a gap with the fix, rather than left as a
  claim that sounds stronger than it is.

The doc marks every claim **[implemented]**, with the test that pins it, or
**[proposed]**, with an effort estimate — and ends with a prioritised gap list where
the top five items total under a hundred lines.

---

## Repository layout

```
src/llmfs/
  model/      RoPE, RMSNorm, SwiGLU, GQA attention, KV cache, transformer
  data/       tokenizer, FineWeb-Edu preparation, memory-mapped shard loader
  train/      trainer, optimiser and schedules, checkpointing, distributed setup
  eval/       evaluation and generation entrypoints
  viz/        attention extraction, head statistics, static export, live server
  ablation/   sweep runner, paired-seed analysis, tables and plots
  bench/      training + inference throughput, memory, cost, provenance
configs/      gpt2-124m, llama-124m, debug, and 11 single-axis ablation arms
tests/        223 tests — component correctness, config validation, end-to-end training
web/          the interactive site: explainer, RoPE explorer, ablations (69 tests)
scripts/      GPU pod automation, and the exporters that pin the site to the model
docs/         reproduction protocol, fault-tolerance design
notebooks/    exploration only; nothing here is the source of truth
legacy/       the original tutorial scripts, kept for reference
```

---

## Origin

This began as a tutorial reproduction of a small character-level GPT
([video](https://youtu.be/UU1WVnMk4E8), [write-up](https://app.readytensor.ai/publications/building-a-transformer-based-llm-from-scratch-using-pytorch-HMEzasyetWey)),
and the original scripts are preserved unmodified in `legacy/`.

Everything above the `legacy/` directory is a rewrite, not a refactor. The tutorial
code had hard-coded absolute paths, module-level globals, post-norm blocks,
generation that re-ran the full prefix for every token, and its model living inside a
notebook. The current codebase shares no logic with it.

## License

MIT
