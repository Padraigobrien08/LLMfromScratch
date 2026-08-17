# Audit — 2026-08-16

Full-repo audit: claim integrity, model/numerical correctness, test strength,
reproducibility, web app, hygiene. Every Critical/High was independently re-verified
before inclusion. Findings ranked by severity; tick items off as they're resolved.

**Bottom line:** the headline reproduction (3.0503 / 21.12 / 0.3043), every ablation
delta, both scaling tables, and the hand-written model math all survived recomputation
and execution. Failures cluster exactly where artifact-pinning stops.

---

## Resolution — 2026-08-16

Every item above is closed. What changed, beyond the individual fixes:

- **Two real defects, both reproduced and both now regression-tested.** H1 (the
  rank-0-only final evaluation deadlocking a multi-GPU run) was fixed with a
  mutation-verified two-process gloo test in `tests/test_distributed_train.py`, which
  then surfaced a *second* bug in the same code path — the on-boundary `final.pt` write
  was unguarded and raced between ranks. H5 (the site sampler dropping top-k ties) is
  fixed and tied to a hand-computed case on both sides.
- **The weak tests named in H6 and M5 now fail when mutated.** Every replacement in this
  pass was checked by breaking the code it covers and confirming it goes red.
- **Where a number could not be re-measured, the gap is disclosed rather than papered
  over**: the four uncommitted "before" values in `docs/efficiency.md`, the 5090's
  234.7 TFLOP/s ceiling, the third-party HellaSwag reference. `bootstrap.sh` now writes
  the matmul probe to an artifact so the next run does not repeat the omission.
- **The quantization rows were relabelled per-channel** — in the code, the prose, the
  site and both results files, each of which records that only a label changed.
- **New pins**, so these classes of drift fail CI rather than waiting for another audit:
  the batching table cell by cell, the utilisation arithmetic, the ablation token budget
  and seed count, the embedding fraction, the throughput-loss ranges, the SwiGLU
  parameter ratio at five widths, `modern-stack`'s composition, the wall-clock against
  the pipeline's own stage timings, and the site's hand-typed prose figures.
- **CI**: `LLMFS_REQUIRE_VOCAB=1` turns a silent offline skip into a failure, a `locked`
  job installs from `uv.lock` and runs the suite, `pages.yml` cannot deploy past a red
  suite, and both workflows declare their permissions.

Suite: 381 Python tests, 149 browser tests, all green; `ruff` clean; site builds; every
generated artifact still regenerates byte-identically.

---

## Critical

- [x] **C1. Batching table's ttft column traces to the before-mask-fix artifact.**
  `docs/efficiency.md:85-88` prints ttft 4.4 / 6.4 / 5.6 / 8.5 ms.
  `results/benchmarks-cuda.json` (after fix, 42ed0a66): 3.918 / 6.56 / 5.884 / 8.328.
  `results/benchmarks-cuda-before-mask-fix.json` (6c13dcb1): 4.374 / 6.38 / 5.553 / 8.546.
  All four printed values match the **before** file; the tok/s column in the same table
  (256/219/917/3,622) matches the **after** file. One table, two commits.
  `tests/test_documented_results.py` pins the tok/s cells but not ttft.
  *Fix: re-derive the ttft column from benchmarks-cuda.json.* Confidence: high.

- [x] **C2. "The 4090 measured the same way gives 78.9%" — unsupported; recompute gives 76.6%.**
  `docs/scaling.md:174`. `Transformer.flops_per_token()` = 1,087,183,872; best 4090
  variant (compile) 118,250.39 tok/s → 128.56 TFLOP/s achieved; ÷ measured ceiling
  167.9 TFLOP/s = **76.6%** (÷168.1 → 76.5%; ÷~165 vendor → 77.9%). The denominator
  implied by 78.9% (162.94 TFLOP/s) exists in no artifact.
  *Fix: recompute → 76.6% (or find/commit the probe that justified it).* Confidence: high.

## High

- [x] **H1. Documented multi-GPU command hangs after the last step, before `final.pt` — demonstrated.**
  `src/llmfs/train/trainer.py:316-318`: final `_evaluate_and_checkpoint(final=True)`
  runs under `is_main` only, but `evaluate()` → `all_reduce_mean` (`trainer.py:436`) is
  a collective; other ranks have already exited. Skipped only when
  `max_steps % eval_interval == 0`, and `gpt2-124m` has 19073 % 250 = 73.
  Reproduced (2-proc gloo, CPU, debug config, steps=6/interval=4): rank 1 exits 0 with
  "done", rank 0 hangs until killed; run dir has metrics through step 6 and `best.pt`,
  **no `final.pt`**. Control (steps=8/interval=4, on-boundary): both ranks exit 0.
  Note `bench/scaling.py:61` sets `eval_interval = steps+1` (always off-boundary) and
  `scaling.py:317` tolerates nonzero torchrun exits — would have masked this during the
  published scaling runs.
  *Fix (one line): run the final eval on all ranks; saving is already is_main-guarded inside.*
  Confidence: high (reproduced, with control).

- [x] **H2. Quantization rows labeled "per-tensor" measure per-output-channel.**
  `src/llmfs/quant/quantize.py:75`: `group_size=-1` → `groups = in_features` = one
  scale per output **row** (scales shape `(out_features, 1)`, verified 16 distinct
  scales on a 16×64 matrix). `src/llmfs/quant/evaluate.py:45-47` names those rows
  "per-tensor"; the label propagates to `README.md:420` and the lesson at
  `README.md:439` ("One scale per tensor is set by its largest outlier"). True
  per-tensor is not implementable with this API. Table-internal comparisons stay valid;
  the mechanism claim is not what was measured.
  *Fix: relabel rows "per-channel" and soften the lesson, or implement real per-tensor.*
  Confidence: high.

- [x] **H3. Generated ablation report asserts a false methodology; regeneration reproduces it.**
  `results/ablations.md:37`: "Each non-baseline arm is a single seed." Recomputed from
  `results/ablations.json`: all 13 arms ran seeds {1337, 1338, 1339} — 39 runs — and the
  same file's header describes paired multi-seed analysis. Emitted unconditionally by
  `src/llmfs/ablation/report.py:332` (line 223 shows the conditional pattern exists).
  *Fix (one line): condition the caveat on the arm seed count.* Confidence: high.

- [x] **H4. README says the ablation study ran at "~1B tokens"; every run is 524M.**
  `README.md:179` vs `results/ablations.json`: all 39 runs have tokens = 524,288,000;
  `docs/ablations.md` says "524M tokens per run" throughout. "~1B" matches no unit.
  *Fix: "~0.5B tokens per run".* Confidence: high.

- [x] **H5. Site sampler diverges from Python on top-k ties.**
  `web/src/lib/sampling.ts:63-68` keeps exactly k by sort rank;
  `src/llmfs/model/transformer.py:219-222` masks `logits < kth`, keeping all ties.
  Verified: counts [50,30,30,5], top-k 2 → TS probs [0.625, 0.375, 0, 0]; torch
  [0.4545, 0.2727, 0.2727, 0]. Ties are common in the bigram counts the demo samples
  from, so reachable in the UI. `sampling.test.ts` has no tie case.
  *Fix: threshold-at-the-kth-value in TS; add a tie test.* Confidence: high.

- [x] **H6a. `test_gradient_checkpointing_matches_ordinary_training` is vacuous.**
  `tests/test_train.py:270` — never runs the baseline, never compares gradients.
  Mutation: all blocks under `torch.no_grad()` → passed.
  *Fix: actually compare gradients checkpointed vs not.*

- [x] **H6b. `test_rmsnorm_statistic_computed_in_fp32` can't catch the bug it's named for.**
  `tests/test_norm_and_mlp.py:48` — rtol/atol 2e-2 admits the bf16-reduction error
  class. Mutation: removed `.float()` (max err 0.0249) → all 11 tests passed.
  Implementation itself verified correct (bit-exact vs manual fp32 reference).
  *Fix: tighten to ~5e-3 atol.*

- [x] **H6c. Top-p nucleus truncation guarded by nothing; top-k tested only at k=1.**
  `tests/test_model.py:141` asserts output shape only. Mutation: nucleus filter no-op →
  all 63 tests in test_model.py + test_kv_cache.py passed. The exclusive-cumsum
  keep-the-crossing-token rule (`transformer.py:227-229`) is untested.
  *Fix: assert the filtered distribution against a hand-computed case; test k>1 and ties.*

- [x] **H6d. Both "atomic write" tests pass with a plain non-atomic write.**
  `tests/test_train.py:144` and `tests/test_ablation.py:259` assert only file-exists +
  no `*.tmp`. Mutation: direct `torch.save` (no tmp+rename) → passed.
  *Fix: monkeypatch os.replace/torch.save to assert write-to-temp-then-rename ordering.*

## Medium

- [x] **M1. Checkpoints carry no RNG state; resume equivalence is dropout-0-only.**
  `src/llmfs/train/checkpoint.py:43-50` stores no torch/numpy/python RNG, while the
  docstring claims "everything needed to continue as though it had never stopped."
  Verified bitwise-equivalent resume on CPU for shipped configs — but only because LR is
  stateless-from-step, loader is step-derived, and every config has `dropout: 0.0`.
  Any dropout > 0 config silently diverges on resume; no test compares a resumed
  trajectory to an uninterrupted one (`tests/test_train.py:241` checks step counters only).

- [x] **M2. `drafter_forwards` accumulates across benchmark runs.**
  `src/llmfs/infer/speculative.py:119` increments and never clears (`reset()` no-op,
  not overridden); `:251` reports lifetime total; `benchmark.py:90` reuses one
  `ModelDrafter` across prompts × k. Demonstrated: identical runs report 13 then 26.
  Every model-draft row after the first in `results/speculative-cuda.json` carries the
  cumulative sum. README's quoted columns unaffected.
  *Fix: override reset() to zero the counter; benchmark already calls reset per run.*

- [x] **M3a. Uncommitted "before" values in improvement narratives.**
  `docs/efficiency.md:301-311`: "2.37×→3.00×", "7.14×→5.35×", "176→223 tok/s",
  "1,259→1,194" — only the after-values exist in artifacts.

- [x] **M3b. The 234.7 TFLOP/s 5090 matmul probe has no artifact.**
  `docs/scaling.md:153`, `docs/roadmap.md:179`. It denominates the whole "of measured
  ceiling" table (86.1%→81.9%) and the 96%-MFU refusal. `scaling-5090x8.json` has
  empty provenance (disclosed).

- [x] **M3c. "commit 42ed0a6 … 168.1 measured TFLOP/s" misattributed.**
  `docs/efficiency.md:338` — 42ed0a66's artifacts record 167.9; 168.1 belongs to the
  before-fix commit 6c13dcb1.

- [x] **M4a. "lost 74–85% of throughput on MPS" → actual range 81–86%.**
  `docs/efficiency.md:206` vs `results/quantization.json` (81.4/81.8/85.5/85.5/84.9).

- [x] **M4b. "Generated tokens 128→1024" mislabels the sweep axis.**
  `README.md:320-331`, `docs/efficiency.md:27,53` — sweep keys on `total_len`
  (prompt 32); generated counts are 96/224/480/992. The block_size-crossover punchline
  is about total length and survives; the header doesn't.

- [x] **M4c. "the token embedding is 33% of this model — 147 MiB of the 471" → 31.2%.**
  `docs/efficiency.md:174-175`, `docs/roadmap.md:82`. Disagrees with its own fraction
  and the site's correct "31%".

- [x] **M4d. "Untying would cost as much as adding two whole blocks" → ≈5.4 blocks.**
  `web/src/content/blocks.ts:162` — 38.63M added ÷ ~7.09M/block, off ~2.7×, on the page
  advertising "nothing here is arithmetic anybody did by hand".

- [x] **M4e. "Every change that improves loss costs throughput and vice versa" — contradicted by its own table.**
  `README.md:243-245` (lr-3e-3: −0.1251 loss AND +1.1% tput; sched-wsd likewise).
  `docs/ablations.md:213` scopes it correctly to the architecture arms.

- [x] **M5a. Tokenizer/viz pins skip silently offline.**
  `tests/test_tokenizer.py:92,123`, `tests/test_viz.py:137` skip when tiktoken can't
  fetch the GPT-2 vocab — a cold-cache/offline runner silently drops the 50257 check,
  the Python half of the bidirectional tokenizer pin (README:527), and ~7 viz tests.
  Nothing asserts these ran.

- [x] **M5b. `modern-stack` arm pinned by nothing.**
  Excluded from the single-axis test (`tests/test_config.py:130-161`); drift in its 5
  fields would silently corrupt the "components are additive" comparison.

- [x] **M5c. Quant round-trip test bound is global-max, not per-group.**
  `tests/test_quant.py:39` — `.max()` collapses per-group steps; an element may err up
  to the largest group's step/2 (measured spread 2.7× at gs=32) — loosest exactly in
  the outlier regime the module exists for.

- [x] **M5d. No test exercises trainer `no_sync`/all-reduce logic or seeded-training determinism.**
  `trainer.py:359-363` untested (scaling CLI tests monkeypatch `run` away);
  README:363's "tested rather than asserted" describes a one-time GPU measurement.
  Determinism verified bitwise on CPU here, but by hand, not by the suite.

- [x] **M6. SwiGLU 2/3-width parameter match holds only at friendly widths.**
  `src/llmfs/model/config.py:98-103` — round-up to 256 after the 2/3 scaling.
  Measured SwiGLU/GELU param ratios: d=768 → 1.0000 exact (shipped config: claim true);
  d=384 → 1.0000; d=512 → **1.125**; d=1024 → 1.031; d=128/64 → **1.50**.
  The ablation scale is 512-wide, so the `mlp-swiglu` arm carries a 12.5% parameter
  advantage — worth a caveat in docs/ablations.md.

## Low

- [x] `src/llmfs/quant/evaluate.py:53-71` — docstring says "non-overlapping blocks";
  stride 512 < block 1024 double-counts tokens past 512. Δppl unaffected; absolute ppl
  is a nonstandard estimator.
- [x] `README.md:239` "180× more" → 171× at full precision (0.125124/0.0007304).
- [x] `docs/reproduction.md:47` "7.1 h" → 6.99h from run.log (19,073 × 1.305s = 6.91h).
- [x] `README.md:447` / `docs/efficiency.md:174` "caps at 2.42×" — own table shows 2.47×
  (2.42 is the best-quality scheme, not the cap).
- [x] `docs/efficiency.md:151` "16 MiB in extra scales" → 15.2 (211.59 − 196.40).
- [x] `docs/efficiency.md:188` "2.02×" reconstructs to 1.93–1.99 from the artifact.
- [x] `README.md:312` "34% slower" is the worst length only (17.8% at 1024); "flat at
  ~170" spans 161–172.
- [x] `docs/ablations.md:156` "half again the non-embedding count" → ≈1.0× ("as much again").
- [x] `README.md:242` "within a third of the noise floor" → 0.335, marginally above.
- [x] `docs/efficiency.md:180` / `web/src/content/testShowcase.ts:80` — "196 MiB becomes
  217": 217 appears in no artifact.
- [x] HellaSwag reference 0.2955 is hardcoded into `results/hellaswag.json`, so tests
  verify a constant against itself; 3.29 target honestly flagged as secondary-sourced.
- [x] Unpinned retyped web prose numbers (all currently correct, drift risk only):
  "31%" (`chapterBodies.tsx:142`, `chapters.ts:72`), "786,432" (`chapterBodies.tsx:262`),
  "about 7.5 in loss" (`:355`), "51M" (`:435`), "all 14 cases" (`:108`),
  "over 50 steps / fifty steps" (`Scaling.tsx:55,92`), "30% regression"
  (`Efficiency.tsx:134`).
- [x] `web/src/components/LossCurve.tsx:17-18` — `Y_LO = 2.95` would clip a rerun that
  finishes below it.
- [x] `web/scripts/check-counts.mjs` checks exactly one number (browser test count);
  its "no figure the site prints is unchecked" comment overreaches (see prose list above).
- [x] `web/src/lib/reproductionCurve.test.ts:41-50` — comment claims a comparison the
  test doesn't make. (I ran it: crossingOf(val, 3.29) → step 6500, byte-identical to
  MEASURED — the fact holds; the test doesn't check it.)
- [x] `scripts/gpu.sh:488` + `scripts/remote/watchdog.sh` (~line 83) — RunPod API key
  sent as URL query parameter; proxy/access logs could capture it. Header auth
  (`Authorization: Bearer`) is strictly better. No key value committed anywhere,
  including all of git history.
- [x] CI installs from pyproject `>=` ranges (`uv pip install`), never from `uv.lock`
  (`uv sync` would) — lock-based reproduction asserted but never exercised.
- [x] `.github/workflows/pages.yml` deploys on push to main independent of ci.yml —
  red tests don't stop a deploy; rebuilds rather than reusing the tested artifact.
- [x] `web/src/node_modules/` untracked vitest cache (`.gitignore:44` only covers
  `web/node_modules/`); gitignore or set vitest cacheDir.
- [x] Dead .gitignore rules: `bench/scratch/` (no such dir), deleted plan docs
  (`.gitignore:62-63` — possibly intentional recreation guards).
- [x] `data/wizard/meta.json` tracked while sibling shards are ignored — regenerable
  artifact half-committed.
- [x] `data/bpe_tokenizer.json` — 9.7MB, largest tracked file, legacy-era, used by one
  test, provenance undocumented.
- [x] Notebooks/legacy leak local paths with the (already-public) username; no source
  attribution for the (correctly stripped) public-domain Oz text.
- [x] `ci.yml` has no explicit permissions block (repo-default token) — minor hardening.

---

## Verified sound (audit coverage)

Recomputed from artifacts or executed, not read:

- **Headline reproduction**: 3.0503 / 21.12 / 0.3043 / +0.0088 / −0.2397; full
  100M-token split; crossing at step 6,500 (34%); 44.1% MFU; ~401k tok/s and
  1,305 ms/step verbatim in run.log; ~$23; FLOP arithmetic.
- **Model math by execution**: RoPE relative-position property, norm preservation,
  prefill/decode position consistency, odd-dim rejection; RMSNorm bit-exact vs manual
  fp32 reference; GQA (both paths) vs manual reference incl. n_kv_head=1; bottom-right
  mask; full-vs-incremental cache equivalence at every position incl. chunked prefill,
  block boundary, clean overflow errors, rewind_to after garbage; decode step reaches
  SDPA with attn_mask=None over an exactly-valid-length cache; no future leakage
  (bitwise + zero gradients); weight tying, GPT-2 init, deterministic seeded
  generation; quant round-trip bounds, 4-bit packing, QuantLinear ≡ F.linear,
  tied-head refusal; speculative decoding lossless vs greedy across 160 runs incl.
  adversarial and contract-violating drafters.
- **Every artifact-pinned number**: ablation deltas/verdicts/noise floor (39 runs,
  csv↔json↔md consistent), both scaling tables cell-by-cell, amortisation fit and
  predictions, both cache sweep tables vs their respective artifacts, full quantization
  and speculative tables, H100/4090 training tables incl. both OOMs, Young/Daly
  arithmetic in fault-tolerance.md.
- **Suites**: pytest 346 passed / 0 failed / 0 skipped locally; npm build clean;
  npm test 137/137; check-counts passes; measured.ts/testShowcase.ts regenerate exactly.
- **Web pinning**: TS↔Python ablations math bit-identical on the real artifact; RoPE
  convention/theta/rotation identical; all five web/src/data fixtures + model-sizes +
  reproduction-curve regenerate byte-identically; all 16 web/public/data JSONs
  byte-identical to results/.
- **Reproducibility**: all 15 console scripts resolve; all 16 configs load; every
  documented flag exists; 11 single-axis arms verified single-axis; single-run
  determinism and interrupt/resume bitwise-identical on CPU.
- **Hygiene**: no secrets in tree or any historical blob; .gpu.env/.gpu-state properly
  ignored; CI runs lint, lock-sync check, pytest 3.10–3.12, web tests, build, full CPU
  end-to-end, no path filters.
- **Strong tests confirmed strong**: causality-by-perturbation with positive control
  across 10 variants, eager-vs-SDPA cross-check, SDPA-dispatch spy, chunked-verify
  interior-queries test, arange-corpus loader tests, doc-pinning suites.

Correction to the audit brief: tests/ is ~3,785 lines, not ~12k.

## Not checked, and why

- GPU-measured numbers themselves (H100/4090/5090 throughput, memory, TFLOP/s probes):
  no CUDA hardware; audited for artifact-consistency and internal arithmetic only.
- H1 under NCCL: demonstrated on gloo/CPU (indefinite block); NCCL would abort on the
  watchdog timeout instead — same user-facing failure, not separately confirmed.
- External reference provenance (GPT-2's 3.29 val loss, 0.2955 HellaSwag): third-party
  figures; confirming them means reproducing the references.
- FineWeb-Edu preparation at scale: multi-GB download; argparser and code path only.
- The deployed site was not diffed against a local build; source, tests, and build were.
