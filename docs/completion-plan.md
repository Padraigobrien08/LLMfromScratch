# Plan to finish

A working document, not a showcase one. [docs/roadmap.md](roadmap.md) covers work
deliberately *outside* the project — kernels, multi-node. This covers the path to calling it
done.

**Done means four things:**

1. It reads as a 90+/100 pinned portfolio repository.
2. A researcher forms the judgement "this person knows what they are doing" — from evidence,
   not assertion.
3. Someone who has never trained a model can work through the site and come out understanding
   what a language model does and what this project measured.
4. The broadsheet site is a **core component** — the paper itself, not documentation of a
   paper that lives elsewhere.

Goal 4 is the one that reorders everything. If the site is the paper, then a paper with no
results section is not nearly finished, and that is the current state.

---

## Where it actually stands

**The Python side is complete.** Seven pillars measured, 327 tests, CI green, every documented
number enforced against `results/` by `tests/test_documented_results.py`. Nothing on the
critical path needs a GPU again.

**The site is roughly half built, and the built half is good.** The broadsheet redesign — a
front page, eight explainer chapters, a masthead, a dateline rail, plate numerals, a design
system — exists and builds clean with 84 browser tests. The RoPE explorer is ported into it.

What is missing is the part a researcher came for:

| Route | State |
| --- | --- |
| `#/` front page | Built |
| `#/chapter/1…8` explainer | Built — this is the newbie path, and it is the strongest part |
| `#/rope` | Built, ported to the new design |
| `#/ablations` | Renders inside a `<Legacy>` wrapper — pre-redesign styling |
| `#/architecture` | `<Placeholder title="The architecture page" />` |
| `#/tests` | `<Placeholder title="The test-suite page" />` |
| reproduction | **No page.** The headline result is absent from the site |
| efficiency | **No page.** Quantization, speculative decoding, the KV-cache bug |
| scaling | **No page.** 95.1% on PCIe, the accumulation fit |

So: of the four results a researcher would weigh, one is on the site in legacy styling and
three are not on it at all. The site currently argues that the author can explain a
transformer. The repository proves considerably more than that.

---

## Phase 0 — stop losing work (do first, minutes)

**31 files of the redesign have been uncommitted since 2026-08-10.** That includes the entire
`content/` and `design-system/` directories and both new pages. A lost laptop, a bad `git
checkout`, or a `git stash` at the wrong moment costs the whole redesign.

Commit it, in coherent pieces, even mid-refactor. Placeholders and a `<Legacy>` wrapper are
fine in a commit; unversioned work is not.

*Coordination note:* that work belongs to a parallel session. Nothing in later phases should
be started until it is committed, or the two will collide in the same files.

---

## Phase 1 — the site may not claim more than the repo (half a day)

`content/projectState.ts` is the site's single source of truth for repository claims, with a
comment on each naming what it is read from. Good discipline — and it has already drifted:

```
pythonTests: 223    // actually 327
browserTests: 69    // actually 84
```

It also carries no scaling or efficiency figures, because those did not exist when it was
written.

Hand-transcription always drifts. `tests/test_documented_results.py` solved exactly this for
the docs by reading `results/*.json` and asserting the figures appear where cited; mutation
testing showed the first version was far too weak, and fixing it took three passes. **Extend
the same enforcement across the language boundary.**

The mechanism: a committed generator turns `results/*.json` into a TypeScript module —

```
results/*.json  ->  llmfs-export-web  ->  web/src/content/measured.ts  (generated, committed)
```

— and a test asserts the checked-in file matches what the generator produces, so a stale
export fails CI rather than shipping. The site then imports figures instead of restating them.
`frontFigures.ts` already argues for this: *"A front page that asserts numbers it does not
derive is exactly what the rest of the site is arguing against."* This finishes that argument.

Also worth doing here: the site should read the test counts from a generated file too, so
"327 tests green" cannot become a lie the moment someone adds a test.

**Why this is Phase 1 and not polish.** A researcher who spot-checks one number against the
repo and finds it stale stops trusting every other number on the page. This is the cheapest
possible defence of goal 2.

---

## Phase 2 — the paper's body (the bulk of the work)

Four plates, each a chapter-sized page in the broadsheet design, each built around **one
interactive figure that carries the argument**. This is where goals 2, 3 and 4 are all won or
lost at once, because the same figure has to satisfy a researcher and teach a newcomer.

The design rule that makes that possible: *the interaction is the explanation.* Not a chart
with a tooltip — a control whose movement demonstrates the claim.

### Plate: The reproduction

The trust anchor, and it belongs at the front of the results. Val loss **3.0503** against a
**3.29** target pre-registered before the run; HellaSwag **0.3043** against the published
GPT-2 124M **0.2955**; 44.1% MFU held flat for seven hours; ~$23.

*Figure:* the loss curve with the target line drawn across it, and a scrubber. Dragging it
shows the crossing at step 6,500 — 34% of the run — and the run continuing to improve after
it. A newcomer learns what "training" looks like; a researcher sees a pre-registered target
met with room to spare.

*Data:* `results/reproduction.json`, `results/hellaswag.json`, `metrics.jsonl` for the curve.

### Plate: What actually matters (ablations)

Already has a playground; needs lifting out of `<Legacy>` into the broadsheet design. The
finding is genuinely interesting and under-sold: **the optimiser dominates the architecture.**
Learning rate is worth −0.1251 and the schedule −0.1034, while RMSNorm is worth +0.0007 — and
components compose almost additively (predicted −0.0872, observed −0.0886).

*Figure:* the existing arm comparison, plus the paired-seed logic made visible — three seeds
per arm, significance as "the per-seed deltas do not straddle zero", not a p-value. That
distinction is exactly what a researcher notices.

### Plate: Making it fast, and the bug that hid in the numbers

The most instructive plate for goal 2, because its subject is a mistake. The KV cache was
**34% slower** than recomputation and I explained it away as a hardware property; the sweep
that was supposed to confirm the explanation exposed a mask that forfeited the fused flash
kernel. Fixing it gained 1.30–1.38×.

*Figure:* the cache-vs-recompute curves with a toggle for the mask bug — flip it and watch the
cache go from losing to winning. Underneath: quantization's memory/quality trade-off (int8
costs +0.013 perplexity, int4 g128 +1.351) and speculative decoding's 5.35× with the
losslessness invariant stated.

A researcher reading "here is a bug I talked myself into, and here is the shape of the data
that eventually gave it away" learns more about the author than any clean result does.

### Plate: Eight GPUs, and why the interconnect barely matters

The best interactive figure available in the whole project, because the mechanism is
*predictive*. 95.1% efficiency on 8 GPUs with **no NVLink**, then the accumulation sweep:
96.6 → 95.2 → 92.1 → 86.0 as the all-reduce is amortised over less compute.

*Figure:* a slider for gradient accumulation. The measured points appear, the fitted
`a + b/accum` curve is drawn through **two** of them, and the other two — predicted before
they were measured — land on it. A newcomer sees "sharing costs time"; a researcher sees a
mechanism proposed, quantified, and then tested out of sample.

*Data:* `results/scaling-5090x8.json`, `results/comm-accum{1,2,4,8}.json`.

---

## Phase 3 — finish the shell (a day)

- **`#/architecture`** — currently a placeholder. The config-driven `Transformer`: one class,
  GPT-2 or Llama-style by YAML, with the component table and the parameter-budget figure that
  already exists. Small, and it removes a visible placeholder.
- **`#/tests`** — currently a placeholder. This one is more interesting than it sounds, and it
  is *the* page for goal 2: not "327 tests pass" but *what the tests assert*. RoPE's
  translation invariance to 1e-6. Causality checked bitwise across all ten architecture
  variants. That the decode step takes the *fast* path, not merely the correct one — the test
  whose absence hid a 30% regression behind a green suite. Show a handful of property tests as
  claims, each with the one-line reason it exists.
- **Attention visualizer** — currently a separate static page at `/attention/`, outside the
  React app and the design system. Either bring it inside as a plate or give it the masthead
  and dateline so it stops looking like a different website.

---

## Phase 4 — the 90+ pass (a day)

Ordered by what a reviewer notices first.

- **README top screen.** Currently strong — results before origin story. Check it reads on a
  phone and that the first table is the reproduction, not the status table.
- **A screenshot or two in the README.** A pinned repo is judged from the card and the first
  screen; a broadsheet front page is the single most distinctive thing here and it is currently
  invisible until you click through.
- **Mobile.** A broadsheet layout on a 375px screen is the obvious failure mode. Check the
  chapter rail, the dateline, and every figure.
- **Accessibility.** Keyboard reachability for every interactive figure, `aria-current` on the
  rail, contrast in both themes. Interactive figures with mouse-only controls fail goal 3 for
  a real fraction of readers.
- **Lighthouse / bundle.** The build already warns about chunk size; code-split the explorers
  so the front page is not paying for them.
- **Commit history.** Already real and well-messaged. Worth one pass to confirm no
  `wip`/`fix` noise in the recent stretch.
- **Pinned-repo hygiene.** Description, topics, social preview image, and the site link in the
  About box.

---

## What is explicitly not required

- Anything in [docs/roadmap.md](roadmap.md) — fused dequant kernel, flash-compatible verify
  mask, `QuantEmbedding`, adaptive `k`, batched speculation, multi-node scaling. All scoped,
  none on the critical path.
- Any further GPU spend. Every figure the site needs is already in `results/`.
- A second interconnect. Measured out of relevance: PCIe reaches 95.1% and the fit attributes
  only ~2.8 points to the all-reduce.

---

## Order, and why

```
Phase 0  commit the redesign            minutes   — stops losing work
Phase 1  one number, one source         half day  — defends every later claim
Phase 2  four results plates            the bulk  — this is the project's case
Phase 3  architecture + tests + viz     a day     — removes visible placeholders
Phase 4  mobile, a11y, README, pinning  a day     — what a reviewer sees first
```

Phase 1 before Phase 2 because the plates will quote figures, and quoting them from a
generated module is easier than retrofitting it afterwards. Phase 4 last because polishing a
site that is missing its results section optimises the wrong thing.

The honest summary of the gap: **the repository currently proves more than the site shows.**
Phase 2 closes that, and it is most of the remaining work.
