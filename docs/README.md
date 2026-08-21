# Documentation index

## The end goal

A decoder-only language model written from scratch, **reproducing GPT-2 124M** and then used
as the subject of a measurement study (architecture, optimisation, inference efficiency and
multi-GPU scaling) where every claim is traceable to an artifact rather than to prose.

The end state has two halves, and the second is the one that makes this more than a training
script:

1. **A reproduction anyone can check.** A validation-loss target fixed before the run, hit and
   independently corroborated on a public benchmark, from one command on one rented GPU.
2. **An interactive site that *is* the paper**, not documentation of a paper living elsewhere.
   The measurements below become plates you can interrogate: drag the accumulation slider and
   watch the communication cost curve; toggle the mask bug and watch the KV cache go from
   losing to winning, with the same figures a researcher would check and an explainer path a
   reader with no background can follow end to end.

The documents in this directory are the written half. They are the source the site's results
plates are built from, and they are held to one rule: **no number appears here that is not read
from `results/`.** `tests/test_documented_results.py` enforces the headline figures against
those artifacts, this index included; a figure edited in prose without a re-run behind it
fails CI rather than quietly becoming false.

**Current state.** All seven measurement pillars are done and written up, and all four are now
plates on the site, each built around one interactive figure that carries its argument, because
a claim you can move is harder to arrange after the fact than a claim you can only read. Drag the
scrubber along the reproduction and watch the target line get crossed a third of the way in;
switch the mask bug back on and watch the KV cache change sides; slide gradient accumulation and
watch a two-parameter model land on points it never saw.

Two further pages answer the question that follows, *is any of this actually held down?* One walks
the architecture block by block, naming the test behind each claim and saying plainly where there
is no test to name; the other shows what a dozen of those tests assert and the bug each exists to
catch, collected from the tests themselves rather than typed.

These documents remain the long form: the protocol, the provenance, the caveats and the reasoning
each page compresses. [The repository README](../README.md) carries the honest status table.

---

## The documents

| Document | The question it answers |
| --- | --- |
| [reproduction.md](reproduction.md) | Did it actually reproduce GPT-2 124M? Protocol, target provenance, hardware, samples. Val loss **3.0503** against a pre-registered ≤ 3.29; HellaSwag **0.3043** against the published 0.2955. |
| [ablations.md](ablations.md) | Which design decisions matter? Twelve arms × three seeds, paired. **The optimiser dominates the architecture**: learning rate is worth −0.1251, RMSNorm +0.0007. |
| [efficiency.md](efficiency.md) | How fast can it run, and what did the benchmark catch? The KV cache was **34% slower than recomputation** until a mask that forfeited the fused kernel was found. Plus quantization and speculative decoding. |
| [scaling.md](scaling.md) | Does it scale, and what does the interconnect cost? **95.1% on 8 GPUs with no NVLink**, and a two-point fit that predicted the rest of the accumulation sweep before it was measured. |
| [fault-tolerance.md](fault-tolerance.md) | What happens when the hardware fails? Design doc: failure taxonomy, checkpointing, resumption, silent corruption, and what breaks at 1,000+ GPUs. |
| [gpu-runbook.md](gpu-runbook.md) | How do you actually run this on rented hardware without wasting money? Which GPU, what it costs, and the `scripts/gpu.sh` workflow. |
| [roadmap.md](roadmap.md) | What was deliberately *not* built? Seven scoped items (fused dequant kernel, flash-compatible verify mask, multi-node), each with what to learn first and how you would know it worked. |
| [front-page-hierarchy.md](front-page-hierarchy.md) | What is still wrong with the front page, after the type scale was fixed? Six ranked items (the loudest element is a footnote, the same figures printed twice, three route buttons pointing at the same hrefs as three nav items), plus the fit constraints any change has to respect and the one containment measurement I could not settle. |

**Start with [reproduction.md](reproduction.md)** if you want the trust anchor, or
[scaling.md](scaling.md) if you want the most recent work and the best single example of the
standard the rest is held to.

---

## Two conventions worth knowing before you read

**Every figure is read from an artifact.** `results/*.json` holds the measured output of each
run: `reproduction.json`, `ablations.json`, `scaling-5090x8.json`, `comm-accum{1,2,4,8}.json`,
the quantization and speculative sweeps, each carrying its own provenance block (GPU, arch,
driver, torch build, measured TFLOP/s, commit). Nothing in these documents is retyped from
memory, and the test suite checks that.

The same artifacts cross into the site as generated code: `llmfs-export-web` writes
`web/src/content/measured.ts`, and CI fails if the committed module is not what the generator
emits, so a page and a document cannot disagree about a number they both quote.

**Design docs mark what is real.** [fault-tolerance.md](fault-tolerance.md) tags every claim
**[implemented]**, naming the test that pins it, or **[proposed]**, with an effort estimate.
A design document that reads as though it describes running code, when it describes an
intention, is the failure mode that convention exists to prevent.

The same instinct runs through the rest: a result whose seeds disagree is reported as *not a
result*, an MFU column with no trustworthy peak to divide by is left empty rather than filled
with a vendor figure that measurement contradicts, and the mistakes (the KV cache explained
away, the learning-rate prediction that was wrong) are written up where they happened.
