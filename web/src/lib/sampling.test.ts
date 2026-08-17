import { describe, expect, it } from "vitest";

import { type Candidate, perplexity, sample, scoreCandidates, uniformLoss } from "./sampling";

const candidates: Candidate[] = [
  { id: 1, text: " cat", count: 50 },
  { id: 2, text: " dog", count: 30 },
  { id: 3, text: " house", count: 15 },
  { id: 4, text: " zebra", count: 5 },
];

const all = { temperature: 1, topK: null, topP: null };
const probs = (o: Parameters<typeof scoreCandidates>[1]) =>
  scoreCandidates(candidates, o).map((s) => s.prob);

describe("temperature", () => {
  it("reproduces the empirical frequencies exactly at temperature 1", () => {
    // softmax(log(count)) is count/total — no fitting, so what the page shows at
    // temperature 1 is the corpus statistic itself.
    expect(probs(all)).toEqual([0.5, 0.3, 0.15, 0.05].map((p) => expect.closeTo(p, 12)));
  });

  it("collapses onto the single most likely token as it approaches zero", () => {
    expect(probs({ ...all, temperature: 0 })).toEqual([1, 0, 0, 0]);
    const cold = probs({ ...all, temperature: 0.05 });
    expect(cold[0]).toBeGreaterThan(0.999);
  });

  it("flattens towards uniform as it rises", () => {
    const hot = probs({ ...all, temperature: 100 });
    for (const p of hot) expect(p).toBeCloseTo(0.25, 2);
  });

  it("preserves the ranking at every temperature", () => {
    for (const temperature of [0.1, 0.7, 1, 2, 10]) {
      const p = probs({ ...all, temperature });
      expect([...p].sort((a, b) => b - a)).toEqual(p);
    }
  });
});

describe("top-k", () => {
  it("keeps exactly k candidates and renormalises them", () => {
    const scored = scoreCandidates(candidates, { ...all, topK: 2 });
    expect(scored.filter((s) => s.kept).map((s) => s.text)).toEqual([" cat", " dog"]);
    expect(scored[0]!.prob + scored[1]!.prob).toBeCloseTo(1, 12);
    // 0.5 and 0.3 renormalise to 0.625 and 0.375 — the ratio survives, the mass does not.
    expect(scored[0]!.prob).toBeCloseTo(0.625, 12);
    expect(scored[3]!.prob).toBe(0);
  });

  it("is a no-op when k exceeds the candidate count", () => {
    expect(probs({ ...all, topK: 99 })).toEqual(probs(all));
  });

  it("keeps every token tied with the k-th, the way torch does", () => {
    // `transformer.py:222` masks `logits < kth`, so a tie at the cutoff survives whole.
    // Truncating at rank k instead kept an arbitrary one of the tied tokens — arbitrary
    // because the sort does not define an order between equal probabilities — and this
    // page samples bigram counts, where ties are ordinary. Torch gives 50/30/30 the
    // shares below; the rank rule gave [0.625, 0.375, 0, 0].
    const tied: Candidate[] = [
      { id: 1, text: " cat", count: 50 },
      { id: 2, text: " dog", count: 30 },
      { id: 3, text: " fox", count: 30 },
      { id: 4, text: " zebra", count: 5 },
    ];
    const scored = scoreCandidates(tied, { ...all, topK: 2 });
    expect(scored.filter((s) => s.kept).map((s) => s.text)).toEqual([" cat", " dog", " fox"]);
    expect(scored.map((s) => s.prob)).toEqual(
      [50 / 110, 30 / 110, 30 / 110, 0].map((p) => expect.closeTo(p, 12)),
    );
  });

  it("drops everything at k = 0", () => {
    expect(scoreCandidates(candidates, { ...all, topK: 0 }).some((s) => s.kept)).toBe(false);
  });
});

describe("top-p", () => {
  it("keeps the token that crosses the threshold, not just those below it", () => {
    // Cumulative before " dog" is 0.5, which is under 0.6, so " dog" is kept even
    // though including it takes the mass to 0.8. Dropping it is the off-by-one that
    // makes top_p = 0 sample from nothing.
    const scored = scoreCandidates(candidates, { ...all, topP: 0.6 });
    expect(scored.filter((s) => s.kept).map((s) => s.text)).toEqual([" cat", " dog"]);
  });

  it("always keeps at least one token, even at p = 0", () => {
    const scored = scoreCandidates(candidates, { ...all, topP: 0 });
    const kept = scored.filter((s) => s.kept);
    expect(kept).toHaveLength(1);
    expect(kept[0]!.text).toBe(" cat");
    expect(kept[0]!.prob).toBeCloseTo(1, 12);
  });

  it("keeps everything at p = 1", () => {
    expect(scoreCandidates(candidates, { ...all, topP: 1 }).every((s) => s.kept)).toBe(true);
  });

  it("applies after top-k, on the already-truncated set", () => {
    // top-k=2 leaves 0.625/0.375; p=0.6 then keeps both, because the mass before
    // " dog" is 0.625 measured on the renormalised set... and is not under 0.6.
    const scored = scoreCandidates(candidates, { ...all, topK: 2, topP: 0.6 });
    expect(scored.filter((s) => s.kept).map((s) => s.text)).toEqual([" cat"]);
  });
});

describe("sample", () => {
  it("selects by cumulative probability", () => {
    const scored = scoreCandidates(candidates, all);
    expect(sample(scored, 0.0)!.text).toBe(" cat");
    expect(sample(scored, 0.49)!.text).toBe(" cat");
    expect(sample(scored, 0.51)!.text).toBe(" dog");
    expect(sample(scored, 0.99)!.text).toBe(" zebra");
  });

  it("never returns a token the cutoffs dropped", () => {
    const scored = scoreCandidates(candidates, { ...all, topK: 1 });
    for (const u of [0, 0.25, 0.5, 0.75, 0.999]) {
      expect(sample(scored, u)!.text).toBe(" cat");
    }
  });
});

describe("perplexity", () => {
  it("turns a loss into a number of equally likely options", () => {
    expect(perplexity(0)).toBeCloseTo(1, 12);
    expect(perplexity(Math.log(27))).toBeCloseTo(27, 10);
  });

  it("puts the uniform-guess baseline where information theory says", () => {
    // A model that has learned nothing about GPT-2's vocabulary scores ln(50257).
    expect(uniformLoss(50257)).toBeCloseTo(10.82, 2);
    expect(perplexity(uniformLoss(50257))).toBeCloseTo(50257, 6);
  });
});
