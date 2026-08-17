import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { MEASURED } from "./measured";
import { GPT2_124M, parameters } from "../lib/modelsize";

/**
 * Figures typed into prose, checked against what they describe.
 *
 * Every *generated* figure on this site is regenerated and diffed by
 * `tests/test_web_export.py`, and `check-counts.mjs` covers the one field that job cannot
 * see. Neither reaches a number a sentence spells out — "31% of the entire model", "all 14
 * cases", "over 50 steps" — and those are the ones a reader is most likely to take at face
 * value, because they read as explanation rather than as data.
 *
 * All of them were correct when this file was written. That is the point: the risk is
 * drift, and drift is invisible until someone re-derives the number by hand. Each check
 * below re-derives one, from the artifact or the calculation it came from.
 */

const ROOT = resolve(__dirname, "../../..");
const read = (p: string) => readFileSync(resolve(ROOT, p), "utf8");
const json = (p: string) => JSON.parse(read(p));

const CHAPTERS = read("web/src/content/chapterBodies.tsx");
const CHAPTER_INDEX = read("web/src/content/chapters.ts");
const SCALING_PAGE = read("web/src/pages/Scaling.tsx");
const EFFICIENCY_PAGE = read("web/src/pages/Efficiency.tsx");

describe("the explainer's parameter arithmetic", () => {
  const p = parameters(GPT2_124M);

  it("states the embedding's share as the breakdown computes it", () => {
    const share = `${Math.round((p.tokenEmbedding / p.total) * 100)}%`;
    expect(CHAPTERS).toContain(`${share} of the entire model`);
    expect(CHAPTER_INDEX).toContain(`${share} of a 124M model`);
  });

  it("states the embedding's shape from the config", () => {
    expect(CHAPTERS).toContain(
      `${GPT2_124M.vocabSize.toLocaleString()} × ${GPT2_124M.nEmbd} numbers`,
    );
  });

  it("states the position table's size from the config", () => {
    expect(CHAPTERS).toContain(`${p.positionEmbedding.toLocaleString()} parameters`);
    expect(p.positionEmbedding).toBe(GPT2_124M.blockSize * GPT2_124M.nEmbd);
  });
});

describe("the explainer's figures about the runs", () => {
  it("states the whole useful range of loss, from uniform to the GPT-2 target", () => {
    // "knows nothing" is uniform over GPT-2's vocabulary, ln(50257) ≈ 10.82;
    // "reproduces GPT-2" is the pre-registered 3.29. The run's own 3.05 is past it.
    const range = Math.log(50257) - MEASURED.reproduction.targetLoss;
    expect(range).toBeGreaterThan(7.45);
    expect(range).toBeLessThan(7.55);
    expect(CHAPTERS).toContain("7.5 in loss");
  });

  it("states the ablation scale from the sweep's own records", () => {
    const params = json("results/ablations.json").arms[0].params;
    expect(CHAPTERS).toContain(`${Math.round(params / 1e6)}M parameters`);
  });

  it("states how many tokenizer cases the fixture actually holds", () => {
    const cases = json("web/src/data/tokenizer-fixture.json").cases.length;
    expect(CHAPTERS).toContain(`all ${cases} cases`);
  });
});

describe("the results pages", () => {
  it("states the scaling sweep's step count from the artifact", () => {
    const steps = json("results/scaling-5090x8.json").steps;
    expect(steps).toBe(50);
    expect(SCALING_PAGE).toContain(`over ${steps} steps`);
    expect(SCALING_PAGE).toContain("fifty steps");
  });

  it("states the mask-fix regression as the sweep measured it", () => {
    // The fix bought 1.30–1.38× on the cached path; "a 30% regression" is the smallest
    // of those expressed as what it had cost.
    const gains = MEASURED.cache.points.map((p) => p.gainFromFix);
    const smallest = Math.round((Math.min(...gains) - 1) * 100);
    expect(smallest).toBe(30);
    expect(EFFICIENCY_PAGE).toContain(`${smallest}% regression`);
  });
});
