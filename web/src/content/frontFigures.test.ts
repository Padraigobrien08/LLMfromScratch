import { describe, expect, it } from "vitest";

import { GPT2_124M, parameters } from "../lib/modelsize";
import { frontFigures } from "./frontFigures";
import { MEASURED } from "./measured";

describe("front page figures", () => {
  it("prints the parameter count the budget figure computes, not a quoted one", () => {
    expect(parameters(GPT2_124M).total).toBe(124_475_904);
    expect(frontFigures()[0]!.value).toBe("124.5M");
  });

  it("reads the loss, the suite and the sweep from the measured export", () => {
    const [, loss, tests, runs] = frontFigures();
    expect(loss!.value).toBe(MEASURED.reproduction.loss.toFixed(2));
    expect(tests!.value).toBe(String(MEASURED.tests.python));
    expect(runs!.value).toBe(String(MEASURED.ablations.runs));
  });

  /**
   * The pairing is the point. A figure with no page behind it is an assertion, which is
   * the one thing this site is built not to make.
   */
  it("gives every figure a different page that proves it", () => {
    const figures = frontFigures();
    expect(figures).toHaveLength(4);
    expect(new Set(figures.map((f) => f.href)).size).toBe(4);
    for (const f of figures) expect(f.href, `${f.label} links nowhere`).toMatch(/^#\/\w/);
  });
});
