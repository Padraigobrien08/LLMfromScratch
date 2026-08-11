import { describe, expect, it } from "vitest";

import { GPT2_124M, parameters } from "../lib/modelsize";
import { frontFigures, ropeLogitSpread } from "./frontFigures";

describe("front page figures", () => {
  it("prints the parameter count the budget figure computes, not a quoted one", () => {
    expect(parameters(GPT2_124M).total).toBe(124_475_904);
    expect(frontFigures()[0]!.value).toBe("124.5M");
  });

  /**
   * The claim the figure exists to make. If RoPE's logit ever depended on absolute
   * position this would climb off float64 noise and the front page would say so —
   * which is the point of computing it rather than quoting it.
   */
  it("finds the logit stays put while the pair slides the whole context", () => {
    const { spread, samples } = ropeLogitSpread();
    expect(samples).toBe(497);
    expect(spread).toBeLessThan(1e-9);
  });

  it("labels the loss as measured against the pre-registered target", () => {
    const [, loss] = frontFigures();
    expect(loss!.value).toBe("3.05");
    expect(loss!.label).toContain("3.29");
  });
});
