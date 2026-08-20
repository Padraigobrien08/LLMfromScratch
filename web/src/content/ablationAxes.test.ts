import { describe, expect, it } from "vitest";

import { ABLATION_AXES } from "./ablationAxes";
import { AXIS, MODERN_STACK } from "../lib/ablations";
import { TOGGLES } from "../pages/Ablations";

describe("the generated ablation registry", () => {
  it("backs every toggle the page offers, and nothing is missing", () => {
    // The toggles are the single-axis arms: everything in AXIS except the baseline
    // and the combination. A new arm added to the registry without a toggle would
    // be measurable but unreachable; a toggle without an arm would 404 on the data.
    const singleAxisArms = Object.keys(AXIS)
      .filter((name) => name !== "baseline" && name !== "modern-stack")
      .sort();
    expect(TOGGLES.map((t) => t.name).sort()).toEqual(singleAxisArms);
  });

  it("composes the modern stack from arms the registry knows", () => {
    expect(MODERN_STACK.length).toBeGreaterThan(0);
    for (const name of MODERN_STACK) {
      expect(AXIS, `${name} is in the stack but not the registry`).toHaveProperty(name);
    }
    expect(ABLATION_AXES.modernStack).not.toContain("modern-stack");
  });
});
