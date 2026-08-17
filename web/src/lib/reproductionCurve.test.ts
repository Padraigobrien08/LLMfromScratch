import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { MEASURED } from "../content/measured";
import { floorFor } from "../components/LossCurve";
import { type ValPoint, crossingOf, thin, valAt } from "./reproductionCurve";

const VAL: ValPoint[] = [
  { step: 250, loss: 6.0, perplexity: 403 },
  { step: 500, loss: 5.2, perplexity: 181 },
  { step: 750, loss: 3.2, perplexity: 24.5 },
  { step: 1000, loss: 3.1, perplexity: 22.2 },
];

describe("valAt", () => {
  it("reports the most recent measurement, never a future one", () => {
    // 749 is after the 500-step eval and before the 750-step one. Rounding to the
    // nearest would hand back a number that did not exist yet, and on this page that
    // would show the target as met before it was.
    expect(valAt(VAL, 749)?.step).toBe(500);
    expect(valAt(VAL, 750)?.step).toBe(750);
  });

  it("has nothing to report before the first evaluation", () => {
    expect(valAt(VAL, 0)).toBeNull();
    expect(valAt(VAL, 249)).toBeNull();
  });

  it("holds the last measurement past the end of the run", () => {
    expect(valAt(VAL, 10_000)?.step).toBe(1000);
  });
});

describe("crossingOf", () => {
  it("finds the first point to meet the target", () => {
    expect(crossingOf(VAL, 3.29)?.step).toBe(750);
  });

  it("reports nothing when the target was never met", () => {
    expect(crossingOf(VAL, 1.0)).toBeNull();
  });

  /**
   * The generator recorded the crossing in the artifact so the page would not have to
   * compute it. This asserts the two agree — if they ever did not, one of them would be
   * telling a reader the run hit its target at the wrong moment.
   *
   * The comment here used to promise exactly that comparison and the test made only half
   * of it: it checked the recorded crossing was at or below the target, which any point
   * past the crossing also satisfies. Recomputing it from the curve is the check the
   * comment described, and it is the one that would catch an exporter drifting from the
   * page. (They agree: step 6,500, byte-identical.)
   */
  it("agrees with the crossing the exporter recorded", () => {
    const { crossing, targetLoss } = MEASURED.reproduction;
    expect(crossing).not.toBeNull();
    expect(crossing!.loss).toBeLessThanOrEqual(targetLoss);

    // The curve the page actually plots, read from the same artifact the site fetches.
    const curve = JSON.parse(
      readFileSync(resolve(__dirname, "../../../results/reproduction-curve.json"), "utf8"),
    ) as { val: ValPoint[]; targetLoss: number };

    const recomputed = crossingOf(curve.val, curve.targetLoss);
    expect(recomputed).not.toBeNull();
    expect(recomputed!.step).toBe(crossing!.step);
    expect(recomputed!.loss).toBe(crossing!.loss);
  });
});

describe("thin", () => {
  const series = Array.from({ length: 1000 }, (_, i) => i);

  it("leaves a short series alone", () => {
    expect(thin([1, 2, 3], 10)).toEqual([1, 2, 3]);
  });

  it("keeps the last point, because the end of a run is the result", () => {
    const out = thin(series, 100);
    expect(out.length).toBeLessThanOrEqual(101);
    expect(out.at(-1)).toBe(999);
    expect(out[0]).toBe(0);
  });
});

describe("the chart's window", () => {
  it("drops its floor rather than clipping a better run", () => {
    // Y_LO was a constant 2.95. The shipped run's validation curve bottoms out at 3.08 so
    // it fits, but a rerun that trained further would have drawn itself outside the frame
    // — silently, on the one chart whose whole job is showing a curve reach a target.
    const curve = {
      finalStep: 100,
      targetLoss: 3.29,
      train: [{ step: 100, loss: 2.4, mfu: null }],
      val: [{ step: 100, loss: 2.5, perplexity: 12.2 }],
    };
    expect(floorFor(curve as never)).toBeLessThanOrEqual(2.5);

    const shipped = JSON.parse(
      readFileSync(resolve(__dirname, "../../../results/reproduction-curve.json"), "utf8"),
    );
    // The published run keeps the window the static figure uses.
    expect(floorFor(shipped)).toBe(2.95);
  });
});
