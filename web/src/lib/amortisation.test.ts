import { describe, expect, it } from "vitest";

import { MEASURED } from "../content/measured";
import { curvePoints, efficiencyAt, lossAt, residual } from "./amortisation";

const { a, b, fittedFrom } = MEASURED.accumulation.fit;
const POINTS = MEASURED.accumulation.points;

describe("the amortisation curve", () => {
  /**
   * The cross-language pin, and the reason this file exists.
   *
   * `predictedEfficiency` on each point was computed by Python, from the fit Python
   * produced. This evaluates the same model in TypeScript and asserts the two agree.
   * Without it the slider could quietly draw a curve the repository's own scaling
   * report would not recognise — the same failure mode the RoPE port is pinned against.
   */
  it("reproduces the Python-computed prediction at every measured accumulation", () => {
    for (const point of POINTS) {
      expect(efficiencyAt(a, b, point.accum)).toBeCloseTo(point.predictedEfficiency, 12);
    }
  });

  /**
   * The whole argument of the plate: the fit saw two points and landed on the other two.
   * If it were fitted to all four this would be a curve through data rather than a test,
   * so the claim worth asserting is that it passes *exactly* through the two it used.
   */
  it("passes exactly through the two points it was fitted to", () => {
    for (const accum of fittedFrom) {
      const point = POINTS.find((p) => p.accum === accum);
      expect(point, `no measured point at accum ${accum}`).toBeDefined();
      expect(point!.predicted).toBe(false);
      expect(residual(a, b, accum, point!.efficiency)).toBeCloseTo(0, 10);
    }
  });

  it("predicts the two it did not see to within a point", () => {
    const predicted = POINTS.filter((p) => p.predicted);
    expect(predicted).toHaveLength(2);
    for (const point of predicted) {
      expect(Math.abs(residual(a, b, point.accum, point.efficiency))).toBeLessThan(1);
    }
  });

  it("costs more efficiency as the all-reduce is spread over less work", () => {
    expect(lossAt(a, b, 1)).toBeGreaterThan(lossAt(a, b, 8));
    // The asymptote is the part that does not amortise, which is the point of the split.
    expect(efficiencyAt(a, b, 1e9)).toBeCloseTo((100 - a) / 100, 9);
  });

  it("samples the curve evenly in 1/accum, so the knee is not drawn as a straight line", () => {
    const points = curvePoints(a, b, 1, 8, 4);
    expect(points[0]!.accum).toBeCloseTo(8, 9);
    expect(points.at(-1)!.accum).toBeCloseTo(1, 9);
    const inverses = points.map((p) => 1 / p.accum);
    const gaps = inverses.slice(1).map((v, i) => v - inverses[i]!);
    for (const gap of gaps) expect(gap).toBeCloseTo(gaps[0]!, 12);
  });
});
