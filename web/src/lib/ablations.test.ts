import { describe, expect, it } from "vitest";

import {
  type ArmRun,
  type Payload,
  MODERN_STACK,
  compare,
  groupByName,
  meanCurve,
  resolveSelection,
} from "./ablations";

function run(name: string, seed: number, valLoss: number | null, extra: Partial<ArmRun> = {}): ArmRun {
  return {
    name,
    seed,
    status: valLoss === null ? "diverged" : "completed",
    val_loss: valLoss,
    perplexity: null,
    steps: 2000,
    tokens: 500_000_000,
    wall_clock_s: 3600,
    tokens_per_sec: 140_000,
    params: 42_000_000,
    error: null,
    history: [],
    ...extra,
  };
}

const payload = (arms: ArmRun[]): Payload => ({ meta: {}, arms });

describe("pairing", () => {
  /**
   * The case the whole study design exists for.
   *
   * The baseline moves by 0.10 across seeds, and the arm improves every seed by
   * 0.02. Compared as means that difference vanishes inside the spread; compared
   * within each seed it is unmistakable, because both runs shared the batch order
   * that produced the spread in the first place.
   */
  it("resolves an effect far smaller than the baseline's seed spread", () => {
    const { rows, baseline } = compare(
      payload([
        run("baseline", 1, 3.4),
        run("baseline", 2, 3.5),
        run("baseline", 3, 3.45),
        run("norm-rmsnorm", 1, 3.38),
        run("norm-rmsnorm", 2, 3.48),
        run("norm-rmsnorm", 3, 3.43),
      ]),
    );

    expect(baseline.spread).toBeCloseTo(0.1, 10);
    const arm = rows.find((r) => r.name === "norm-rmsnorm")!;
    expect(arm.paired).toBe(true);
    expect(arm.delta).toBeCloseTo(-0.02, 10);
    // Every seed agreed, so it counts — even though |delta| is a fifth of the spread
    // and an unpaired test would have thrown it away.
    expect(arm.significant).toBe(true);
    expect(Math.abs(arm.delta!) < baseline.spread).toBe(true);
  });

  it("refuses an effect whose sign the seeds disagree about", () => {
    const { rows } = compare(
      payload([
        run("baseline", 1, 3.4),
        run("baseline", 2, 3.5),
        run("baseline", 3, 3.45),
        run("no-bias", 1, 3.39),
        run("no-bias", 2, 3.52),
        run("no-bias", 3, 3.44),
      ]),
    );
    const arm = rows.find((r) => r.name === "no-bias")!;
    expect(arm.deltas).toEqual([-0.01, 0.02, -0.01].map((d) => expect.closeTo(d, 10)));
    expect(arm.significant).toBe(false);
  });

  it("falls back to the unpaired bar when only one seed is shared", () => {
    const { rows } = compare(
      payload([run("baseline", 1, 3.4), run("baseline", 2, 3.5), run("gqa-2", 1, 3.39)]),
    );
    const arm = rows.find((r) => r.name === "gqa-2")!;
    expect(arm.paired).toBe(false);
    // 0.01 does not clear the 0.10 baseline spread, so it is not a result.
    expect(arm.significant).toBe(false);
  });

  it("keeps a diverged arm as a finding rather than dropping it", () => {
    const { rows } = compare(
      payload([run("baseline", 1, 3.4), run("lr-3e-3", 1, null), run("lr-3e-3", 2, null)]),
    );
    const arm = rows.find((r) => r.name === "lr-3e-3")!;
    expect(arm.status).toBe("diverged");
    expect(arm.valLoss).toBeNull();
    expect(arm.delta).toBeNull();
    expect(arm.significant).toBe(false);
  });

  it("excludes a diverged seed from an otherwise completed arm's mean", () => {
    const { rows } = compare(
      payload([
        run("baseline", 1, 3.4),
        run("baseline", 2, 3.4),
        run("wd-zero", 1, 3.3),
        run("wd-zero", 2, null),
      ]),
    );
    const arm = rows.find((r) => r.name === "wd-zero")!;
    expect(arm.nSeeds).toBe(1);
    expect(arm.valLoss).toBeCloseTo(3.3, 10);
  });
});

describe("meanCurve", () => {
  it("averages only the steps every seed reached", () => {
    const runs = [
      run("baseline", 1, 3.4, {
        history: [
          { step: 100, val_loss: 4.0 },
          { step: 200, val_loss: 3.5 },
        ],
      }),
      run("baseline", 2, 3.5, {
        history: [
          { step: 100, val_loss: 4.2 },
          { step: 200, val_loss: 3.7 },
          { step: 300, val_loss: 3.4 },
        ],
      }),
    ];
    const curve = meanCurve(runs);
    // Step 300 is dropped: averaging one run into a two-run curve would silently
    // change what the tail of the line means.
    expect(curve.map((p) => p.step)).toEqual([100, 200]);
    expect(curve[0]!.loss).toBeCloseTo(4.1, 10);
  });

  it("ignores a diverged run entirely", () => {
    const runs = [
      run("lr-3e-3", 1, null, { history: [{ step: 100, val_loss: 9.9 }] }),
      run("lr-3e-3", 2, 3.4, { history: [{ step: 100, val_loss: 4.0 }] }),
    ];
    expect(meanCurve(runs)[0]!.loss).toBeCloseTo(4.0, 10);
  });
});

describe("resolveSelection", () => {
  it("maps no toggles to the baseline and one to that arm", () => {
    expect(resolveSelection([])).toEqual({ kind: "baseline" });
    expect(resolveSelection(["pos-rope"])).toEqual({ kind: "arm", name: "pos-rope" });
  });

  it("recognises exactly the modern-stack set, in any order", () => {
    expect(resolveSelection([...MODERN_STACK].reverse())).toEqual({
      kind: "combination",
      name: "modern-stack",
    });
  });

  it("calls any other combination unmeasured rather than predicting it", () => {
    expect(resolveSelection(["pos-rope", "mlp-swiglu"]).kind).toBe("unmeasured");
    // A subset of modern-stack is still not modern-stack.
    expect(resolveSelection(MODERN_STACK.slice(0, 4)).kind).toBe("unmeasured");
  });
});

describe("groupByName", () => {
  it("collects every seed of an arm together", () => {
    const grouped = groupByName([run("baseline", 1, 3.4), run("baseline", 2, 3.5)]);
    expect(grouped.get("baseline")).toHaveLength(2);
  });
});
