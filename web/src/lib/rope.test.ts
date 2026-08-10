import { describe, expect, it } from "vitest";

import fixture from "../data/rope-fixture.json";
import { dot, dotRelative, dotRotated, rotate, seededVector } from "./rope";

const { q, k, theta, head_dim: headDim, cases } = fixture;

/**
 * The port is only useful if it is the same function as the Python one. Everything
 * else in this file would pass just as happily against a plausible-looking rotation
 * that the models never used.
 */
/**
 * Why the tolerance is 1e-6 and not floating-point epsilon:
 * `RotaryEmbedding._build_tables` computes its cos/sin tables in **float32**, then
 * casts to the activation dtype. This port computes them in float64, so it is
 * strictly the more precise of the two and the gap is the Python side's table
 * quantisation — measured at 1.2e-8 near position 5, growing to 3.0e-7 by position
 * 200 as `pos * inv_freq` consumes more of the fp32 mantissa.
 *
 * That is not a defect to fix. It is the same choice HF Llama makes, and it sits
 * several orders of magnitude below the ~1e-2 resolution of the bf16 activations
 * these tables are consumed by. The bound is asserted rather than assumed, so if
 * the tables ever move to a lower precision this test says so.
 */
const TABLE_PRECISION = 1e-6;

describe("parity with llmfs.model.rope", () => {
  it("reproduces every rotated vector from the Python fixture", () => {
    for (const c of cases) {
      const qRot = rotate(q, c.m, theta);
      const kRot = rotate(k, c.n, theta);
      for (let i = 0; i < headDim; i++) {
        expect(Math.abs(qRot[i]! - c.q_rot[i]!)).toBeLessThan(TABLE_PRECISION);
        expect(Math.abs(kRot[i]! - c.k_rot[i]!)).toBeLessThan(TABLE_PRECISION);
      }
    }
  });

  it("reproduces every attention logit from the Python fixture", () => {
    for (const c of cases) {
      expect(Math.abs(dotRotated(q, k, c.m, c.n, theta) - c.logit)).toBeLessThan(
        TABLE_PRECISION,
      );
    }
  });

  it("matches exactly where the fp32 tables are exact", () => {
    // Position 0 makes every angle 0, so cos/sin are 1 and 0 in any precision.
    // Parity there is unconditional, and isolates a genuine port error from the
    // table quantisation the tolerance above is absorbing.
    const zero = cases.find((c) => c.m === 0 && c.n === 0)!;
    const qRot = rotate(q, 0, theta);
    for (let i = 0; i < headDim; i++) expect(qRot[i]).toBeCloseTo(zero.q_rot[i]!, 15);
  });
});

describe("the property RoPE exists for", () => {
  /**
   * The same assertion as `tests/test_rope.py::test_relative_position_property`, in
   * the language the explorer is written in. If this fails, the page is drawing a
   * claim the model does not honour.
   */
  it("gives a logit that depends only on m - n", () => {
    const qv = seededVector(64, 7);
    const kv = seededVector(64, 99);
    for (const gap of [0, 1, 5, 37]) {
      const reference = dotRotated(qv, kv, gap, 0);
      for (const offset of [1, 4, 17, 128, 999]) {
        expect(dotRotated(qv, kv, gap + offset, offset)).toBeCloseTo(reference, 10);
      }
    }
  });

  it("agrees with the closed form that never forms an absolute position", () => {
    const qv = seededVector(32, 3);
    const kv = seededVector(32, 4);
    for (const [m, n] of [
      [0, 0],
      [9, 2],
      [400, 397],
      [2, 9], // negative offsets: keys ahead of the query, as cross-attention would see
    ] as const) {
      expect(dotRelative(qv, kv, m - n)).toBeCloseTo(dotRotated(qv, kv, m, n), 10);
    }
  });

  it("is a rotation, so it never changes a vector's length", () => {
    const v = seededVector(48, 11);
    const before = Math.sqrt(dot(v, v));
    for (const pos of [0, 1, 63, 4096]) {
      const after = rotate(v, pos);
      expect(Math.sqrt(dot(after, after))).toBeCloseTo(before, 12);
    }
  });

  it("leaves position 0 untouched", () => {
    const v = seededVector(16, 5);
    const rotated = rotate(v, 0);
    for (let i = 0; i < v.length; i++) expect(rotated[i]).toBeCloseTo(v[i]!, 15);
  });
});

describe("seededVector", () => {
  it("is deterministic, so a shared link shows the same picture", () => {
    expect(Array.from(seededVector(16, 42))).toEqual(Array.from(seededVector(16, 42)));
    expect(Array.from(seededVector(16, 42))).not.toEqual(Array.from(seededVector(16, 43)));
  });
});
