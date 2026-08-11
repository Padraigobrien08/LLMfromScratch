/**
 * The communication-cost model behind the scaling plate.
 *
 * Scaling loss — the percentage points of efficiency an 8-GPU run gives up against one
 * GPU — decomposes as `a + b/accum`. The intuition is that `a` is the part that does not
 * amortise (kernel launches, the optimiser step, whatever the all-reduce cannot overlap)
 * and `b/accum` is the all-reduce itself, paid once per optimiser step and therefore
 * spread across however many micro-batches accumulate into it.
 *
 * `a` and `b` are fitted in Python, in `llmfs.bench.scaling.fit_amortisation`, to exactly
 * two of the four measured points. This module only *evaluates* the curve, which is what
 * lets the slider ask it about accumulation values nobody ran.
 */

/** Scaling loss in percentage points at a given accumulation. */
export function lossAt(a: number, b: number, accum: number): number {
  return a + b / accum;
}

/** Efficiency as a fraction, the way the artifacts record it. */
export function efficiencyAt(a: number, b: number, accum: number): number {
  return (100 - lossAt(a, b, accum)) / 100;
}

/**
 * Points along the curve for drawing, sampled evenly in `1/accum`.
 *
 * Evenly in `1/accum` rather than in `accum` because that is the axis the model is
 * linear in: sampling uniformly in accum crowds every interesting part of the curve
 * into the first tenth of the range and then spends nine hundred points drawing a
 * straight line.
 */
export function curvePoints(
  a: number,
  b: number,
  minAccum: number,
  maxAccum: number,
  samples = 120,
): Array<{ accum: number; efficiency: number }> {
  const from = 1 / maxAccum;
  const to = 1 / minAccum;
  return Array.from({ length: samples + 1 }, (_, i) => {
    const inv = from + ((to - from) * i) / samples;
    const accum = 1 / inv;
    return { accum, efficiency: efficiencyAt(a, b, accum) };
  });
}

/**
 * How far a measurement sits from what the curve said it would be, in points.
 *
 * Signed, and reported rather than absolute, because the direction matters: the fit
 * predicting *better* efficiency than was measured is a different kind of wrong from
 * predicting worse, and averaging the two away would hide a systematic bias.
 */
export function residual(a: number, b: number, accum: number, measured: number): number {
  return measured * 100 - (100 - lossAt(a, b, accum));
}
