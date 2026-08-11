/**
 * The reproduction run's loss curve, as the plate reads it.
 *
 * The artifact is `results/reproduction-curve.json`, lifted out of the run's own
 * `metrics.jsonl` by `scripts/dump_reproduction_curve.py`. Two series at very different
 * resolutions — a train loss every ten steps, a validation loss every 250 — which is why
 * the scrubber reports them differently: the train series is what the run *felt* like,
 * and the validation series is what it is judged on.
 */

export type TrainPoint = { step: number; loss: number; mfu: number | null };
export type ValPoint = { step: number; loss: number; perplexity: number };

export type Curve = {
  targetLoss: number;
  finalStep: number;
  tokens: number;
  crossing: (ValPoint & { fractionOfRun: number }) | null;
  train: TrainPoint[];
  val: ValPoint[];
};

/**
 * The most recent validation point at or before a step.
 *
 * At-or-before rather than nearest, deliberately. The scrubber is a claim about what was
 * known *by* that step, and rounding forward to a measurement that had not been taken
 * yet would let the page show the target as met slightly before it was.
 */
export function valAt<T extends { step: number }>(points: T[], step: number): T | null {
  let found: T | null = null;
  for (const point of points) {
    if (point.step > step) break;
    found = point;
  }
  return found;
}

/**
 * Thin a series to roughly one point per pixel column.
 *
 * Nearly two thousand training points across a thousand-unit viewBox is more vertices
 * than the path can resolve; the surplus renders as a slightly fuzzier line and a
 * slightly slower drag, and buys nothing.
 */
export function thin<T>(points: T[], limit: number): T[] {
  if (points.length <= limit) return points;
  const stride = Math.ceil(points.length / limit);
  const out = points.filter((_, i) => i % stride === 0);
  const last = points.at(-1)!;
  if (out.at(-1) !== last) out.push(last);
  return out;
}

/** The first validation point to meet the pre-registered target, if any did. */
export function crossingOf(val: ValPoint[], targetLoss: number): ValPoint | null {
  return val.find((point) => point.loss <= targetLoss) ?? null;
}
