import { GPT2_124M, formatCount, parameters } from "../lib/modelsize";
import { dotRotated, seededVector } from "../lib/rope";
import { href } from "../router";
import { PROJECT } from "./projectState";

/**
 * The four figures across the top of the front page.
 *
 * Three of them are computed here rather than quoted: the parameter count from the
 * same arithmetic the parameter-budget figure uses, and the RoPE spread by actually
 * sliding the pair. A front page that asserts numbers it does not derive is exactly
 * what the rest of the site is arguing against.
 */

/** The RoPE explorer's own defaults, so its figure and this one cannot disagree. */
export const ROPE_DEFAULTS = { headDim: 16, theta: 10_000, seed: 7, offset: 16, maxPos: 512 };

/**
 * How far the attention logit moves while the pair slides the whole sequence.
 *
 * This is `tests/test_rope.py`'s claim, evaluated: hold the offset, walk both
 * positions from the start of the context to the end, and take the full range of
 * the logit. If the claim were false this would be a visible number.
 */
export function ropeLogitSpread(): { spread: number; samples: number } {
  const { headDim, theta, seed, offset, maxPos } = ROPE_DEFAULTS;
  const q = seededVector(headDim, seed);
  const k = seededVector(headDim, seed + 1000);

  let min = Infinity;
  let max = -Infinity;
  let samples = 0;
  for (let n = 0; n + offset <= maxPos; n++) {
    const logit = dotRotated(q, k, n + offset, n, theta);
    if (logit < min) min = logit;
    if (logit > max) max = logit;
    samples++;
  }
  return { spread: max - min, samples };
}

/**
 * `href` is the page that proves the figure, and lives here rather than in the page so
 * a figure and its evidence cannot be separated by an edit to one of them.
 */
export type Figure = { value: string; label: string; href: string };

export function frontFigures(): Figure[] {
  const { spread, samples } = ropeLogitSpread();
  return [
    {
      value: formatCount(parameters(GPT2_124M).total),
      label: "parameters, exact against the real Transformer",
      href: href({ kind: "architecture" }),
    },
    {
      value: PROJECT.reproduction.loss.toFixed(2),
      label: `measured validation loss, against a pre-registered target of ${PROJECT.reproduction.targetLoss}`,
      href: href({ kind: "reproduction" }),
    },
    {
      value: spread.toExponential(0).replace("e-", "e−"),
      label: `how far the RoPE logit moves across ${samples} positions`,
      href: href({ kind: "rope" }),
    },
    {
      value: String(PROJECT.pythonTests),
      label: "tests, asserting properties rather than shapes",
      href: href({ kind: "tests" }),
    },
  ];
}
