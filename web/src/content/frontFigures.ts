import { GPT2_124M, formatCount, parameters } from "../lib/modelsize";
import { href } from "../router";
import { PROJECT } from "./projectState";

/**
 * The four figures under the front page's standfirst.
 *
 * The page claims the project is worth taking seriously in the second or two before a
 * reader decides whether to scroll. These are what carry that: a parameter count that
 * matches the real Transformer, a loss measured against a target fixed in advance, the
 * size of the suite, and the size of the sweep.
 *
 * They ran here once and were cut, on the grounds that the dateline rail above prints the
 * same numbers on every page. It does — at 11.5px, tracked, in the furniture. That is a
 * masthead's job, not evidence's: a reader scanning for whether this is a toy skips it.
 * The rail states the project's status; this states its scale, at reading size, once.
 *
 * `href` is the page that proves the figure and lives here rather than in the page, so a
 * figure and its evidence cannot be separated by an edit to one of them. Every value is
 * derived — the parameter count from the same arithmetic the budget figure uses, the rest
 * from the generated `measured.ts` — because a front page that asserts numbers it does not
 * derive is what the rest of the site argues against.
 */
export type Figure = { value: string; label: string; href: string };

export function frontFigures(): Figure[] {
  return [
    {
      value: formatCount(parameters(GPT2_124M).total),
      label: "parameters",
      href: href({ kind: "architecture" }),
    },
    {
      value: PROJECT.reproduction.loss.toFixed(2),
      label: "validation loss",
      href: href({ kind: "reproduction" }),
    },
    {
      value: String(PROJECT.pythonTests),
      label: "tests",
      href: href({ kind: "tests" }),
    },
    {
      value: String(PROJECT.ablations.runs),
      label: "ablation runs",
      href: href({ kind: "ablations" }),
    },
  ];
}
