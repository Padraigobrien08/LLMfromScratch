import { MEASURED } from "./measured";

/**
 * Live project state, printed on the dateline rail of every page.
 *
 * These are claims about the repository, and this file used to make them as literals
 * with a comment naming what each was read from. The discipline was real and it still
 * failed: `pythonTests: 223` and `browserTests: 69` sat here for weeks after both had
 * moved, because a comment naming a source is not a link to one. Hand-transcription
 * across a language boundary drifts, and nothing written in Python was ever going to
 * notice a stale number written in TypeScript.
 *
 * So everything measurable now comes from `measured.ts`, which `llmfs-export-web`
 * generates from `results/*.json` and live test collection, and which
 * `tests/test_web_export.py` asserts is still current. What is left below is the
 * handful of facts no run produces.
 */
export const PROJECT = {
  author: "Padraig O'Brien",
  licence: "MIT",
  repo: "https://github.com/Padraigobrien08/LLMfromScratch",

  /**
   * `tests/conftest.py::ARCH_VARIANTS` — every property test runs against all ten.
   *
   * The one figure here still written by hand, because it comes from a test fixture
   * rather than from a run, so there is no artifact to generate it from.
   * `test_site_reports_the_real_number_of_architecture_variants` pins it to the
   * fixture's length instead.
   */
  archVariants: 10,

  pythonTests: MEASURED.tests.python,
  browserTests: MEASURED.tests.browser,
  reproduction: MEASURED.reproduction,
  ablations: MEASURED.ablations,
} as const;

/** The five items of the dateline rail, left to right. */
export const DATELINE: string[] = [
  PROJECT.author,
  `${PROJECT.pythonTests} tests green`,
  `GPT-2 124M · val loss ${PROJECT.reproduction.loss.toFixed(2)}`,
  `Ablation sweep complete · ${PROJECT.ablations.runs} runs`,
  PROJECT.licence,
];
