/**
 * Live project state, printed on the dateline rail of every page.
 *
 * These are claims about the repository, so each one names what it is read from. The
 * rule the whole site is built on is that it may never claim more than the README —
 * so when one of these moves, it moves here, once.
 */
export const PROJECT = {
  author: "Padraig O'Brien",
  licence: "MIT",
  repo: "https://github.com/Padraigobrien08/LLMfromScratch",

  /** `pytest tests -q` — 223 collected across thirteen files. Generated in stage 6. */
  pythonTests: 223,

  /** `npm test --prefix web` — the ports of rope, modelsize, sampling and tokenizer. */
  browserTests: 69,

  /** `tests/conftest.py::ARCH_VARIANTS` — every property test runs against all ten. */
  archVariants: 10,

  /**
   * `results/reproduction.json`: the measured validation loss of the GPT-2 124M run
   * at step 19,000, against the 3.29 target pre-registered in `configs/gpt2-124m.yaml`
   * and `docs/reproduction.md`.
   */
  reproduction: { loss: 3.05, targetLoss: 3.29, step: 19_000 },

  /** `docs/ablations.md` / `results/ablations.json`: the sweep, now run. */
  ablations: { arms: 12, seeds: 3, runs: 39, gpuHours: 7.6 },
} as const;

/** The five items of the dateline rail, left to right. */
export const DATELINE: string[] = [
  PROJECT.author,
  `${PROJECT.pythonTests} tests green`,
  `GPT-2 124M · val loss ${PROJECT.reproduction.loss.toFixed(2)}`,
  `Ablation sweep complete · ${PROJECT.ablations.runs} runs`,
  PROJECT.licence,
];
