/**
 * The eight chapters, as the front page's path and the chapter rail both need them.
 *
 * `kicker` is the subject alone, which is what the path prints; `dateline` is the
 * full "Chapter one · tokenization" a chapter page prints above its headline.
 */
export type Chapter = {
  n: number;
  kicker: string;
  title: string;
  /** The one-line case for reading it, printed on the front page. */
  blurb: string;
};

const ORDINALS = [
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
] as const;

export const CHAPTERS: Chapter[] = [
  {
    n: 1,
    kicker: "tokenization",
    title: "A model never sees your text",
    blurb:
      "Type a sentence and watch the real GPT-2 vocabulary cut it up. The ids are the only thing the model ever sees.",
  },
  {
    n: 2,
    kicker: "embeddings",
    title: "Each token becomes a list of numbers",
    blurb:
      "Drag layers and width and watch 31% of a 124M model turn out to be a lookup table before any computation happens.",
  },
  {
    n: 3,
    kicker: "the problem",
    title: "Words only mean things in context",
    blurb:
      "Three sentences whose meaning lives in an earlier word. The problem attention exists to solve, stated before the solution.",
  },
  {
    n: 4,
    kicker: "attention",
    title: "Letting every token look back",
    blurb:
      "Queries, keys, and a weighted average — and the catch that a weighted average cannot tell you what order anything came in.",
  },
  {
    n: 5,
    kicker: "position",
    title: "Telling the model where each token sits",
    blurb:
      "Learned position tables versus rotation. The chapter that hands you off to the explorer.",
  },
  {
    n: 6,
    kicker: "sampling",
    title: "Out comes a probability for every token",
    blurb:
      "A real next-token distribution counted from the corpus, with temperature, top-k and top-p in the sampler's own order.",
  },
  {
    n: 7,
    kicker: "loss",
    title: "Training is making the right token less surprising",
    blurb:
      "A validation loss, translated into how many equally likely options the model is choosing between.",
  },
  {
    n: 8,
    kicker: "the honest question",
    title: "So does any of the design actually matter?",
    blurb:
      "Whether any of it survives being measured against the noise of changing the random seed.",
  },
];

export const chapter = (n: number): Chapter => CHAPTERS[n - 1] ?? CHAPTERS[0]!;

/** "Chapter one · tokenization", the kicker a chapter page prints. */
export const chapterDateline = (c: Chapter): string =>
  `Chapter ${ORDINALS[c.n - 1]} · ${c.kicker}`;
