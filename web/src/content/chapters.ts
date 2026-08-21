/**
 * The eight chapters, as the front page's path and the chapter rail both need them.
 *
 * `kicker` is the subject alone, which is what the path prints; `dateline` is the
 * full "Chapter one · tokenization" a chapter page prints above its headline.
 */
export type Chapter = {
  n: number;
  kicker: string;
  /**
   * What the chapter concludes — the sentence a reader remembers it by.
   *
   * Still the label everywhere the path is navigated: the rail, the chapter feet, the
   * front page. On the page itself it is no longer the opening line but the answer,
   * printed where the argument reaches it.
   */
  title: string;
  /**
   * What the chapter asks. This is the page's opening move.
   *
   * The titles are good sentences and bad openings: "Letting every token look back"
   * tells a reader who already knows about attention which chapter they are on, and
   * tells everyone else nothing they can be curious about. A chapter that opens on its
   * own question gives the reader something to be wrong about first, which is the only
   * way the answer lands as an answer.
   */
  question: string;
  /**
   * The one sentence of chain that makes the question inevitable — what has been
   * established, and what it still cannot do. Absent on the first chapter, which has no
   * "so far".
   *
   * Deliberately not a summary of every previous chapter: only the immediately relevant
   * link, or it becomes an index nobody reads. One sentence rather than a stack of lines,
   * because it is set inline behind its own lead like `Caveat.` and `Provenance.` — the
   * page already has furniture for a quiet aside in its own voice, and a second pattern
   * for the same job would just be a box.
   */
  storySoFar?: string;
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
    question: "How can a language model read English if it never sees your words?",
  },
  {
    n: 2,
    kicker: "embeddings",
    title: "Each token becomes a list of numbers",
    question: "Why do words have to become numbers?",
    storySoFar:
      "Text is now a list of ids. But an id is an index, and an index cannot be added, scaled or compared.",
  },
  {
    n: 3,
    kicker: "the problem",
    title: "Words only mean things in context",
    question: "Why can't a model understand words one at a time?",
    storySoFar:
      "Every token is now a vector the model can compute with. Each one was looked up alone, knowing nothing of the words beside it.",
  },
  {
    n: 4,
    kicker: "attention",
    title: "Every token looks back",
    question: "How can one word influence another?",
    storySoFar:
      "Meaning depends on words that came earlier, so the model needs a way for tokens to exchange information.",
  },
  {
    n: 5,
    kicker: "position",
    title: "The model knows where each token sits",
    question: "How does the model know which word came first?",
    storySoFar:
      "Attention lets every token look at every earlier one, but a weighted average has no inherent notion of order.",
  },
  {
    n: 6,
    kicker: "sampling",
    title: "Out comes a probability for every token",
    question: "How does a model decide what to write next?",
    storySoFar:
      "The model now has representations carrying both content and position, and no way yet to turn either into a word.",
  },
  {
    n: 7,
    kicker: "loss",
    title: "Training is making the right token less surprising",
    question: "What does it actually mean for a model to learn?",
    storySoFar:
      "The model can put a probability on every token in the vocabulary. Whether that distribution was any good is what training has to decide.",
  },
  {
    n: 8,
    kicker: "the honest question",
    title: "The optimiser mattered more than the architecture",
    question: "Do the design choices actually change anything?",
    storySoFar:
      "Every chapter above settled a design decision and the papers all report improvements, none of which is evidence until it survives being measured.",
  },
];

export const chapter = (n: number): Chapter => CHAPTERS[n - 1] ?? CHAPTERS[0]!;

/** "Chapter one · tokenization", the kicker a chapter page prints. */
export const chapterDateline = (c: Chapter): string =>
  `Chapter ${ORDINALS[c.n - 1]} · ${c.kicker}`;
