import { type Route, href } from "../router";
import { CHAPTERS } from "./chapters";
import { MEASURED } from "./measured";

/**
 * The shape of the argument: three destinations, and the four plates one of them holds.
 *
 * Eight chapters explaining what a language model is, then the four measured results —
 * which are the paper's body, not an appendix to it — then the explorers for anyone who
 * wants to take a mechanism apart.
 *
 * Results before explorers is a deliberate ordering. A reader who has finished the
 * chapters has been told how a transformer works; what they have not yet been shown is
 * whether any of it was built correctly, and that is the question the four plates
 * answer.
 *
 * The four plates are numbered I–IV here and nowhere else. Their pages print the
 * numeral in their own kickers and their feet chain in this order, both read from
 * `PLATES` — so reordering this array reorders the paper, and no page can disagree
 * with the front one about which plate it is.
 */
export type PlateKind = "reproduction" | "ablations" | "efficiency" | "scaling";

const ROMAN = ["I", "II", "III", "IV"] as const;

type PlateSource = {
  kind: PlateKind;
  /** The half of the kicker that follows "Plate II — Measured results ·". */
  subject: string;
  /** What every link to this plate calls it: the front page's path row, and the feet. */
  title: string;
  /**
   * The page's own headline, where it is deliberately not the title.
   *
   * Absent means the page prints `title` — from here, rather than from a second copy
   * in its own JSX. Two copies is how plates III and IV came to be advertised as
   * "the bug in the numbers" and "the cost of talking" by links that landed on pages
   * headlined something else; a reader following a link could not confirm they had
   * arrived where it promised.
   *
   * A plate that declares a headline is saying the departure is the point, and says
   * why in a comment beside it. A test asserts a declared headline actually differs
   * from the title, so this cannot quietly become the second copy again.
   */
  headline?: string;
};

const PLATE_SOURCE: PlateSource[] = [
  {
    kind: "reproduction",
    subject: "the trust anchor",
    title: "Did it reproduce GPT-2 124M?",
    // The link asks; the page answers. It is the question a reader arrives with, and
    // the page's whole argument is that the answer worth having is one they can check
    // without taking the answerer's word for it — so the headline states that rather
    // than repeating the question back.
    headline: "A number someone else can check",
  },
  {
    kind: "ablations",
    subject: "the sweep",
    title: "What actually matters",
    // The link is the promise; the headline is the promise and the finding. The
    // subordinate clause is dropped in the path and the feet, where the title sits
    // beside three others and would crowd them — a reader still lands on the words
    // they clicked.
    headline: "What actually matters, and what only sounds like it does",
  },
  {
    kind: "efficiency",
    subject: "inference",
    title: "Making it fast, and the bug that hid in the numbers",
    // The link names the subject; the page states what it found. The bug is the page's
    // argument — a cache doing strictly less arithmetic lost on the clock, and a green
    // suite said nothing, because every test asserted the answer and none the path.
    headline: "The cache was slower, and every test passed",
  },
  {
    kind: "scaling",
    subject: "eight GPUs",
    title: "Eight GPUs, and why the interconnect barely matters",
    // Not the 95%, which is the number the title already promises. The page's evidence is
    // its shape: two points fitted a two-parameter model and the other two, across a
    // fourfold range, could have refuted it and did not.
    headline: "Two points fitted the curve; the other two tested it",
  },
];

export type Plate = PlateSource & { numeral: string; href: string };

/** The four plates, numbered by their position rather than by hand. */
export const PLATES: Plate[] = PLATE_SOURCE.map((plate, i) => ({
  ...plate,
  numeral: ROMAN[i] ?? String(i + 1),
  href: href({ kind: plate.kind } as Route),
}));

/** The kicker a plate page prints above its headline, numeral included. */
export function plateKicker(kind: PlateKind): string {
  const p = plate(kind);
  return `Plate ${p.numeral} — Measured results · ${p.subject}`;
}

/**
 * The headline a plate page prints — the same sentence its links use, unless the
 * plate declared otherwise. Pages call this instead of holding the string themselves,
 * so a title edited here cannot leave a page advertising the old one.
 */
export function plateHeadline(kind: PlateKind): string {
  const p = plate(kind);
  return p.headline ?? p.title;
}

function plate(kind: PlateKind): Plate {
  return PLATES.find((p) => p.kind === kind)!;
}

/**
 * The three destinations the front page offers, and nothing below them.
 *
 * The front page used to print all sixteen stops — eight chapters, four plates, three
 * explorers and the attention page — as one ordered list. Every one of those is still
 * reachable, but enumerating them made the landing page a table of contents set at
 * article scale: a reader had to read sixteen rows to learn there were three ways in.
 * "Tokenization" and "Embeddings" are second-level navigation, and the chapter rail
 * and the plate feet already carry them where they belong, next to the reading.
 *
 * So this says only what the three routes are and how much is down each. The counts
 * are derived rather than written, because the failure they guard against is the one
 * this repository keeps having: a front page advertising eight chapters after a ninth
 * was added, in a sentence nothing tests.
 */
export type Destination = {
  /** `01`–`03`, printed above the title. */
  numeral: string;
  title: string;
  /**
   * One line, and it carries the extent rather than stating it separately — "8 chapters,
   * no prior knowledge assumed" says what a reader is choosing and how much of it there
   * is in the space a bare count used to take a row of its own.
   */
  blurb: string;
  cta: string;
  href: string;
};

export function destinations(): Destination[] {
  return [
    {
      numeral: "01",
      title: "Learn how the model works",
      blurb: `${CHAPTERS.length} chapters, no prior knowledge assumed.`,
      cta: "Start the path →",
      href: href({ kind: "chapter", n: 1 }),
    },
    {
      numeral: "02",
      title: "See whether it actually worked",
      blurb: `${PLATES.length} results pages, read in order, with ${MEASURED.ablations.runs} ablation runs behind them.`,
      cta: "See the results →",
      href: href({ kind: "reproduction" } as Route),
    },
    {
      numeral: "03",
      title: "Inspect the implementation",
      blurb: "Rotary embeddings, the stack block by block, and the tests that pin them.",
      cta: "Open the explorers →",
      href: href({ kind: "rope" } as Route),
    },
  ];
}
