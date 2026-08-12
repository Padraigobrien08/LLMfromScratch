import { type Route, href } from "../router";
import { CHAPTERS } from "./chapters";

/**
 * The guided path: the whole argument of the site, in the order it should be read.
 *
 * Eight chapters explaining what a language model is, then the four measured results —
 * which are the paper's body, not an appendix to it — then the explorers for anyone who
 * wants to take a mechanism apart. The external attention page is last because it
 * leaves the site.
 *
 * Results before explorers is a deliberate reordering. A reader who has finished the
 * chapters has been told how a transformer works; what they have not yet been shown is
 * whether any of it was built correctly, and that is the question the four plates
 * answer.
 *
 * The four plates are numbered I–IV here and nowhere else. Their pages print the
 * numeral in their own kickers and their feet chain in this order, both read from
 * `PLATES` — so reordering this array reorders the paper, and no page can disagree
 * with the front one about which plate it is.
 */
export type StopGroup = "chapters" | "plates" | "deep-end" | "external";

export type Stop = {
  /** `01`–`08` for the chapters, `I`–`IV` for the plates, `✲` for a deep-end page, `↗` for the one that leaves. */
  numeral: string;
  group: StopGroup;
  kicker: string;
  title: string;
  blurb: string;
  href: string;
  cta: "Read" | "Open" | "Visit";
  external?: boolean;
};

/**
 * The heading a group prints above its first stop on the front page.
 *
 * The chapters have none: they run directly under "The path", which is their heading.
 * A group with a heading does not repeat it as a per-row kicker — that repetition,
 * seven rows of it, was what the grouping is for.
 */
export const GROUP_HEADING: Partial<Record<StopGroup, string>> = {
  plates: "The results, as four plates",
  "deep-end": "The deep end",
};

export type PlateKind = "reproduction" | "ablations" | "efficiency" | "scaling";

const ROMAN = ["I", "II", "III", "IV"] as const;

type PlateSource = {
  kind: PlateKind;
  /** The half of the kicker that follows "Plate II — Measured results ·". */
  subject: string;
  title: string;
  blurb: string;
};

const PLATE_SOURCE: PlateSource[] = [
  {
    kind: "reproduction",
    subject: "the trust anchor",
    title: "Did it reproduce GPT-2 124M?",
    blurb:
      "A validation-loss target fixed before the run, met a third of the way in, and corroborated on a public benchmark against the published figure. Drag the scrubber along the run.",
  },
  {
    kind: "ablations",
    subject: "the sweep",
    title: "What actually matters",
    blurb:
      "Twelve arms at three seeds, every comparison paired against the baseline run that saw its data in the same order — including the changes that measurably did nothing.",
  },
  {
    kind: "efficiency",
    subject: "inference",
    title: "Making it fast, and the bug in the numbers",
    blurb:
      "The KV cache was slower than recomputing until a mask that forfeited the fused kernel turned up. Flip the bug back on and watch the curve change sides.",
  },
  {
    kind: "scaling",
    subject: "eight GPUs",
    title: "Eight GPUs, and the cost of talking",
    blurb:
      "95% of linear scaling over the worst interconnect in the building, and a two-parameter model fitted to two points that then predicted the other two.",
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
  const plate = PLATES.find((p) => p.kind === kind)!;
  return `Plate ${plate.numeral} — Measured results · ${plate.subject}`;
}

const EXPLORERS: Array<Omit<Stop, "numeral" | "cta" | "group">> = [
  {
    kicker: "The deep end",
    title: "The RoPE explorer",
    blurb:
      "Move two tokens along a sequence and watch the attention logit between them refuse to change. The property the test suite asserts, running live.",
    href: href({ kind: "rope" } as Route),
  },
  {
    kicker: "The deep end",
    title: "The stack, block by block",
    blurb:
      "One Transformer class, two architectures, decided entirely by config. Click a block for its shape, its share of the parameter budget, and the test that pins it.",
    href: href({ kind: "architecture" } as Route),
  },
  {
    kicker: "The deep end",
    title: "What the suite actually asserts",
    blurb:
      "Not a test count — the specific claims a dozen of them make, and the bug each one exists to catch. Collected from the tests themselves, so a rename cannot leave the page lying.",
    href: href({ kind: "tests" } as Route),
  },
];

export function pathStops(attentionHref: string): Stop[] {
  return [
    ...CHAPTERS.map(
      (c): Stop => ({
        numeral: String(c.n).padStart(2, "0"),
        group: "chapters",
        kicker: c.kicker,
        title: c.title,
        blurb: c.blurb,
        href: href({ kind: "chapter", n: c.n }),
        cta: "Read",
      }),
    ),
    ...PLATES.map(
      (plate): Stop => ({
        numeral: plate.numeral,
        group: "plates",
        kicker: "Measured results",
        title: plate.title,
        blurb: plate.blurb,
        href: plate.href,
        cta: "Open",
      }),
    ),
    ...EXPLORERS.map((stop): Stop => ({ ...stop, numeral: "✲", group: "deep-end", cta: "Open" })),
    {
      numeral: "↗",
      group: "external",
      kicker: "Separate page",
      title: "The attention explorer",
      blurb:
        "Every attention weight, per layer and per head, in a single self-contained HTML file built by CI from a model CI trains.",
      href: attentionHref,
      cta: "Visit",
      external: true,
    },
  ];
}
