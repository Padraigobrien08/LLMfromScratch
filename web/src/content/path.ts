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
 */
export type Stop = {
  /** `01`–`08` for the chapters, `✲` for a deep-end page, `↗` for the one that leaves. */
  numeral: string;
  kicker: string;
  title: string;
  blurb: string;
  href: string;
  cta: "Read" | "Open" | "Visit";
  external?: boolean;
};

const DEEP_END: Array<Omit<Stop, "numeral" | "cta">> = [
  {
    kicker: "Measured results",
    title: "Did it reproduce GPT-2 124M?",
    blurb:
      "A validation-loss target fixed before the run, met a third of the way in, and corroborated on a public benchmark against the published figure. Drag the scrubber along the run.",
    href: href({ kind: "reproduction" } as Route),
  },
  {
    kicker: "Measured results",
    title: "What actually matters",
    blurb:
      "Twelve arms at three seeds, every comparison paired against the baseline run that saw its data in the same order — including the changes that measurably did nothing.",
    href: href({ kind: "ablations" } as Route),
  },
  {
    kicker: "Measured results",
    title: "Making it fast, and the bug in the numbers",
    blurb:
      "The KV cache was slower than recomputing until a mask that forfeited the fused kernel turned up. Flip the bug back on and watch the curve change sides.",
    href: href({ kind: "efficiency" } as Route),
  },
  {
    kicker: "Measured results",
    title: "Eight GPUs, and the cost of talking",
    blurb:
      "95% of linear scaling over the worst interconnect in the building, and a two-parameter model fitted to two points that then predicted the other two.",
    href: href({ kind: "scaling" } as Route),
  },
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
        kicker: c.kicker,
        title: c.title,
        blurb: c.blurb,
        href: href({ kind: "chapter", n: c.n }),
        cta: "Read",
      }),
    ),
    ...DEEP_END.map((stop): Stop => ({ ...stop, numeral: "✲", cta: "Open" })),
    {
      numeral: "↗",
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
