import { type Route, href } from "../router";
import { CHAPTERS } from "./chapters";

/**
 * The guided path: the whole argument of the site, in the order it should be read.
 *
 * Eight chapters, then the four places to go once the chapters have been read. The
 * external explorer is last because it leaves the site.
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
      "One Transformer class, two architectures, decided entirely by config. Click a block for its shape, its parameters and the test that pins it.",
    href: href({ kind: "architecture" } as Route),
  },
  {
    kicker: "The deep end",
    title: "What the suite actually asserts",
    blurb:
      "Eleven tests and the specific bug each one exists to catch — the ones that would otherwise survive into a training run.",
    href: href({ kind: "tests" } as Route),
  },
  {
    // Not in the original handoff, which was written while the sweep was still
    // pending. It has since run, and a site that hides its own measured results to
    // stay faithful to a design would be the wrong kind of faithful.
    kicker: "Measured results",
    title: "The ablation playground",
    blurb:
      "Twelve arms at three seeds, every comparison paired against the baseline run that saw its data in the same order — including the two changes that measurably did nothing.",
    href: href({ kind: "ablations" } as Route),
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
