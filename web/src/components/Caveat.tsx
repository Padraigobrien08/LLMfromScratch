import type { CSSProperties, ReactNode } from "react";

/**
 * A small-print note with its department named, printed as furniture rather than as
 * more grey.
 *
 * Two departments, because the site has two jobs for this furniture and the chapters
 * were doing both of them unlabelled.
 *
 * `Caveat.` is what a page is *not* claiming — the run that was expected to diverge and
 * did not, the compression ceiling the headline number does not mention, the interconnect
 * the experiment could not isolate. They are the most valuable paragraphs on the site and
 * they were set as 15px grey body copy, which is exactly what a reader learns to skip.
 *
 * `Provenance.` is where a figure came from and what pins it there — the fixture the
 * browser tokenizer is asserted against, the corpus a distribution was counted from, the
 * test that holds a property nobody could see in a loss curve. The chapters ended their
 * sections in these with no lead at all, so one grey block meant two different things
 * depending on which page a reader had reached.
 *
 * Naming the department fixes that once: the reader meets a lead a second time and
 * already knows what kind of paragraph follows. The type stays deliberately quiet — the
 * lead is the signal, not a colour or a box.
 *
 * `columns` is for the long ones. Past about six lines a 70ch measure of small grey text
 * is a wall; two columns at the closing pair's own gap and measure is the page's existing
 * answer to that, and the only one it needs.
 *
 * `narrow` sets the note on the chapters' 66ch measure rather than the plates' 70ch, so a
 * note sits in its page's own column instead of four characters past it.
 */
type Props = {
  /** Set where the note runs past ~6 lines at desktop width. */
  columns?: boolean;
  /** Set inside a chapter, whose prose is 66ch where a plate's is 70ch. */
  narrow?: boolean;
  style?: CSSProperties;
  children: ReactNode;
};

function Note({ lead, columns, narrow, style, children }: Props & { lead: string }) {
  const className = [narrow ? "prose-caveat" : "caveat-wide", columns ? "caveat-columns" : ""]
    .filter(Boolean)
    .join(" ");

  return (
    <p className={className} style={style}>
      <span className="caveat-lead">{lead}</span> {children}
    </p>
  );
}

export default function Caveat(props: Props) {
  return <Note lead="Caveat." {...props} />;
}

/** Where a figure came from and what holds it there. Chapter measure unless told otherwise. */
export function Provenance({ narrow = true, ...props }: Props) {
  return <Note lead="Provenance." narrow={narrow} {...props} />;
}
