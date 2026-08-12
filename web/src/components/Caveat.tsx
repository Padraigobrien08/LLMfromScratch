import type { CSSProperties, ReactNode } from "react";

/**
 * The thing this page is not claiming, printed as furniture rather than as more grey.
 *
 * Every plate ends in one of these — the run that was expected to diverge and did not,
 * the compression ceiling the headline number does not mention, the interconnect the
 * experiment could not isolate. They are the most valuable paragraphs on the site and
 * they were set as 15px grey body copy, which is exactly what a reader learns to skip.
 *
 * Naming the department fixes that once: the reader meets `Caveat.` a second time and
 * already knows what kind of paragraph follows. The type stays deliberately quiet — the
 * lead is the signal, not a colour or a box.
 *
 * `columns` is for the long ones. Past about six lines a 70ch measure of small grey text
 * is a wall; two columns at the closing pair's own gap and measure is the page's existing
 * answer to that, and the only one it needs.
 */
type Props = {
  /** Set where the caveat runs past ~6 lines at desktop width. */
  columns?: boolean;
  style?: CSSProperties;
  children: ReactNode;
};

export default function Caveat({ columns, style, children }: Props) {
  return (
    <p className={columns ? "caveat-wide caveat-columns" : "caveat-wide"} style={style}>
      <span className="caveat-lead">Caveat.</span> {children}
    </p>
  );
}
