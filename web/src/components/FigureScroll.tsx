import type { ReactNode } from "react";

/**
 * A chart that slides inside its own box rather than shrinking below legibility.
 *
 * The plate charts draw their axis labels at 11 units in a 1000-unit viewBox. Scaled
 * into a 375px phone that is under four physical pixels — the labels are still there,
 * still correct, and completely unreadable, which on a page whose argument *is* the
 * figures is the worst of the three options. Below the breakpoint the SVG holds a width
 * where the type is legible and this box scrolls, exactly as `.table-scroll` does for a
 * six-column table and the RoPE ruler does for its rail.
 *
 * `tabIndex={0}` and the label are the same requirement they are there: a scroll
 * container only a finger can move is unreachable for a keyboard user, and one that
 * takes focus has to announce itself as something.
 *
 * The scrubber and the readouts stay outside this box, at the page's full width — the
 * primary way to move through a figure must never be the thing that scrolls.
 */
export default function FigureScroll({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="figure-scroll" role="region" aria-label={label} tabIndex={0}>
      {children}
    </div>
  );
}
