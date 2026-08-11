import type { CSSProperties, ReactNode } from "react";

/**
 * A table that scrolls inside its own box rather than dragging the page sideways.
 *
 * Six-column results tables do not fit a 375px screen and never will. The failure that
 * matters is not the table — it is that an over-wide table makes the *whole document*
 * scroll horizontally, so every paragraph on the page drifts off the left edge while the
 * reader tries to get back. Confining the overflow keeps the article a column of text
 * with one thing in it that happens to slide.
 *
 * `tabIndex={0}` is not decoration. A scroll container that only a mouse or a finger can
 * move is unreachable content for a keyboard user, and browsers only make one focusable
 * if you ask. With it, the region takes focus and the arrow keys scroll it — which is why
 * it also needs a label, so what just took focus is announced as something.
 */
type Props = {
  /** Announced when the scroll region takes focus, so it is not just "region". */
  label: string;
  className?: string;
  style?: CSSProperties;
  children: ReactNode;
};

export default function DataTable({ label, className, style, children }: Props) {
  return (
    <div className="table-scroll" role="region" aria-label={label} tabIndex={0} style={style}>
      <table className={className ? `table ${className}` : "table"}>{children}</table>
    </div>
  );
}
