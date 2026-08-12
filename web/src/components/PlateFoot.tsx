import { PLATES, type PlateKind } from "../content/path";
import { href } from "../router";

/**
 * Where a plate leads.
 *
 * The four measured results are the body of the paper, and a reader who arrives at one
 * of them from the nav — which is how a stranger arrives — previously had no way to
 * reach the other three without going back to the front page. This chains them in the
 * order `PLATES` declares, with the front page at both ends.
 *
 * Reuses the chapter foot's furniture verbatim: it is the same object doing the same
 * job, and a second one would be a second thing to learn.
 */
export default function PlateFoot({ current }: { current: PlateKind }) {
  const here = PLATES.findIndex((plate) => plate.kind === current);
  const before = PLATES[here - 1];
  const after = PLATES[here + 1];

  const previous = before
    ? { href: before.href, label: `← ${before.title}` }
    : { href: href({ kind: "front" }), label: "← Front page" };

  const next = after
    ? { href: after.href, label: `${after.title} →` }
    : { href: href({ kind: "front" }), label: "Front page →" };

  return (
    <div className="chapter-foot">
      <a href={previous.href}>{previous.label}</a>
      <a href={next.href}>{next.label}</a>
    </div>
  );
}
