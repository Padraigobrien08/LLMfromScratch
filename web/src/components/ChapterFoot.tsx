import { CHAPTERS } from "../content/chapters";
import { href } from "../router";

/**
 * Where a chapter leads. The path runs from the front page through the eight
 * chapters and out into the RoPE explorer, so the first chapter's previous and the
 * last chapter's next leave the sequence rather than dead-ending in it.
 */
export default function ChapterFoot({ current }: { current: number }) {
  const previous =
    current > 1
      ? { href: href({ kind: "chapter", n: current - 1 }), label: `← ${CHAPTERS[current - 2]!.title}` }
      : { href: href({ kind: "front" }), label: "← Front page" };

  const next =
    current < CHAPTERS.length
      ? { href: href({ kind: "chapter", n: current + 1 }), label: `${CHAPTERS[current]!.title} →` }
      : { href: href({ kind: "rope" }), label: "The RoPE explorer →" };

  return (
    <div className="chapter-foot">
      <a href={previous.href}>{previous.label}</a>
      <a href={next.href}>{next.label}</a>
    </div>
  );
}
