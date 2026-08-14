import { CHAPTERS, chapter } from "../content/chapters";
import { href } from "../router";

/**
 * Where a chapter leads. The path runs from the front page through the eight chapters
 * and out into the RoPE explorer, so the first chapter's previous and the last
 * chapter's next leave the sequence rather than dead-ending in it.
 *
 * The forward link carries the next chapter's question above its name. A foot that
 * printed only titles said what the next page is called; the question says why the
 * reader is not finished — a reader leaving chapter four should leave on "how does the
 * model know which word came first?", which is the thing chapter four could not answer.
 *
 * It reads the question from `chapters.ts` rather than restating it, so the sentence the
 * foot promises and the sentence the next page opens on are the same sentence.
 */
export default function ChapterFoot({ current }: { current: number }) {
  const previous =
    current > 1
      ? { href: href({ kind: "chapter", n: current - 1 }), label: `← ${CHAPTERS[current - 2]!.title}` }
      : { href: href({ kind: "front" }), label: "← Front page" };

  const nextChapter = current < CHAPTERS.length ? chapter(current + 1) : null;

  return (
    <div className="chapter-foot">
      <a href={previous.href}>{previous.label}</a>

      {nextChapter ? (
        <a className="chapter-foot-next" href={href({ kind: "chapter", n: current + 1 })}>
          <span className="chapter-foot-question">{nextChapter.question}</span>
          <span className="chapter-foot-title">{nextChapter.title} →</span>
        </a>
      ) : (
        <a href={href({ kind: "rope" })}>The RoPE explorer →</a>
      )}
    </div>
  );
}
