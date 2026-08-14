import ChapterFoot from "../components/ChapterFoot";
import ChapterRail from "../components/ChapterRail";
import { CHAPTER_BODIES } from "../content/chapterBodies";
import { chapter, chapterDateline } from "../content/chapters";

/**
 * One chapter, on its own route.
 *
 * The order here is the argument's order, and it is the point of the page. A chapter
 * opens on what has been established and what it still cannot do, then asks the
 * question that follows from it — and only then does the body answer, under the
 * sentence the chapter is named for.
 *
 * It used to open on that name. "Letting every token look back" is a good sentence to
 * remember a chapter by and a poor way to begin one: it tells a reader who already
 * understands attention which page they are on, and gives everyone else nothing to be
 * curious about. The name still labels the chapter everywhere the path is navigated;
 * on the page it is now the answer, printed where the argument earns it.
 */
export default function Chapter({ n }: { n: number }) {
  const meta = chapter(n);
  const Body = CHAPTER_BODIES[n - 1]!;

  return (
    <div className="shell chapter">
      <ChapterRail current={n} />
      <p className="kicker">{chapterDateline(meta)}</p>

      {/* Where the reader has got to, in the one sentence that makes the question next.
          Set behind its own lead like `Caveat.` and `Provenance.`, because it is the same
          kind of object — a quiet aside in the paper's own voice — and because the page
          already has furniture for that. As a bordered stack of lines it cost 90px and
          read as a panel, which is the direction an LMS lies in. */}
      {meta.storySoFar && (
        <p className="chapter-recap">
          <span className="chapter-recap-lead">The story so far.</span> {meta.storySoFar}
        </p>
      )}

      <h1 className="chapter-question">{meta.question}</h1>

      <Body />
      <ChapterFoot current={n} />
    </div>
  );
}
