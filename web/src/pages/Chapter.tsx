import ChapterFoot from "../components/ChapterFoot";
import ChapterRail from "../components/ChapterRail";
import { CHAPTER_BODIES } from "../content/chapterBodies";
import { chapter, chapterDateline } from "../content/chapters";

/**
 * One chapter, on its own route.
 *
 * The rail, the dateline kicker, the headline and the foot are the same on all eight;
 * the body is the only thing that changes, which is what makes them chapters of one
 * argument rather than eight loose pages.
 */
export default function Chapter({ n }: { n: number }) {
  const meta = chapter(n);
  const Body = CHAPTER_BODIES[n - 1]!;

  return (
    <div className="shell chapter">
      <ChapterRail current={n} />
      <p className="kicker">{chapterDateline(meta)}</p>
      <h1 className="chapter-title">{meta.title}</h1>
      <Body />
      <ChapterFoot current={n} />
    </div>
  );
}
