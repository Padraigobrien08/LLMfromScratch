import { CHAPTERS } from "../content/chapters";
import { href } from "../router";

/**
 * The eight chapters as a rail, so a reader can see the whole path from inside any
 * one of them. Each carries its title as a tooltip — the numeral alone says where
 * you are, not what is there.
 */
export default function ChapterRail({ current }: { current: number }) {
  return (
    <nav className="chapter-rail" aria-label="Chapters">
      {CHAPTERS.map((c) => (
        <a
          key={c.n}
          href={href({ kind: "chapter", n: c.n })}
          title={c.title}
          aria-current={c.n === current ? "page" : undefined}
        >
          {String(c.n).padStart(2, "0")}
        </a>
      ))}
    </nav>
  );
}
