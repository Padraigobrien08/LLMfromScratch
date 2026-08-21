import { DATELINE, DATELINE_FRONT, PROJECT } from "../content/projectState";
import { type Route, href } from "../router";

/**
 * The nav's destinations, named for what a reader wants rather than for what the page
 * contains.
 *
 * `RoPE` and `Architecture` used to sit here as themselves. Both are implementation
 * concepts: they tell a reader who already knows what rotary embeddings are that this
 * site has a page about them, and tell everyone else nothing. They are now the two
 * pages behind `Explore`, which is a thing a reader can want before they have the
 * vocabulary to ask for it.
 *
 * Each entry stays lit across the pages it leads to, so a reader eight chapters or
 * three plates in can still see where they are: `Learn` across every chapter, `Results`
 * across all four plates — they are one argument read in order, not four unrelated
 * pages — and `Explore` across both explorers.
 */
const NAV: Array<{ label: string; target: Route; activeFor: Route["kind"][] }> = [
  { label: "Front page", target: { kind: "front" }, activeFor: ["front"] },
  { label: "Learn", target: { kind: "chapter", n: 1 }, activeFor: ["chapter"] },
  {
    label: "Results",
    target: { kind: "reproduction" },
    activeFor: ["reproduction", "ablations", "efficiency", "scaling"],
  },
  { label: "Explore", target: { kind: "rope" }, activeFor: ["rope", "architecture"] },
  { label: "Tests", target: { kind: "tests" }, activeFor: ["tests"] },
];

export default function Masthead({ route }: { route: Route }) {
  return (
    <>
      <div className="masthead-bar" />
      {/* The front page sets itself wider than an article — it is a front page, not a
          column of prose — and a masthead held to the reading measure above it would
          print the title of a narrower paper than the one underneath. So the nameplate
          takes the same width as whatever it sits over. */}
      <header className={`shell${route.kind === "front" ? " front-shell" : ""}`}>
        <div className="masthead-row">
          <div className="wordmark">
            <a href="#/">NanoGPT From Scratch</a>
            {/* The design system is called Broadsheet and the whole page is set as one —
                plates, dateline, colophon, CMYK numerals. "A laboratory notebook" was the
                one string pulling the other way, and it also described the wrong thing: a
                notebook is private and chronological, where this is curated and published. */}
            <span className="wordmark-sub">Laboratory notes, printed</span>
          </div>
          <nav className="masthead-nav" aria-label="Sections">
            {NAV.map((item) => (
              <a
                key={item.label}
                href={href(item.target)}
                aria-current={item.activeFor.includes(route.kind) ? "page" : undefined}
              >
                {item.label}
              </a>
            ))}
            <a href={PROJECT.repo} target="_blank" rel="noopener">
              GitHub ↗
            </a>
          </nav>
        </div>

        <div className="rule-heavy" />
        {/* The front page prints the three measured figures itself, at reading size and
            linked to their proof pages, forty pixels below this rail — so there the rail
            keeps only the two facts the page does not repeat: whose work this is and
            under what licence. Everywhere else it carries all five. */}
        <div className="dateline">
          {(route.kind === "front" ? DATELINE_FRONT : DATELINE).map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
        <div className="rule-hair" />
      </header>
    </>
  );
}
