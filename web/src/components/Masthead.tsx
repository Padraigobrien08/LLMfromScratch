import { DATELINE, PROJECT } from "../content/projectState";
import { type Route, href } from "../router";

/**
 * The nav's five destinations. `The path` points at chapter one and stays lit for
 * every chapter, so a reader eight pages in can still see where they are.
 */
const NAV: Array<{ label: string; target: Route; activeFor: Route["kind"][] }> = [
  { label: "Front page", target: { kind: "front" }, activeFor: ["front"] },
  { label: "The path", target: { kind: "chapter", n: 1 }, activeFor: ["chapter"] },
  { label: "RoPE", target: { kind: "rope" }, activeFor: ["rope"] },
  { label: "Architecture", target: { kind: "architecture" }, activeFor: ["architecture"] },
  { label: "Tests", target: { kind: "tests" }, activeFor: ["tests"] },
];

export default function Masthead({ route }: { route: Route }) {
  return (
    <>
      <div className="masthead-bar" />
      <header className="shell">
        <div className="masthead-row">
          <div className="wordmark">
            <a href="#/">LLMfromScratch</a>
            <span className="wordmark-sub">A laboratory notebook</span>
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
        <div className="dateline">
          {DATELINE.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
        <div className="rule-hair" />
      </header>
    </>
  );
}
