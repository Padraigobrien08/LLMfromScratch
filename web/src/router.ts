import { useEffect, useState } from "react";

/**
 * A hash router in fifty lines, deliberately.
 *
 * GitHub Pages serves static files and has no SPA fallback, so a path-based route
 * like `/rope` 404s on a hard refresh or a shared link — the failure only shows up
 * after deploy, on exactly the link someone else opened. Hashes are handled entirely
 * client-side and cannot 404, which is worth more here than pretty URLs.
 */

/** The eight explainer chapters, each on its own route. */
export const CHAPTER_COUNT = 8;

export type Route =
  | { kind: "front" }
  | { kind: "chapter"; n: number }
  | { kind: "rope" }
  | { kind: "architecture" }
  | { kind: "tests" }
  | { kind: "ablations" };

export const FRONT: Route = { kind: "front" };

/**
 * Parse a `window.location.hash` into a route.
 *
 * Kept pure and separate from the hook so the route table can be tested without a
 * DOM. Anything unrecognised — including a chapter number outside 1–8 — falls back
 * to the front page rather than rendering a half-page for a URL that does not exist.
 */
export function parseRoute(hash: string): Route {
  const path = hash.replace(/^#\/?/, "").replace(/\/$/, "");

  if (path === "") return FRONT;
  if (path === "rope") return { kind: "rope" };
  if (path === "architecture") return { kind: "architecture" };
  if (path === "tests") return { kind: "tests" };
  if (path === "ablations") return { kind: "ablations" };

  const chapter = /^chapter\/(\d+)$/.exec(path);
  if (chapter) {
    const n = Number(chapter[1]);
    if (Number.isInteger(n) && n >= 1 && n <= CHAPTER_COUNT) return { kind: "chapter", n };
  }

  return FRONT;
}

/** The href for a route, for links and for `aria-current` comparisons. */
export function href(route: Route): string {
  switch (route.kind) {
    case "front":
      return "#/";
    case "chapter":
      return `#/chapter/${route.n}`;
    default:
      return `#/${route.kind}`;
  }
}

export function useRoute(): Route {
  const read = () => parseRoute(window.location.hash);
  const [route, setRoute] = useState<Route>(read);

  useEffect(() => {
    const onChange = () => {
      setRoute(read());
      // The design is a newspaper, not a slideshow: every navigation starts at the
      // masthead rather than halfway down the previous page's scroll position.
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return route;
}
