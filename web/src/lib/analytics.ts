import type { Route } from "../router";

/**
 * Web analytics for a hash router, and only where the endpoint exists.
 *
 * Vercel's script auto-tracks by patching `pushState`/`replaceState` and listening for
 * `popstate`. This site navigates by fragment (`#/rope`), which does none of those: a
 * fragment navigation fires `hashchange` and nothing else, so auto-tracking would report
 * one pageview per session — the landing — and never see the eight chapters or the four
 * plates. Passing `route` and `path` to `<Analytics>` turns auto-tracking off and reports
 * a pageview whenever they change, which is exactly the hook a hash router needs.
 */

/** The `route`/`path` pair for a route: the pattern to group by, and the page itself. */
export interface Pageview {
  /** The route pattern, with the chapter number as a parameter so the eight group as one. */
  route: string;
  /** The concrete page, which is what the pattern's per-path breakdown is built from. */
  path: string;
}

export function pageviewFor(route: Route): Pageview {
  switch (route.kind) {
    case "front":
      return { route: "/", path: "/" };
    case "chapter":
      return { route: "/chapter/[n]", path: `/chapter/${route.n}` };
    default:
      return { route: `/${route.kind}`, path: `/${route.kind}` };
  }
}

/**
 * Whether to load the script at all.
 *
 * `@vercel/analytics` fetches `/_vercel/insights/script.js`, a path Vercel serves and
 * nobody else does. The same build is also deployed to GitHub Pages, where that request
 * 404s and the package logs a "failed to load" line to every visitor's console — a
 * broken-looking site in exchange for numbers that could never arrive. So Pages opts out
 * by hostname, and everything else — the custom domain, preview deployments on
 * `*.vercel.app`, and localhost, where the package logs events instead of sending them —
 * opts in.
 */
export function analyticsEnabled(hostname: string): boolean {
  return !hostname.endsWith(".github.io");
}
