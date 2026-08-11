import type { Payload } from "./ablations";

/**
 * Load the sweep's published results, or report that there are none.
 *
 * A missing file is the expected state before the sweep has run, not a failure — but
 * the status code alone cannot detect it. The Vite dev server and any host with an
 * SPA fallback answer a missing path with index.html and a 200, so the content type
 * is what actually distinguishes "not published yet" from "published and broken".
 *
 * Two pages depend on this answer in opposite directions: the playground renders its
 * results when the file is there, and the front page's illustrative preview renders
 * only when it is not. They share one implementation so they can never disagree
 * about which state the site is in.
 */
export async function fetchAblations(): Promise<Payload | null> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/ablations.json`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  if (!(response.headers.get("content-type") ?? "").includes("application/json")) return null;
  return (await response.json()) as Payload;
}
