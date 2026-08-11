import type { Payload } from "./ablations";
import { fetchResult } from "./resultsData";

/**
 * Load the sweep's published results, or report that there are none.
 *
 * Two pages depend on this answer in opposite directions: the playground renders its
 * results when the file is there, and the front page's illustrative preview renders
 * only when it is not. They share one implementation so they can never disagree about
 * which state the site is in.
 *
 * The fetching moved to `resultsData.ts` once there were four results plates reading
 * artifacts the same way. The SPA-fallback trap it handles — a missing file answered
 * with index.html and a cheerful 200 — was learned here and is not specific to here.
 */
export function fetchAblations(): Promise<Payload | null> {
  return fetchResult<Payload>("ablations.json");
}
