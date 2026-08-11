import { useEffect, useState } from "react";

/**
 * Load a published artifact out of `results/`, or report that it is not there.
 *
 * A missing file is the expected state before a run has happened, not a failure — but
 * the status code alone cannot detect it. The Vite dev server and any host with an SPA
 * fallback answer a missing path with index.html and a 200, so the content type is what
 * actually distinguishes "not published yet" from "published and broken". That
 * distinction was learned once on the ablations page and applies to every results page,
 * which is why it lives here rather than being reimplemented per plate.
 */
export async function fetchResult<T>(name: string): Promise<T | null> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/${name}`);
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  if (!(response.headers.get("content-type") ?? "").includes("application/json")) return null;
  return (await response.json()) as T;
}

export type LoadState<T> =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "error"; message: string }
  | { status: "ready"; data: T };

/**
 * The same four-state load, once, for every plate that reads an artifact.
 *
 * Four states rather than three: "absent" and "error" have to stay distinguishable,
 * because a page that renders "not published yet" over a server fault is lying about
 * which of the two happened, and the fix for each is entirely different.
 */
export function useResult<T>(name: string): LoadState<T> {
  const [state, setState] = useState<LoadState<T>>({ status: "loading" });

  useEffect(() => {
    let live = true;
    fetchResult<T>(name).then(
      (data) => live && setState(data === null ? { status: "absent" } : { status: "ready", data }),
      (error: unknown) =>
        live &&
        setState({
          status: "error",
          message: error instanceof Error ? error.message : String(error),
        }),
    );
    return () => {
      live = false;
    };
  }, [name]);

  return state;
}
