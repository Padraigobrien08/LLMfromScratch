import { useEffect, useState } from "react";

/**
 * A hash router in twenty lines, deliberately.
 *
 * GitHub Pages serves static files and has no SPA fallback, so a path-based route
 * like `/rope` 404s on a hard refresh or a shared link — the failure only shows up
 * after deploy, on exactly the link someone else opened. Hashes are handled entirely
 * client-side and cannot 404, which is worth more here than pretty URLs.
 */
export function useRoute(): string {
  const read = () => window.location.hash.replace(/^#\/?/, "") || "";
  const [route, setRoute] = useState(read);

  useEffect(() => {
    const onChange = () => {
      setRoute(read());
      window.scrollTo(0, 0);
    };
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  return route;
}
