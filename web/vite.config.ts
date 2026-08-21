import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { type Plugin, defineConfig } from "vite";

// Deployed to https://<user>.github.io/nanogpt-from-scratch/, so every asset URL needs the
// repository name as a prefix. Overridable because a local `vite preview` and any
// fork with a different repo name both need a different base.
const base = process.env.SITE_BASE ?? "/nanogpt-from-scratch/";

const RESULTS = fileURLToPath(new URL("../results/", import.meta.url));

/**
 * Serve `results/*.json` in development, the way the deploy does.
 *
 * `.github/workflows/pages.yml` copies those files into the site before building it,
 * so the deployed pages render measurements and their "not published yet" states stay
 * hidden. The dev server has no such step: without this, a results page shows its
 * empty state locally and its real state only after a deploy — which is exactly where
 * that class of bug is most expensive to find, because the only way to see it is to
 * ship it.
 *
 * Both this and the workflow step once named `ablations.json` specifically. Four more
 * results pages would have meant four more edits in two files kept in sync by memory,
 * so both now take the whole directory. Absent results still fall through to the 404
 * the pages are written to expect.
 */
function serveResults(): Plugin {
  return {
    name: "llmfs-serve-results",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = (req.url ?? "").split("?")[0] ?? "";
        const prefix = `${base}data/`;
        if (!path.startsWith(prefix)) return next();

        // Only ever a bare filename out of results/: no traversal, no subdirectories.
        const name = path.slice(prefix.length);
        if (!/^[\w.-]+\.json$/.test(name)) return next();

        // `RESULTS` is a filesystem path, not a URL — joining it as one throws.
        const file = join(RESULTS, name);
        if (!existsSync(file)) return next();
        res.setHeader("content-type", "application/json");
        res.end(readFileSync(file));
      });
    },
  };
}

export default defineConfig({
  base,
  plugins: [react(), serveResults()],
  build: {
    outDir: "dist",
    // The site is a handful of pages; a source map costs nothing to ship and makes
    // the deployed build debuggable rather than a wall of minified identifiers.
    sourcemap: true,
  },
});
