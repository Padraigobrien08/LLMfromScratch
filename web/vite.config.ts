import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

import react from "@vitejs/plugin-react";
import { type Plugin, defineConfig } from "vite";

// Deployed to https://<user>.github.io/LLMfromScratch/, so every asset URL needs the
// repository name as a prefix. Overridable because a local `vite preview` and any
// fork with a different repo name both need a different base.
const base = process.env.SITE_BASE ?? "/LLMfromScratch/";

const RESULTS = fileURLToPath(new URL("../results/ablations.json", import.meta.url));

/**
 * Serve the sweep's results in development, the way the deploy does.
 *
 * `.github/workflows/pages.yml` copies `results/ablations.json` into the site before
 * building it, so the deployed playground renders results and the front page's
 * illustrative preview stays hidden. The dev server has no such step: without this,
 * both pages show their "not published yet" state locally and the difference only
 * turns up after a deploy. Absent results still fall through to the 404 the pages
 * are written to expect.
 */
function serveAblationResults(): Plugin {
  return {
    name: "llmfs-serve-ablation-results",
    apply: "serve",
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const path = (req.url ?? "").split("?")[0];
        if (path !== `${base}data/ablations.json` || !existsSync(RESULTS)) return next();
        res.setHeader("content-type", "application/json");
        res.end(readFileSync(RESULTS));
      });
    },
  };
}

export default defineConfig({
  base,
  plugins: [react(), serveAblationResults()],
  build: {
    outDir: "dist",
    // The site is a handful of pages; a source map costs nothing to ship and makes
    // the deployed build debuggable rather than a wall of minified identifiers.
    sourcemap: true,
  },
});
