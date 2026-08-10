import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Deployed to https://<user>.github.io/LLMfromScratch/, so every asset URL needs the
// repository name as a prefix. Overridable because a local `vite preview` and any
// fork with a different repo name both need a different base.
const base = process.env.SITE_BASE ?? "/LLMfromScratch/";

export default defineConfig({
  base,
  plugins: [react()],
  build: {
    outDir: "dist",
    // The site is a handful of pages; a source map costs nothing to ship and makes
    // the deployed build debuggable rather than a wall of minified identifiers.
    sourcemap: true,
  },
});
