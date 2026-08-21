/**
 * Build the reviewable local snapshot of the site.
 *
 *     npm run snapshot --prefix web
 *
 * The deployed site is assembled by `.github/workflows/pages.yml`, which copies the
 * measured results in, builds, and lays the attention explorer alongside. Reproducing
 * that by hand is three commands in the right order, and getting one of them wrong is
 * quiet: the build's `--emptyOutDir` removes the explorer every time, so a snapshot that
 * looked complete would 404 on `/attention/` until someone clicked it.
 *
 * The difference from the deploy is the base path. The published site lives under
 * `/nanogpt-from-scratch/`; this builds at the filesystem root so the snapshot can be served
 * from any static server, which is the only way to look at it without a dev server.
 */
import { execFileSync } from "node:child_process";
import { cpSync, existsSync, mkdirSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = dirname(dirname(fileURLToPath(import.meta.url)));
const root = dirname(web);
const out = join(root, "local-site");

/* The results pages fetch these at runtime and render an honest "not published yet" when
   one is missing, so a snapshot without them is a working page telling a lie about the
   state of the project. */
const results = join(root, "results");
const publicData = join(web, "public", "data");
mkdirSync(publicData, { recursive: true });
const json = readdirSync(results).filter((f) => f.endsWith(".json"));
for (const f of json) cpSync(join(results, f), join(publicData, f));
console.log(`published ${json.length} results files`);

execFileSync("npx", ["vite", "build", "--outDir", out, "--emptyOutDir"], {
  cwd: web,
  stdio: "inherit",
  env: { ...process.env, SITE_BASE: "/" },
});

/* Last, because the build empties the directory first. The explorer is generated from a
   trained checkpoint by `llmfs-viz`, so what is copied here is whatever was last built —
   CI regenerates it on every deploy. */
const explorer = join(root, "site", "attention.html");
if (existsSync(explorer)) {
  mkdirSync(join(out, "attention"), { recursive: true });
  cpSync(explorer, join(out, "attention", "index.html"));
  console.log("copied the attention explorer");
} else {
  console.log("no site/attention.html — run `llmfs-viz` if you want /attention/ locally");
}

console.log(`\nsnapshot ready: ${out}`);
console.log("serve it with:  python3 -m http.server 4173 --directory local-site");
