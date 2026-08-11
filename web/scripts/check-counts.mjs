/**
 * Assert the browser test count in the generated module is the live one.
 *
 * `measured.ts` is produced by `llmfs-export-web`, and `tests/test_web_export.py`
 * re-derives every field of it except this one — enumerating the vitest suite needs a
 * Node toolchain that CI's Python job does not install. So the one field that job has
 * to take on trust is checked here instead, in the job that always has vitest.
 *
 * Between the two, no figure the site prints is unchecked.
 *
 *     npm run check:counts --prefix web
 */
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const web = dirname(dirname(fileURLToPath(import.meta.url)));

const listed = JSON.parse(
  execFileSync("npx", ["vitest", "list", "--json"], { cwd: web, encoding: "utf8" }),
);
const live = listed.length;

const module = readFileSync(join(web, "src/content/measured.ts"), "utf8");
const match = module.match(/"browser":\s*(\d+)/);
if (!match) {
  console.error("measured.ts does not state a browser test count");
  process.exit(1);
}
const committed = Number(match[1]);

if (committed !== live) {
  console.error(
    `measured.ts says ${committed} browser tests; vitest collects ${live}.\n` +
      "Run `llmfs-export-web` and commit the result — the site prints this number.",
  );
  process.exit(1);
}

console.log(`browser test count is current (${live})`);
