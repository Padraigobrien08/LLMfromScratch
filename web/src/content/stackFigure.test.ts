import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

import { describe, expect, it } from "vitest";

import { BLOCKS, SIZES, type Variant } from "./blocks";
import { FIGURE_LABELS, figurePanel } from "./stackFigure";
import { parameters } from "../lib/modelsize";
import { figureParts } from "../lib/stackGeometry";

/**
 * The figure's two standing promises, asserted rather than trusted.
 *
 * **Thickness is the budget.** The drawing's whole argument is that a slab's height is
 * that part's share of the parameter total — the reason the token embedding looks like
 * a third of the object. That is only true if the parts sum to the total, so this
 * checks they do, exactly, in both architectures.
 *
 * **A pin names a test that exists.** Every "What holds it" line claims a specific test
 * asserts a specific thing, on a page whose entire argument is that its claims are
 * checkable. A cited test that does not exist is worse than no citation: a reader who
 * goes looking finds nothing and has no way to tell which of the other claims are real.
 * This walks every pin the figure can print — including the nine it takes from
 * `blocks.ts` — and fails if the file is missing or the named test is not in it.
 */

const VARIANTS: Variant[] = ["gpt2", "llama"];
const REPO = resolve(__dirname, "../../..");
const BLOCK_IDS = [...new Set(["whole", ...FIGURE_LABELS.map((l) => l.blockId)])];

/** `web/src/lib/x.test.ts` is repo-relative; a bare `test_x.py` lives under `tests/`. */
function pinPath(test: string): string {
  const file = test.split("::")[0]!;
  return resolve(REPO, file.includes("/") ? file : `tests/${file}`);
}

describe("the parts sum to the budget", () => {
  it.each(VARIANTS)("%s", (variant) => {
    const parts = figureParts(variant);
    const summed = Object.values(parts).reduce((a, b) => a + b, 0);
    expect(summed).toBeCloseTo(parameters(SIZES[variant]).total, 6);
  });
});

describe("every pin names a test that exists", () => {
  const pins = VARIANTS.flatMap((variant) =>
    BLOCK_IDS.map((id) => ({ id, variant, pins: figurePanel(id, variant, "#/").pins })).filter(
      (row) => row.pins !== null,
    ),
  );

  it("the figure cites at least one pin per architecture", () => {
    expect(pins.length).toBeGreaterThan(VARIANTS.length);
  });

  it.each(pins)("$variant · $id", ({ pins: pin }) => {
    const path = pinPath(pin!.test);
    expect(existsSync(path), `${pin!.test} — no such file`).toBe(true);

    const named = pin!.test.split("::")[1];
    if (named) {
      const source = readFileSync(path, "utf8");
      expect(source.includes(`def ${named}(`), `${pin!.test} — no such test in the file`).toBe(true);
    }
  });
});

describe("every drawn part has a panel", () => {
  it.each(BLOCK_IDS)("%s", (id) => {
    for (const variant of VARIANTS) {
      const panel = figurePanel(id, variant, "#/");
      expect(panel.name.length).toBeGreaterThan(0);
      expect(panel.what.length).toBeGreaterThan(0);
      expect(panel.shape.length).toBeGreaterThan(0);
      expect(panel.links.length).toBeGreaterThan(0);
    }
  });
});

describe("the output-head panel's arithmetic", () => {
  it("quotes the cost of untying that modelsize.ts computes", () => {
    // "Untying would cost as much as adding two whole blocks" was off by 2.7×, on the
    // page that advertises that nothing here is arithmetic anybody did by hand. The
    // prose is checked against the computation it describes.
    const tied = parameters(SIZES.gpt2);
    const added = parameters({ ...SIZES.gpt2, tieEmbeddings: false }).total - tied.total;
    const perBlock =
      parameters({ ...SIZES.gpt2, nLayer: SIZES.gpt2.nLayer + 1 }).total - tied.total;

    const head = BLOCKS.find((b) => b.id === "output-head")!;
    expect(head.what).toContain(`${(added / 1e6).toFixed(1)}M parameters`);
    expect(added / perBlock).toBeGreaterThan(5);
    expect(added / perBlock).toBeLessThan(6);
    expect(head.what).toContain("five and a half");
  });

  it("quotes the same cost in the figure's own tie panel", () => {
    // The tie strand and the output head are the same fact told twice, one click
    // apart in the same detail panel — the output-head copy above got the "two more
    // blocks" error fixed, but the tie panel kept the wrong figure until this pinned
    // it too.
    const tied = parameters(SIZES.gpt2);
    const added = parameters({ ...SIZES.gpt2, tieEmbeddings: false }).total - tied.total;

    const tie = figurePanel("tie", "gpt2", "#/");
    expect(tie.what).toContain(`${(added / 1e6).toFixed(1)}M parameters`);
    expect(tie.what).toContain("five and a half");
  });
});
