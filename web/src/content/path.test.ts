import { describe, expect, it } from "vitest";

import { CHAPTERS } from "./chapters";
import { MEASURED } from "./measured";
import { PLATES, destinations, plateHeadline } from "./path";

describe("the three destinations", () => {
  /**
   * The front page no longer lists the chapters, so it can no longer be checked by
   * reading it: "8 chapters" is now a claim about a collection the reader cannot see
   * from there. This is the assertion that keeps it true when a ninth is written.
   */
  it("counts the chapters and plates rather than stating them", () => {
    const [learn, results] = destinations();
    expect(learn!.blurb).toContain(`${CHAPTERS.length} chapters`);
    expect(results!.blurb).toContain(`${PLATES.length} measured plates`);
    expect(results!.blurb).toContain(`${MEASURED.ablations.runs} ablation runs`);
  });

  /** Three ways in, three different pages, every one of them a real route. */
  it("sends each row somewhere, and somewhere different", () => {
    const rows = destinations();
    expect(rows).toHaveLength(3);
    expect(new Set(rows.map((r) => r.href)).size).toBe(3);
    for (const row of rows) expect(row.href).toMatch(/^#\/\w/);
  });
});

describe("the plates", () => {
  /**
   * The whole point of `headline` being optional. A plate that declares one is saying
   * its page departs from what its links call it, deliberately; a declared headline
   * that merely repeats the title is two copies of one string again, which is the
   * arrangement that let plates III and IV be advertised by links that landed on
   * pages headlined something else.
   */
  it("declares a headline only where the page departs from the link", () => {
    for (const plate of PLATES) {
      if (plate.headline !== undefined) {
        expect(plate.headline, `plate ${plate.numeral} restates its title`).not.toBe(plate.title);
      }
    }
  });

  /** A plate that declares nothing prints the title its links use — no third string. */
  it("prints the link's own words where nothing else is declared", () => {
    for (const plate of PLATES) {
      expect(plateHeadline(plate.kind)).toBe(plate.headline ?? plate.title);
    }
    expect(plateHeadline("efficiency")).toBe(
      PLATES.find((p) => p.kind === "efficiency")!.title,
    );
  });
});
