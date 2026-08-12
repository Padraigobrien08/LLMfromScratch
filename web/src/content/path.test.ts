import { describe, expect, it } from "vitest";

import { PLATES, plateHeadline } from "./path";

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
