import { describe, expect, it } from "vitest";

import { CLAIM_PREFIX, claimHeadline } from "./claim";
import { TEST_SHOWCASE } from "../content/testShowcase";

describe("showcase claim headlines", () => {
  /**
   * The guard the strip rests on. The markers live in the Python tests, where nothing
   * stops someone phrasing the next one as "the mask is…" instead of "that the mask
   * is…" — and a claim that lost its first word to a prefix it never had would be a
   * page lying about a test, which is the one failure this whole page exists to rule
   * out. Fail here instead.
   */
  it("finds every collected claim phrased to follow the verb", () => {
    for (const row of TEST_SHOWCASE) {
      expect(`${row.file}::${row.name} → ${row.pins}`).toContain(`→ ${CLAIM_PREFIX}`);
    }
  });

  it("starts the headline at the claim's own subject", () => {
    expect(claimHeadline("that the causal mask is bottom-right aligned")).toBe(
      "The causal mask is bottom-right aligned",
    );
  });

  /** A claim that does not carry the prefix is printed whole rather than trimmed. */
  it("leaves a claim it does not recognise alone", () => {
    expect(claimHeadline("RoPE is relative")).toBe("RoPE is relative");
  });
});
