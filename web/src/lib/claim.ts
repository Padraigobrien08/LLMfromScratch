/**
 * A showcase claim, as a headline.
 *
 * The `pins=` markers in the Python tests are phrased to follow the verb — "that the
 * causal mask is bottom-right aligned" — because that is how they read in the source,
 * and the source is where they belong. The page used to print them as "It asserts
 * {pins}", which gave twelve consecutive headlines the same three opening words; the
 * page's own headline already says what the rows are.
 *
 * So the clause comes off here, at the display layer, leaving the generated file
 * untouched. `claim.test.ts` asserts every marker still carries the prefix, so a claim
 * phrased some other way would fail the suite rather than come out mangled.
 */
export const CLAIM_PREFIX = "that ";

export function claimHeadline(pins: string): string {
  if (!pins.startsWith(CLAIM_PREFIX)) return pins;
  return pins.slice(CLAIM_PREFIX.length).replace(/^./, (c) => c.toUpperCase());
}
