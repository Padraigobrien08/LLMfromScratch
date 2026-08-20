/**
 * Figure V — the drawing contract.
 *
 * These are the rules the figure is drawn by, not decoration. Every value here is
 * referenced from the engine; none of them is repeated as a literal there. Adding a
 * block, a label or a variant means editing THIS file first.
 *
 * 1. FACETS — exactly three tones per solid, in one fixed order: top (lightest) ·
 *    front (mid) · side (darkest), every edge a hairline of `--color-text`. No fourth
 *    tone, no gradient, no light source, no shadow. A solid's three tones always come
 *    from ONE plate's three steps, so a new part cannot introduce a new value
 *    relationship.
 *
 * 2. INK IS ROLE, NOT IDENTITY — hue encodes what a part does, taken from the four
 *    process plates. Parts that share a tensor share an ink; that is why the output
 *    head is cyan, because it is the token embedding transposed. A new block joins an
 *    existing role in `ROLE_INK`, it does not get a new hue. Unassigned blocks fall
 *    back to neutral and warn — see `validate`.
 *
 * 3. SELECTION IS DEPTH, NOT COLOUR — the selected part deepens one step inside its
 *    own plate and doubles its ink edge; its label takes the accent rule. Selection
 *    never recolours a part into another role's ink. This is the rule that lets the
 *    four plates carry meaning: if selection were a hue, it would collide with them.
 *
 * 4. LEADERS — every label is ruled out to a margin: a short angled run off the
 *    surface dot, one bend, then a short horizontal into the type. Labels are assigned
 *    a margin by `SIDES` and never cross the object to reach it; within a margin they
 *    are ordered by the height of their anchor, so leaders cannot invert.
 *
 * 5. LABEL TIERS — three, and only three: region (small caps), part (roman), flow
 *    (italic, quieter). A label without a tier is a bug, not a default.
 *
 * 6. MOTION — a rest pose the figure returns to, an orbit clamped so the stack never
 *    inverts, and nothing at all under `prefers-reduced-motion`.
 */

/**
 * The facet order — index 0 top, 1 front, 2 side — is stated in rule 1 above and
 * encoded positionally where the engine builds its face geometry. It used to also be
 * an exported `FACET` array here that nothing imported: a constant declaring a
 * contract the code enforced by index was the header's own promise broken in place.
 *
 * Below what height a solid stops being drawn, picked, or labelled. One decision —
 * "this slab is too thin to exist on screen" — that the engine used to state as a
 * bare 0.004 at four separate sites, against this header's rule that nothing here is
 * repeated as a literal there. A morph collapses absent parts toward zero height, so
 * the threshold is what keeps their faces, hit-targets and labels from lingering as
 * one-pixel slivers.
 */
export const MIN_VISIBLE_HEIGHT = 0.004;

export type PlateName = "neutral" | "cyan" | "magenta" | "yellow" | "ink";
export type Tier = "region" | "part" | "flow";
export type Side = "left" | "right";

/** A plate whose three steps are tokens, and the same plate one step deeper. */
type StepPlate = { steps: readonly string[]; sel: readonly string[]; tint?: never };
/** A plate held as a single token, tinted against the paper. */
type Tint = { ink: string; ground: string; steps: readonly number[] };
type TintPlate = { tint: Tint; selTint: Tint; steps?: never };

/**
 * The four plates, as token names — never hexes, because the sheet owns the values.
 * `--color-process-yellow` is the site's existing process yellow; the figure adds no
 * colour of its own.
 */
export const PLATES: Record<PlateName, StepPlate | TintPlate> = {
  neutral: {
    steps: ["--color-neutral-100", "--color-neutral-300", "--color-neutral-500"],
    sel: ["--color-neutral-300", "--color-neutral-500", "--color-neutral-700"],
  },
  cyan: {
    steps: ["--color-accent-100", "--color-accent-200", "--color-accent-300"],
    sel: ["--color-accent-200", "--color-accent-300", "--color-accent-500"],
  },
  magenta: {
    steps: ["--color-accent-2-100", "--color-accent-2-200", "--color-accent-2-300"],
    sel: ["--color-accent-2-200", "--color-accent-2-300", "--color-accent-2-500"],
  },
  // The process yellow ships as one token, so its three steps are press tints of it
  // against the paper — the only derived colour in the figure.
  yellow: {
    tint: { ink: "--color-process-yellow", ground: "--color-bg", steps: [0.16, 0.34, 0.58] },
    selTint: { ink: "--color-process-yellow", ground: "--color-bg", steps: [0.34, 0.58, 1] },
  },
  ink: {
    steps: ["--color-neutral-700", "--color-neutral-800", "--color-neutral-900"],
    sel: ["--color-neutral-800", "--color-neutral-900", "--color-text"],
  },
};

/** Role → ink. Add the block to a role; do not add a hue. */
export const ROLE_INK: Record<string, PlateName> = {
  tok_emb: "cyan", // the embedding matrix …
  pos_emb: "cyan",
  head: "cyan", // … and its transpose: same tensor, same ink
  attn: "magenta",
  kv: "magenta", // the cache belongs to attention
  ffn: "yellow",
  norm: "neutral",
  final_norm: "neutral",
  block: "neutral", // a region label over parts that carry their own inks
  inputs: "neutral", // flow, not a layer
  logits: "neutral",
  residual: "ink", // structure, permanently ink
  tie: "ink", // an annotation, not a part
};

export const EDGE = { hairline: 0.8, selected: 2, doubleOffset: 0.013 };

/** Leader geometry, in px of the figure's own frame. */
export const LEADER = {
  marginGap: 24, // dot-to-type clearance at the margin
  bendRun: 15, // length of the final horizontal into the type
  stackGap: 9, // minimum air between two labels in one margin
  dot: { rest: 1.6, active: 2.4 },
  weight: { rest: 0.8, selected: 1.2, hot: 1.4 },
};

/**
 * Which margin each label rules out to. Left is the input side of the stack;
 * everything the block is made of goes right.
 */
export const SIDES: Record<Side, readonly string[]> = {
  left: ["inputs", "tok_emb", "pos_emb", "residual"],
  right: ["norm_a", "attn", "kv", "norm_b", "ffn", "block", "tie", "final_norm", "head", "logits"],
};

/** The three tiers, and what each is for. */
export const TIERS: Record<Tier, string> = {
  region: "the parts a reader should take in first — whole regions of the stack",
  part: "the sub-parts inside the block",
  flow: "markers that are not layers: what enters, what leaves, what runs through",
};

export const MOTION = {
  rest: { a: 0.56, e: 0.38 },
  clampA: [0.1, 1.24] as const,
  clampE: [0.06, 0.62] as const,
  returnEase: 0.06,
  idleBeforeReturn: 900,
  morphMs: 340,
};

export const inkFor = (blockId: string): PlateName => ROLE_INK[blockId] ?? "neutral";

/**
 * Resolve a plate into three concrete tones. `read` is a token reader and `blend`
 * mixes two hexes — both injected so this file stays value-free.
 */
export function facetSet(
  plate: PlateName,
  selected: boolean,
  read: (token: string) => string,
  blend: (a: string, b: string, t: number) => string,
): string[] {
  const p = PLATES[plate];
  if ("tint" in p && p.tint) {
    const t = selected ? p.selTint : p.tint;
    return t.steps.map((s) => blend(read(t.ground), read(t.ink), s));
  }
  const stepped = p as StepPlate;
  return (selected ? stepped.sel : stepped.steps).map(read);
}

export type LabelSpec = { key: string; blockId: string; tier: Tier; side?: Side };

/**
 * Drift detector: warn the moment a block, label or tier is added without being given
 * a rule here. Cheap, and it is the reason this file exists — a silent fallback is how
 * a figure like this rots.
 */
export function validate(labels: readonly LabelSpec[], blockIds: readonly string[]): void {
  const warn = (m: string) => console.warn(`[figure V] ${m}`);
  labels.forEach((L) => {
    if (!TIERS[L.tier]) warn(`label "${L.key}" has no tier — see TIERS`);
    const side: Side = L.side === "left" ? "left" : "right";
    if (!SIDES[side].includes(L.key)) {
      warn(`label "${L.key}" is not assigned to the ${side} margin — see SIDES`);
    }
    if (!ROLE_INK[L.blockId]) {
      warn(`block "${L.blockId}" has no role ink — add it to ROLE_INK, do not add a hue`);
    }
  });
  blockIds.forEach((b) => {
    if (b !== "whole" && !ROLE_INK[b]) warn(`block "${b}" is drawn with no role ink — see ROLE_INK`);
  });
}
