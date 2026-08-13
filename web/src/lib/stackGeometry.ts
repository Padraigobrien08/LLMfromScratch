import { ARCHITECTURES } from "../content/architecture";
import { SIZES, type Variant } from "../content/blocks";
import { parameters } from "./modelsize";

/**
 * The geometry of Figure V, derived from the shipped config and the parameter counts.
 *
 * One source of solids, consumed by both the WebGL figure and the SVG plate, so the
 * two cannot disagree about what the model looks like.
 *
 * The figure's central claim is that **thickness is the parameter budget**: a slab's
 * height is that part's share of the total, which is why the token embedding is
 * visibly a third of the object. That only holds if the shares come from the same
 * arithmetic as the rest of the site, so every number below is read from
 * `parameters()` — fixture-pinned to the real `Transformer` — and nothing here is
 * typed by hand.
 */

/** Everything the figure draws, at the granularity the drawing needs. */
export type PartId =
  | "inputs"
  | "tok_emb"
  | "pos_emb"
  | "norm_a"
  | "attn"
  | "norm_b"
  | "ffn"
  | "final_norm"
  | "head"
  | "logits";

const H = 3.0; // total height allotted to the parameterised slabs
const MIN = 0.062; // floor for slabs that hold no weights of their own
const GAP = 0.05;
const RGAP = 0.16; // between regions
const W = 2.2;
const NORMW = 0.84;
const ZS = 0.3; // one layer of depth
const DL = 0.17; // a layer slab's own depth — the rest is real gap, so you can count them
const SPINE_X = -W / 2 - 0.3;

/** Parts drawn at the floor thickness rather than vanishing: flow, and the tied head. */
const ZERO_MIN = new Set<PartId>(["inputs", "logits", "head"]);

/**
 * Parameters per drawn part.
 *
 * `parameters()` aggregates all `2·n_layer + 1` norms into one figure because that is
 * the useful granularity everywhere else on the site. The figure draws them in three
 * places, so the aggregate is split by count — exact, since every norm in the stack is
 * the same size — and the parts still sum to `total`.
 */
export function figureParts(variant: Variant): Record<PartId, number> {
  const p = parameters(SIZES[variant]);
  const layers = ARCHITECTURES[variant].config.nLayer;
  const perNorm = p.norms / (2 * layers + 1);
  return {
    inputs: 0,
    tok_emb: p.tokenEmbedding,
    pos_emb: p.positionEmbedding,
    norm_a: perNorm * layers,
    attn: p.attention,
    norm_b: perNorm * layers,
    ffn: p.feedForward,
    final_norm: perNorm,
    head: p.lmHead,
    logits: 0,
  };
}

export type Solid = {
  id: string;
  blockId: string;
  xc: number;
  yc: number;
  zc: number;
  w: number;
  h: number;
  d: number;
  /** `face` and `ghost` take the role ink; `ink` is structure; `hidden` is a hit target. */
  tone: "face" | "ghost" | "ink" | "hidden";
  pick: boolean;
  layer?: number;
};

export type Spec = {
  variant: Variant;
  solids: Solid[];
  anchors: Record<string, [number, number, number]>;
  height: number;
  depth: number;
  width: number;
  tieX: number;
  tieZ: number;
  tie: { top: [number, number, number]; bottom: [number, number, number] };
  rope: { xc: number; yc: number; zc: number; r: number; on: boolean };
};

export function buildSpec(variant: Variant): Spec {
  const cfg = ARCHITECTURES[variant].config;
  const parts = figureParts(variant);
  const total = parameters(SIZES[variant]).total;
  const layers = cfg.nLayer;
  const depthFull = layers * ZS;
  const zBase = -depthFull / 2;

  const th = (id: PartId) => {
    const n = parts[id];
    if (n > 0) return Math.max(MIN, (n / total) * H);
    return ZERO_MIN.has(id) ? MIN : 0;
  };

  const solids: Solid[] = [];
  const anchors: Record<string, [number, number, number]> = {};
  let y = 0;

  /** A slab spanning the full depth of the stack — the parts outside the block. */
  const base = (id: PartId, blockId: string, w: number, xc: number) => {
    const h = th(id);
    if (h > 0 && y > 0) y += GAP;
    const yc = y + h / 2;
    anchors[id] = [xc + w / 2, yc, zBase + depthFull / 2];
    solids.push({
      id,
      blockId,
      xc,
      yc,
      zc: zBase,
      w,
      h: h > 0 ? h : 0.0001,
      d: depthFull,
      tone: "face",
      pick: h > 0,
    });
    if (h > 0) y += h;
  };

  base("inputs", "inputs", W, 0);
  base("tok_emb", "tok_emb", W, 0);
  base("pos_emb", "pos_emb", W, 0);

  const yEmb = y;
  y += RGAP;

  /** A slab inside the block, sliced once per layer: the depth *is* the layer count. */
  const layerPart = (id: PartId, blockId: string, w: number, xc: number) => {
    const h = th(id);
    y += GAP;
    const yc = y + h / 2;
    for (let i = 0; i < layers; i++) {
      solids.push({
        id: `${id}@${i}`,
        blockId,
        xc,
        yc,
        zc: -(i + 0.5) * ZS,
        w,
        h,
        d: DL,
        tone: i === 0 ? "face" : "ghost",
        pick: true,
        layer: i,
      });
    }
    anchors[id] = [xc + w / 2, yc, -DL / 2];
    y += h;
    return yc;
  };

  const yNormA = layerPart("norm_a", "norm", NORMW, W / 2 - NORMW / 2);
  const yAttn = layerPart("attn", "attn", W, 0);

  // The cache is a drawer bolted to the side of every attention slab — not a layer,
  // so it is sized off attention rather than off a parameter count it does not have.
  const kvH = th("attn") * 0.62;
  for (let i = 0; i < layers; i++) {
    solids.push({
      id: `kv@${i}`,
      blockId: "kv",
      xc: W / 2 + 0.46,
      yc: yAttn,
      zc: -(i + 0.5) * ZS,
      w: 0.64,
      h: kvH,
      d: DL,
      tone: "ghost",
      pick: true,
      layer: i,
    });
  }
  anchors.kv = [W / 2 + 0.78, yAttn, -DL / 2];

  const yNormB = layerPart("norm_b", "norm", NORMW, W / 2 - NORMW / 2);
  const yFfn = layerPart("ffn", "ffn", W, 0);
  anchors.block = [W / 2, yAttn, -(layers - 0.5) * ZS];

  y += RGAP;
  base("final_norm", "final_norm", NORMW, W / 2 - NORMW / 2);
  base("head", "head", W, 0);
  base("logits", "logits", W, 0);
  const top = y;

  // The spine: unbroken from the top of the embeddings to the head. Ink, always —
  // it is structure, and rule 3 keeps selection from ever recolouring it.
  solids.push({
    id: "spine",
    blockId: "residual",
    xc: SPINE_X,
    yc: (yEmb + top) / 2,
    zc: -0.16,
    w: 0.09,
    h: top - yEmb,
    d: 0.09,
    tone: "ink",
    pick: true,
  });
  anchors.residual = [SPINE_X - 0.045, (yEmb + top) * 0.34, -0.115];

  // Taps and adds: the norms read from the stream beside it, the sublayers add back
  // into it. Pre-norm is visible precisely because these cross and the spine does not.
  const bar = (id: string, yc: number) =>
    solids.push({
      id,
      blockId: "residual",
      xc: (SPINE_X - W / 2) / 2,
      yc,
      w: Math.abs(SPINE_X + W / 2),
      h: 0.038,
      d: 0.07,
      zc: -0.16,
      tone: "ink",
      pick: true,
    });
  bar("tap_a", yNormA);
  bar("add_a", yAttn + th("attn") / 2 + 0.055);
  bar("tap_b", yNormB);
  bar("add_b", yFfn + th("ffn") / 2 + 0.055);

  // The tie: a dashed annotation lying in front of the tower, not a part of it. The
  // solid here is only its hit target.
  const TIE_X = 2.02;
  const TIE_Z = 0.22;
  const tieMid = (anchors.tok_emb![1] + anchors.head![1]) / 2;
  solids.push({
    id: "tie_hit",
    blockId: "tie",
    xc: TIE_X,
    yc: tieMid,
    w: 0.11,
    h: Math.abs(anchors.head![1] - anchors.tok_emb![1]),
    d: 0.11,
    zc: TIE_Z,
    tone: "hidden",
    pick: true,
  });
  anchors.tie = [TIE_X + 0.055, tieMid + 0.12, TIE_Z];

  return {
    variant,
    solids,
    anchors,
    height: top,
    depth: depthFull,
    width: W,
    tieX: TIE_X,
    tieZ: TIE_Z,
    tie: {
      top: [W / 2, anchors.head![1], TIE_Z],
      bottom: [W / 2, anchors.tok_emb![1], TIE_Z],
    },
    rope: {
      xc: 0,
      yc: yAttn,
      zc: -DL / 2,
      r: Math.min(0.17, th("attn") * 0.42),
      on: cfg.posEmb === "rope",
    },
  };
}
