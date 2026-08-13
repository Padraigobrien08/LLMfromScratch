import type * as ThreeNS from "three";

import { ARCHITECTURES } from "../../content/architecture";
import type { Variant } from "../../content/blocks";
import { FIGURE_LABELS, labelText, type FigureLabel } from "../../content/stackFigure";
import { buildSpec, type Solid, type Spec } from "../../lib/stackGeometry";
import {
  EDGE,
  LEADER,
  MOTION,
  PLATES,
  facetSet,
  inkFor,
  validate,
  type PlateName,
} from "./figureRules";

/**
 * The figure itself: solids, leaders, labels, and the orbit.
 *
 * Imperative on purpose. The panel, the plate chrome and the segmented control are
 * React's — they are content — but this half redraws every frame while the reader
 * drags, and positions fourteen labels against a projection that changes with it.
 * Putting that through React state would mean a re-render per frame for no gain.
 *
 * The engine owns no rules. Facet order, role inks, leader geometry, label tiers and
 * the camera limits all live in `figureRules.ts`; this file draws them.
 */

type Mode = "svg" | "gl";

export type StackEngineOptions = {
  canvasWrap: HTMLElement;
  flat: SVGSVGElement;
  leaders: SVGSVGElement;
  labelLayer: HTMLElement;
  variant: Variant;
  selected: string;
  onSelect: (blockId: string) => void;
  onAnnounce: (message: string) => void;
  onMode: (note: string) => void;
};

export type StackEngine = {
  setVariant: (variant: Variant) => void;
  setSelected: (blockId: string) => void;
  destroy: () => void;
};

const VARIANTS: Variant[] = ["gpt2", "llama"];

/** Below this width the labels leave the margins and become a list — see `layoutLabels`.
 *  Kept in step with the matching breakpoint in `styles.css`. */
const LIST_BELOW = 700;

export function mountStackFigure(o: StackEngineOptions): StackEngine {
  const { canvasWrap, flat, leaders, labelLayer } = o;

  const css = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const INK = css("--color-text");
  const ACC2 = css("--color-accent-2");

  /** Mix two hexes. The only place the figure computes a colour rather than reading one. */
  const blend = (a: string, b: string, t: number) => {
    const parse = (s: string) => [1, 3, 5].map((i) => parseInt(s.slice(i, i + 2), 16));
    const [r1, g1, b1] = parse(a);
    const [r2, g2, b2] = parse(b);
    const c = (x: number, y: number) =>
      Math.round(x + (y - x) * t)
        .toString(16)
        .padStart(2, "0");
    return `#${c(r1!, r2!)}${c(g1!, g2!)}${c(b1!, b2!)}`;
  };

  const RAMP = {} as Record<PlateName, string[]>;
  const RAMP_SEL = {} as Record<PlateName, string[]>;
  (Object.keys(PLATES) as PlateName[]).forEach((k) => {
    RAMP[k] = facetSet(k, false, css, blend);
    RAMP_SEL[k] = facetSet(k, true, css, blend);
  });

  /** A solid drawn as structure — the spine and its taps — is ink whatever its role. */
  const inkOf = (s: { tone: string; blockId: string }): PlateName =>
    s.tone === "ink" ? "ink" : inkFor(s.blockId);

  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const REST = MOTION.rest;

  const SPECS: Record<Variant, Spec> = { gpt2: buildSpec("gpt2"), llama: buildSpec("llama") };
  const MAPS: Record<Variant, Map<string, Solid>> = {
    gpt2: new Map(SPECS.gpt2.solids.map((s) => [s.id, s])),
    llama: new Map(SPECS.llama.solids.map((s) => [s.id, s])),
  };
  const idUnion = [...new Set(VARIANTS.flatMap((v) => SPECS[v].solids.map((s) => s.id)))];

  let archA: Variant = o.variant;
  let archB: Variant = o.variant;
  let mix = 1;
  let selected = o.selected;
  let hovered: string | null = null;
  let mode: Mode = "svg";
  let disposed = false;

  const collapsed = (s: Solid): Solid => ({ ...s, h: 0.0001, pick: false });

  /** One solid, interpolated between the two architectures mid-morph. */
  function solidAt(id: string): Solid {
    const a = MAPS[archA].get(id);
    const b = MAPS[archB].get(id);
    const A = a ?? collapsed(b!);
    const B = b ?? collapsed(a!);
    const L = (k: "xc" | "yc" | "zc" | "w" | "h" | "d") => A[k] + (B[k] - A[k]) * mix;
    return {
      id,
      blockId: B.blockId,
      tone: B.tone,
      pick: B.pick && L("h") > 0.004,
      xc: L("xc"),
      yc: L("yc"),
      zc: L("zc"),
      w: L("w"),
      h: Math.max(L("h"), 0.0001),
      d: L("d"),
    };
  }

  const curVariant = () => (mix > 0.5 ? archB : archA);
  const curCfg = () => ARCHITECTURES[curVariant()].config;
  const curSpec = () => SPECS[curVariant()];
  const curHeight = () => SPECS[archA].height + (SPECS[archB].height - SPECS[archA].height) * mix;

  function bounds() {
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    let z0 = Infinity;
    let z1 = -Infinity;
    idUnion.forEach((id) => {
      const s = solidAt(id);
      if (s.h < 0.004) return;
      x0 = Math.min(x0, s.xc - s.w / 2);
      x1 = Math.max(x1, s.xc + s.w / 2);
      y0 = Math.min(y0, s.yc - s.h / 2);
      y1 = Math.max(y1, s.yc + s.h / 2);
      z0 = Math.min(z0, s.zc - s.d / 2);
      z1 = Math.max(z1, s.zc + s.d / 2);
    });
    return { x0, x1, y0, y1, z0, z1 };
  }

  // Selection and hover, with the tie coupling its two ends: touch the strand and both
  // the head and the embedding light, because they are one tensor.
  const spread = (id: string | null): string[] =>
    id === "tie"
      ? ["tie", "head", "tok_emb"]
      : id === "head" || id === "tok_emb"
        ? [id, "tie"]
        : id
          ? [id]
          : [];
  const isSel = (b: string) => spread(selected).includes(b);
  const isHot = (b: string) => !!hovered && spread(hovered).includes(b);

  /* ─── labels ───────────────────────────────────────────────────────────── */

  const anchorId: Record<string, string> = {
    norm_a: "norm_a@0",
    attn: "attn@0",
    norm_b: "norm_b@0",
    ffn: "ffn@0",
    kv: "kv@0",
  };

  const labelEls = new Map<string, HTMLButtonElement>();
  FIGURE_LABELS.forEach((L) => {
    const b = document.createElement("button");
    b.type = "button";
    b.className = `lab lab-${L.tier}${L.side === "left" ? " lab-left" : ""}`;
    b.dataset.key = L.key;
    b.innerHTML = '<span class="lab-t"></span>';
    b.addEventListener("click", () => choose(L.blockId, true));
    b.addEventListener("pointerenter", () => {
      hovered = L.blockId;
      paint();
    });
    b.addEventListener("pointerleave", () => {
      hovered = null;
      paint();
    });
    b.addEventListener("focus", () => {
      hovered = L.blockId;
      paint();
    });
    b.addEventListener("blur", () => {
      hovered = null;
      paint();
    });
    b.addEventListener("keydown", (e) => {
      if (e.key === "ArrowUp" || e.key === "ArrowDown") {
        e.preventDefault();
        // Up walks up the stack, which is down the list: the labels are ordered
        // bottom-first so the leaders cannot invert.
        const vis = FIGURE_LABELS.filter((x) => labelEls.get(x.key)!.dataset.on === "1");
        const i = vis.findIndex((x) => x.key === L.key);
        const n = vis[i + (e.key === "ArrowUp" ? 1 : -1)];
        if (n) {
          labelEls.get(n.key)!.focus();
          choose(n.blockId, false);
        }
      } else if (e.key === "Escape") {
        choose("whole", false);
      }
    });
    labelEls.set(L.key, b);
    labelLayer.appendChild(b);
  });

  function labelAnchor(L: FigureLabel): [number, number, number] {
    const c = curCfg();
    const left = L.side === "left";
    if (L.key === "residual") {
      const s = solidAt("spine");
      return [s.xc - s.w / 2, s.yc - s.h * 0.16, s.zc + s.d / 2];
    }
    if (L.key === "tie") return curSpec().anchors.tie!;
    if (L.key === "block") {
      const s = solidAt(`attn@${c.nLayer - 1}`);
      return [s.xc + s.w / 2, s.yc, s.zc + s.d / 2];
    }
    if (L.key === "pos_emb" && c.posEmb === "rope") {
      const s = solidAt("attn@0");
      return [s.xc - s.w * 0.3, s.yc - s.h / 2, s.zc + s.d / 2];
    }
    const s = solidAt(anchorId[L.key] ?? L.key);
    const y = L.key === "attn" ? s.yc + s.h * 0.34 : s.yc;
    return [left ? s.xc - s.w / 2 : s.xc + s.w / 2, y, s.zc + s.d / 2];
  }

  /** The learned position table is a slab; the rotary one is a ring inside attention. */
  const labelVisible = (L: FigureLabel) =>
    L.key === "pos_emb" && curCfg().posEmb === "learned" ? solidAt("pos_emb").h > 0.004 : true;

  type Placed = {
    L: FigureLabel;
    el: HTMLButtonElement;
    ax: number;
    ay: number;
    y: number;
    eh: number;
    left: boolean;
  };

  function layoutLabels(project: (p: [number, number, number]) => [number, number], rect: { w: number; h: number }) {
    const variant = curVariant();
    const items: Placed[] = [];
    FIGURE_LABELS.forEach((L) => {
      const el = labelEls.get(L.key)!;
      const on = labelVisible(L);
      el.dataset.on = on ? "1" : "0";
      el.style.display = on ? "" : "none";
      el.disabled = !on;
      if (!on) return;
      el.querySelector(".lab-t")!.textContent = labelText(L, variant);
      const [ax, ay] = project(labelAnchor(L));
      items.push({ L, el, ax, ay, y: ay, eh: el.offsetHeight || 16, left: L.side === "left" });
    });

    // Within a margin: sorted by the height of the anchor, then pushed apart just far
    // enough to clear each other. Sorting first is what stops two leaders crossing.
    const place = (list: Placed[]) => {
      list.sort((p, q) => p.ay - q.ay);
      const push = () => {
        for (let i = 1; i < list.length; i++) {
          const need = list[i - 1]!.y + (list[i - 1]!.eh + list[i]!.eh) / 2 + LEADER.stackGap;
          if (list[i]!.y < need) list[i]!.y = need;
        }
      };
      push();
      const last = list[list.length - 1];
      const over = last ? last.y + last.eh / 2 - (rect.h - 4) : 0;
      if (over > 0) list.forEach((p) => (p.y -= over));
      push();
      list.forEach((p) => (p.y = Math.max(p.eh / 2 + 2, p.y)));
    };
    // A narrow column cannot hold two margins of type either side of an object: the
    // gutters eat the width and the drawing collapses to a speck with fourteen leaders
    // converging on it. Below the breakpoint the labels stop being ruled out and become
    // a list under the figure — still buttons, still tiered, still selecting. The
    // drawing takes the full width instead.
    if (matchMedia(`(max-width: ${LIST_BELOW}px)`).matches) {
      labelLayer.dataset.layout = "list";
      items.forEach((p) => {
        p.el.style.top = "";
        p.el.style.left = "";
        p.el.style.right = "";
        p.el.classList.toggle("is-sel", isSel(p.L.blockId));
        p.el.classList.toggle("is-hot", isHot(p.L.blockId));
      });
      leaders.innerHTML = "";
      return;
    }
    labelLayer.dataset.layout = "margins";

    place(items.filter((p) => !p.left));
    place(items.filter((p) => p.left));

    let g = "";
    const cl = canvasWrap.offsetLeft;
    const ct = canvasWrap.offsetTop;
    const fw = labelLayer.clientWidth;
    items.forEach((p) => {
      const right = !p.left;
      const lx = right ? rect.w + LEADER.marginGap : -LEADER.marginGap;
      p.el.style.top = `${p.y + ct}px`;
      if (right) {
        p.el.style.left = `${cl + rect.w + LEADER.marginGap}px`;
        p.el.style.right = "auto";
      } else {
        p.el.style.right = `${Math.min(fw - cl + LEADER.marginGap, fw - p.el.offsetWidth - 2)}px`;
        p.el.style.left = "auto";
      }
      const sel = isSel(p.L.blockId);
      const hot = isHot(p.L.blockId);
      p.el.classList.toggle("is-sel", sel);
      p.el.classList.toggle("is-hot", hot);
      const col = hot ? ACC2 : INK;
      const bend = right ? lx - LEADER.bendRun : lx + LEADER.bendRun;
      const w = hot ? LEADER.weight.hot : sel ? LEADER.weight.selected : LEADER.weight.rest;
      g +=
        `<path d="M ${p.ax} ${p.ay} L ${bend} ${p.y} L ${lx} ${p.y}" fill="none" stroke="${col}" stroke-width="${w}"/>` +
        `<circle cx="${p.ax}" cy="${p.ay}" r="${hot || sel ? LEADER.dot.active : LEADER.dot.rest}" fill="${col}"/>`;
    });
    leaders.innerHTML = g;
  }

  function choose(blockId: string, focusLabel: boolean) {
    selected = blockId;
    o.onSelect(blockId);
    paint();
    if (focusLabel) {
      const L = FIGURE_LABELS.find((x) => x.blockId === blockId);
      if (L && labelEls.get(L.key)!.dataset.on === "1") labelEls.get(L.key)!.focus();
    }
  }

  /* ─── the flat plate: the same solids, projected axonometrically ────────── */

  function axo(a: number, e: number) {
    const ca = Math.cos(a);
    const sa = Math.sin(a);
    const ce = Math.cos(e);
    const se = Math.sin(e);
    return ([x, y, z]: [number, number, number]): [number, number, number] => [
      x * ca - z * sa,
      -(y * ce - (x * sa + z * ca) * se),
      (x * sa + z * ca) * ce + y * se,
    ];
  }

  function renderFlat() {
    const rect = { w: canvasWrap.clientWidth, h: canvasWrap.clientHeight };
    if (rect.w === 0 || rect.h === 0) return;
    const P = axo(REST.a, REST.e);
    const solids = idUnion.map(solidAt).filter((s) => s.h > 0.004 && s.tone !== "hidden");

    const corners: Array<[number, number, number]> = [];
    solids.forEach((s) => {
      for (const dx of [-1, 1])
        for (const dy of [-1, 1])
          for (const dz of [-1, 1])
            corners.push(P([s.xc + (dx * s.w) / 2, s.yc + (dy * s.h) / 2, s.zc + (dz * s.d) / 2]));
    });
    const bb = bounds();
    const tx = curSpec().tieX;
    // The tie hangs outside every slab, so the frame has to be told about it or the
    // dashed strand is the one thing that gets clipped.
    ([[tx, bb.y0, bb.z1], [tx, bb.y1, bb.z1]] as Array<[number, number, number]>).forEach((v) =>
      corners.push(P(v)),
    );

    const xs = corners.map((p) => p[0]);
    const ys = corners.map((p) => p[1]);
    const bx = [Math.min(...xs), Math.max(...xs)];
    const by = [Math.min(...ys), Math.max(...ys)];
    const k = Math.min((rect.w - 20) / (bx[1]! - bx[0]!), (rect.h - 20) / (by[1]! - by[0]!));
    const ox = (rect.w - (bx[1]! - bx[0]!) * k) / 2 - bx[0]! * k;
    const oy = (rect.h - (by[1]! - by[0]!) * k) / 2 - by[0]! * k;
    const to = (v: [number, number, number]): [number, number] => {
      const p = P(v);
      return [p[0] * k + ox, p[1] * k + oy];
    };

    type Face = { pts: Array<[number, number]>; face: number; s: Solid; dep: number };
    const faces: Face[] = [];
    solids.forEach((s) => {
      const [x0, x1] = [s.xc - s.w / 2, s.xc + s.w / 2];
      const [y0, y1] = [s.yc - s.h / 2, s.yc + s.h / 2];
      const [z0, z1] = [s.zc - s.d / 2, s.zc + s.d / 2];
      // Three faces, in the contract's order: top, side, front.
      const quads: Array<[Array<[number, number, number]>, number]> = [
        [[[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]], 0],
        [[[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]], 2],
        [[[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], 1],
      ];
      const dep = P([s.xc, s.yc, s.zc])[2];
      quads.forEach(([q, face]) => faces.push({ pts: q.map(to), face, s, dep }));
    });
    faces.sort((a, b) => a.dep - b.dep);

    let g = "";
    faces.forEach((f) => {
      const sel = isSel(f.s.blockId);
      const set = (sel ? RAMP_SEL : RAMP)[inkOf(f.s)];
      g +=
        `<polygon points="${f.pts.map((p) => p.join(",")).join(" ")}" fill="${set[f.face]}" ` +
        `stroke="${INK}" stroke-width="${sel ? EDGE.selected : EDGE.hairline}" data-block="${f.s.blockId}" class="facet"/>`;
    });

    if (curSpec().rope.on) {
      const r = curSpec().rope;
      const cen = to([r.xc, r.yc, r.zc]);
      g += `<ellipse cx="${cen[0]}" cy="${cen[1]}" rx="${r.r * k}" ry="${r.r * k * 0.5}" fill="none" stroke="${INK}" stroke-width="1"/>`;
    }

    const t = curSpec().tie;
    const tz = curSpec().tieZ;
    const pts = (
      [t.top, [tx, t.top[1], tz], [tx, t.bottom[1], tz], t.bottom] as Array<[number, number, number]>
    ).map(to);
    g +=
      `<path d="M ${pts.map((p) => p.join(" ")).join(" L ")}" fill="none" stroke="${isHot("tie") ? ACC2 : INK}" ` +
      `stroke-width="1" stroke-dasharray="5 4"/>`;

    flat.setAttribute("viewBox", `0 0 ${rect.w} ${rect.h}`);
    flat.innerHTML = g;
    flat.querySelectorAll<SVGElement>(".facet").forEach((el) => {
      const block = el.dataset.block!;
      el.addEventListener("click", () => choose(block, true));
      el.addEventListener("pointerenter", () => {
        hovered = block;
        renderFlat();
      });
      el.addEventListener("pointerleave", () => {
        hovered = null;
        renderFlat();
      });
    });
    layoutLabels(to, rect);
  }

  /* ─── the WebGL figure ──────────────────────────────────────────────────── */

  let THREE: typeof ThreeNS | null = null;
  let renderer: ThreeNS.WebGLRenderer | null = null;
  let scene: ThreeNS.Scene;
  let camera: ThreeNS.OrthographicCamera;
  let group: ThreeNS.Group;
  let meshes: Map<string, ThreeNS.Mesh>;
  let geoms: Record<string, ThreeNS.BufferGeometry>;
  let tie: ThreeNS.Line;
  let ropeRing: ThreeNS.Mesh;
  let ray: ThreeNS.Raycaster;
  let ptr: ThreeNS.Vector2;
  const cam = { ...REST };
  let drag: { x: number; y: number; a: number; e: number; moved: boolean } | null = null;
  let lastInput = 0;
  let raf = 0;

  function initGL(T: typeof ThreeNS) {
    THREE = T;
    const canvas = document.createElement("canvas");
    canvas.className = "stack-gl";
    canvasWrap.appendChild(canvas);
    renderer = new T.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    scene = new T.Scene();
    camera = new T.OrthographicCamera(-1, 1, 1, -1, 0.1, 100);
    group = new T.Group();
    scene.add(group);

    // Three tones baked into the box's vertex colours — no light in the scene at all,
    // which is rule 1: the facets are printed, not lit.
    const tinted = (set: string[]) => {
      const g = new T.BoxGeometry(1, 1, 1);
      const cols: number[] = [];
      const C = (h: string) => new T.Color(h);
      [C(set[2]!), C(set[2]!), C(set[0]!), C(set[2]!), C(set[1]!), C(set[1]!)].forEach((col) => {
        for (let i = 0; i < 4; i++) cols.push(col.r, col.g, col.b);
      });
      g.setAttribute("color", new T.BufferAttribute(new Float32Array(cols), 3));
      return g;
    };
    geoms = {};
    (Object.keys(RAMP) as PlateName[]).forEach((k) => {
      geoms[k] = tinted(RAMP[k]);
      geoms[`${k}$sel`] = tinted(RAMP_SEL[k]);
    });

    const slabMat = new T.MeshBasicMaterial({ vertexColors: true });
    const hitMat = new T.MeshBasicMaterial({ transparent: true, opacity: 0, depthWrite: false });
    const edgeGeo = new T.EdgesGeometry(geoms.neutral!);
    const ink = new T.LineBasicMaterial({ color: INK });

    meshes = new Map();
    idUnion.forEach((id) => {
      const s = solidAt(id);
      const hidden = s.tone === "hidden";
      const m = new T.Mesh(geoms[hidden ? "neutral" : inkOf(s)]!, hidden ? hitMat : slabMat);
      m.name = id;
      m.userData.blockId = s.blockId;
      if (!hidden) {
        m.add(new T.LineSegments(edgeGeo, ink)); // the hairline
        const dbl = new T.LineSegments(edgeGeo, ink); // doubled, for the selected slab
        dbl.visible = false;
        m.add(dbl);
      }
      group.add(m);
      meshes.set(id, m);
    });

    ropeRing = new T.Mesh(new T.TorusGeometry(1, 0.05, 8, 40), new T.MeshBasicMaterial({ color: INK }));
    ropeRing.rotation.x = Math.PI / 2;
    group.add(ropeRing);

    tie = new T.Line(
      new T.BufferGeometry(),
      new T.LineDashedMaterial({ color: INK, dashSize: 0.085, gapSize: 0.06 }),
    );
    group.add(tie);

    ray = new T.Raycaster();
    ptr = new T.Vector2();

    canvasWrap.addEventListener("pointerdown", onDown);
    addEventListener("pointermove", onMove);
    addEventListener("pointerup", onUp);
    canvasWrap.addEventListener("pointerleave", onLeave);
    canvasWrap.addEventListener("click", onClick);
  }

  function pickAt(ev: PointerEvent | MouseEvent): string | null {
    const r = canvasWrap.getBoundingClientRect();
    ptr.set(((ev.clientX - r.left) / r.width) * 2 - 1, -((ev.clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ptr, camera);
    const hits = ray
      .intersectObjects(group.children, false)
      .filter((h) => (h.object as ThreeNS.Mesh).isMesh && h.object.visible);
    return hits.length ? ((hits[0]!.object.userData as { blockId?: string }).blockId ?? null) : null;
  }

  const onDown = (ev: PointerEvent) => {
    drag = { x: ev.clientX, y: ev.clientY, a: cam.a, e: cam.e, moved: false };
  };
  const onMove = (ev: PointerEvent) => {
    if (drag) {
      const dx = ev.clientX - drag.x;
      const dy = ev.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) drag.moved = true;
      cam.a = Math.min(MOTION.clampA[1], Math.max(MOTION.clampA[0], drag.a + dx * 0.005));
      cam.e = Math.min(MOTION.clampE[1], Math.max(MOTION.clampE[0], drag.e + dy * 0.004));
      lastInput = performance.now();
      paint();
    } else if (ev.target === canvasWrap || (ev.target as HTMLElement)?.classList?.contains("stack-gl")) {
      const was = hovered;
      hovered = pickAt(ev);
      canvasWrap.style.cursor = hovered ? "pointer" : "default";
      if (was !== hovered) paint();
    }
  };
  const onUp = () => {
    if (drag) {
      lastInput = performance.now();
      drag = null;
    }
  };
  const onLeave = () => {
    hovered = null;
    paint();
  };
  const onClick = (ev: MouseEvent) => {
    if (drag?.moved) return;
    const b = pickAt(ev);
    choose(b ?? "whole", !!b);
  };

  function updateGL() {
    if (!renderer || !THREE) return;
    const T = THREE;
    const h = curHeight();
    group.position.set(-0.3, -h / 2, curSpec().depth / 2);

    meshes.forEach((m, id) => {
      const s = solidAt(id);
      m.visible = s.h > 0.0025;
      m.scale.set(s.w, s.h, s.d);
      const lift = isHot(s.blockId) && s.tone !== "ink" ? 0.07 : 0;
      m.position.set(s.xc + lift, s.yc, s.zc + lift * 0.6);
      m.userData.blockId = s.blockId;
      if (s.tone === "hidden") return;
      const sel = isSel(s.blockId);
      m.geometry = geoms[inkOf(s) + (sel ? "$sel" : "")]!;
      const dbl = m.children[1] as ThreeNS.LineSegments | undefined;
      if (!dbl) return;
      dbl.visible = sel;
      const k = EDGE.doubleOffset; // rule 3: the selected slab's edge reads doubled
      if (sel) dbl.scale.set(1 + k / s.w, 1 + k / s.h, 1 + k / s.d);
    });

    const r = curSpec().rope;
    const ropeOn = archB === "llama" ? mix : 1 - mix;
    ropeRing.visible = ropeOn > 0.02;
    ropeRing.scale.setScalar(Math.max(0.001, r.r * ropeOn));
    ropeRing.position.set(r.xc, r.yc, r.zc + 0.03);

    const t = curSpec().tie;
    const tx = curSpec().tieX;
    const tz = curSpec().tieZ;
    tie.geometry.setFromPoints(
      ([t.top, [tx, t.top[1], tz], [tx, t.bottom[1], tz], t.bottom] as Array<[number, number, number]>).map(
        (p) => new T.Vector3(p[0], p[1], p[2]),
      ),
    );
    (tie as ThreeNS.Line).computeLineDistances();
    (tie.material as ThreeNS.LineDashedMaterial).color.set(isHot("tie") ? ACC2 : INK);

    const rect = { w: canvasWrap.clientWidth, h: canvasWrap.clientHeight };
    if (rect.w === 0 || rect.h === 0) return;
    if (rect.w !== renderer.domElement.clientWidth || rect.h !== renderer.domElement.clientHeight) {
      renderer.setSize(rect.w, rect.h, false);
    }

    const aspect = rect.w / rect.h;
    const R = 20;
    camera.position.set(
      R * Math.cos(cam.e) * Math.sin(cam.a),
      R * Math.sin(cam.e),
      R * Math.cos(cam.e) * Math.cos(cam.a),
    );
    camera.lookAt(0, 0, 0);
    camera.updateMatrixWorld();
    group.updateMatrixWorld();

    // Fit the frame to the object at this orbit, so the stack never drifts out of its
    // fixed height as the reader drags.
    const bb = bounds();
    let x0 = Infinity;
    let x1 = -Infinity;
    let y0 = Infinity;
    let y1 = -Infinity;
    const inv = camera.matrixWorldInverse;
    const tmp = new T.Vector3();
    for (const bx of [bb.x0, Math.max(bb.x1, tx)])
      for (const by of [bb.y0, bb.y1])
        for (const bz of [bb.z0, bb.z1]) {
          tmp.set(bx, by, bz).applyMatrix4(group.matrixWorld).applyMatrix4(inv);
          x0 = Math.min(x0, tmp.x);
          x1 = Math.max(x1, tmp.x);
          y0 = Math.min(y0, tmp.y);
          y1 = Math.max(y1, tmp.y);
        }
    const viewH = Math.max(y1 - y0, (x1 - x0) / aspect) * 1.04;
    const viewW = viewH * aspect;
    const mx = (x0 + x1) / 2;
    const my = (y0 + y1) / 2;
    camera.left = mx - viewW / 2;
    camera.right = mx + viewW / 2;
    camera.top = my + viewH / 2;
    camera.bottom = my - viewH / 2;
    camera.updateProjectionMatrix();
    renderer.render(scene, camera);

    const v = new T.Vector3();
    layoutLabels((p) => {
      v.set(p[0], p[1], p[2]).applyMatrix4(group.matrixWorld).project(camera);
      return [(v.x * 0.5 + 0.5) * rect.w, (-v.y * 0.5 + 0.5) * rect.h];
    }, rect);
  }

  function paint() {
    if (disposed) return;
    if (mode === "gl" && renderer) updateGL();
    else renderFlat();
  }

  /* ─── state, transition, loop ───────────────────────────────────────────── */

  let tween: { t0: number; ms: number } | null = null;

  function setVariant(key: Variant) {
    if (curVariant() === key) return;
    archA = curVariant();
    archB = key;
    mix = 0;
    if (reduced) {
      mix = 1;
      paint();
      return;
    }
    tween = { t0: performance.now(), ms: MOTION.morphMs };
    const step = () => {
      if (!tween || disposed) return;
      const p = Math.min(1, (performance.now() - tween.t0) / tween.ms);
      mix = p * p * (3 - 2 * p);
      paint();
      if (p < 1) requestAnimationFrame(step);
      else {
        mix = 1;
        tween = null;
        paint();
      }
    };
    step();
  }

  function loop() {
    if (disposed) return;
    if (
      !drag &&
      !reduced &&
      performance.now() - lastInput > MOTION.idleBeforeReturn &&
      Math.abs(cam.a - REST.a) + Math.abs(cam.e - REST.e) > 0.0006
    ) {
      cam.a += (REST.a - cam.a) * MOTION.returnEase;
      cam.e += (REST.e - cam.e) * MOTION.returnEase;
      updateGL();
    }
    raf = requestAnimationFrame(loop);
  }

  /* ─── flat first, upgrade in place ──────────────────────────────────────── */

  function canGL() {
    if (reduced || innerWidth < 768) return false;
    try {
      const c = document.createElement("canvas");
      return !!(c.getContext("webgl2") || c.getContext("webgl"));
    } catch {
      return false;
    }
  }

  validate(FIGURE_LABELS, [...new Set(SPECS.gpt2.solids.map((s) => s.blockId)), "tie"]);
  renderFlat();

  const FLAT_NOTE = "Flat axonometric plate — every label, explainer and link, without the orbit.";
  o.onMode(
    reduced
      ? "Reduced motion: the flat plate. Click a block for its shape, its share of the budget and the test that pins it."
      : FLAT_NOTE,
  );

  const onResize = () => paint();
  addEventListener("resize", onResize);

  if (canGL()) {
    // Dynamic, so three.js lands in its own chunk and never blocks the headline.
    import("three")
      .then((T) => {
        if (disposed) return;
        initGL(T);
        mode = "gl";
        flat.style.display = "none";
        o.onMode(
          "Drag to orbit; scrolling here scrolls the page. Click a block for its shape, its share of the budget and the test that pins it.",
        );
        updateGL();
        loop();
      })
      .catch(() => {
        if (disposed) return;
        mode = "svg";
        flat.style.display = "";
        canvasWrap.querySelector(".stack-gl")?.remove();
        o.onMode(FLAT_NOTE);
        renderFlat();
      });
  }

  return {
    setVariant,
    setSelected: (blockId: string) => {
      if (selected === blockId) return;
      selected = blockId;
      paint();
    },
    destroy: () => {
      disposed = true;
      cancelAnimationFrame(raf);
      removeEventListener("resize", onResize);
      removeEventListener("pointermove", onMove);
      removeEventListener("pointerup", onUp);
      canvasWrap.removeEventListener("pointerdown", onDown);
      canvasWrap.removeEventListener("pointerleave", onLeave);
      canvasWrap.removeEventListener("click", onClick);
      labelEls.forEach((el) => el.remove());
      renderer?.dispose();
      Object.values(geoms ?? {}).forEach((g) => g.dispose());
      canvasWrap.querySelector(".stack-gl")?.remove();
      flat.innerHTML = "";
      leaders.innerHTML = "";
    },
  };
}
