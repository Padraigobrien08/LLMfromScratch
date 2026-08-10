/**
 * The analysis half of `src/llmfs/ablation/report.py`, ported so the page can
 * recompute a comparison as toggles change instead of shipping a frozen table.
 *
 * The rules it mirrors, which are the whole point of the study:
 *   - compare within a seed, never between means, wherever the seeds are shared;
 *   - an arm counts only when the range of its per-seed deltas excludes zero;
 *   - with no shared seeds, fall back to clearing the baseline's whole spread, and
 *     say so.
 * Softening any of those in the browser would let the site report results the
 * repository's own report refuses to.
 */

export type ArmRun = {
  name: string;
  seed: number;
  status: "completed" | "diverged" | "failed";
  val_loss: number | null;
  perplexity: number | null;
  steps: number;
  tokens: number;
  wall_clock_s: number;
  tokens_per_sec: number;
  params: number;
  error: string | null;
  history: Array<{ step: number; val_loss: number }>;
};

export type Payload = { meta: Record<string, unknown>; arms: ArmRun[] };

export type Comparison = {
  name: string;
  axis: string;
  status: string;
  valLoss: number | null;
  delta: number | null;
  deltas: number[];
  paired: boolean;
  significant: boolean;
  nSeeds: number;
  params: number;
  tokensPerSec: number;
  /** Half the spread of the paired deltas — the error bar on `delta`. */
  halfRange: number;
};

/** Kept identical to `report.AXIS` so the two never describe an arm differently. */
export const AXIS: Record<string, string> = {
  baseline: "—",
  "norm-rmsnorm": "LayerNorm → RMSNorm",
  "pos-rope": "learned positions → RoPE",
  "pos-none": "learned positions → none",
  "mlp-swiglu": "GELU → SwiGLU (param-matched)",
  "untied-embeddings": "tied → untied embeddings",
  "no-bias": "bias → no bias",
  "gqa-2": "8 KV heads → 2 (GQA)",
  "sched-wsd": "cosine → WSD schedule",
  "wd-zero": "weight decay 0.1 → 0",
  "lr-3e-4": "lr 1e-3 → 3e-4",
  "lr-3e-3": "lr 1e-3 → 3e-3",
  "modern-stack": "all modern components",
};

/** Exactly the arms `configs/ablations/modern-stack.yaml` combines. */
export const MODERN_STACK = ["norm-rmsnorm", "pos-rope", "mlp-swiglu", "gqa-2", "no-bias"] as const;

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

function completedBySeed(runs: ArmRun[]): Map<number, number> {
  const out = new Map<number, number>();
  for (const r of runs) {
    if (r.status === "completed" && r.val_loss !== null) out.set(r.seed, r.val_loss);
  }
  return out;
}

export function baselineNoise(arms: ArmRun[]): { mean: number | null; spread: number; n: number } {
  const losses = arms
    .filter((a) => a.name === "baseline" && a.status === "completed" && a.val_loss !== null)
    .map((a) => a.val_loss!);
  if (losses.length === 0) return { mean: null, spread: 0, n: 0 };
  if (losses.length === 1) return { mean: losses[0]!, spread: 0, n: 1 };
  return { mean: mean(losses), spread: Math.max(...losses) - Math.min(...losses), n: losses.length };
}

function isSignificant(delta: number, deltas: number[], paired: boolean, spread: number): boolean {
  // Paired: every seed had to agree on the direction. Unpaired: clear the whole
  // baseline spread, which is a far higher bar — and the reason pairing earns its cost.
  if (paired) return Math.min(...deltas) > 0 || Math.max(...deltas) < 0;
  return Math.abs(delta) > spread;
}

export function groupByName(arms: ArmRun[]): Map<string, ArmRun[]> {
  const out = new Map<string, ArmRun[]>();
  for (const a of arms) {
    const list = out.get(a.name);
    if (list) list.push(a);
    else out.set(a.name, [a]);
  }
  return out;
}

export function compare(payload: Payload): {
  rows: Comparison[];
  baseline: { mean: number | null; spread: number; n: number };
} {
  const arms = payload.arms;
  const baseline = baselineNoise(arms);
  const byName = groupByName(arms);
  const baselineBySeed = completedBySeed(byName.get("baseline") ?? []);

  const rows: Comparison[] = [];
  for (const [name, runs] of byName) {
    const armBySeed = completedBySeed(runs);
    const completed = runs.filter((r) => r.status === "completed" && r.val_loss !== null);
    const loss = armBySeed.size ? mean([...armBySeed.values()]) : null;

    let deltas: number[] = [];
    let paired = false;
    if (name !== "baseline" && armBySeed.size && baselineBySeed.size) {
      const shared = [...armBySeed.keys()].filter((s) => baselineBySeed.has(s)).sort();
      if (shared.length) {
        deltas = shared.map((s) => armBySeed.get(s)! - baselineBySeed.get(s)!);
        paired = shared.length > 1;
      }
    }

    let delta: number | null;
    let significant: boolean;
    if (deltas.length) {
      delta = mean(deltas);
      significant = isSignificant(delta, deltas, paired, baseline.spread);
    } else if (name === "baseline" || loss === null || baseline.mean === null) {
      delta = null;
      significant = false;
    } else {
      delta = loss - baseline.mean;
      significant = Math.abs(delta) > baseline.spread;
    }

    rows.push({
      name,
      axis: AXIS[name] ?? "—",
      status: completed.length ? "completed" : (runs[0]?.status ?? "failed"),
      valLoss: loss,
      delta,
      deltas,
      paired,
      significant,
      nSeeds: armBySeed.size,
      params: (completed[0] ?? runs[0])?.params ?? 0,
      tokensPerSec: mean(runs.map((r) => r.tokens_per_sec)),
      halfRange: deltas.length > 1 ? (Math.max(...deltas) - Math.min(...deltas)) / 2 : 0,
    });
  }

  rows.sort((a, b) => {
    if (a.name === "baseline") return -1;
    if (b.name === "baseline") return 1;
    return (a.delta ?? 0) - (b.delta ?? 0);
  });
  return { rows, baseline };
}

/**
 * The mean validation curve across an arm's seeds.
 *
 * Only steps every seed reached are included: averaging over a shrinking number of
 * runs makes the tail of the curve quietly change meaning, and a diverged arm would
 * drag it somewhere no single run ever went.
 */
export function meanCurve(runs: ArmRun[]): Array<{ step: number; loss: number }> {
  const usable = runs.filter((r) => r.status === "completed" && r.history.length);
  if (!usable.length) return [];
  const counts = new Map<number, number[]>();
  for (const run of usable) {
    for (const point of run.history) {
      const at = counts.get(point.step);
      if (at) at.push(point.val_loss);
      else counts.set(point.step, [point.val_loss]);
    }
  }
  return [...counts.entries()]
    .filter(([, losses]) => losses.length === usable.length)
    .map(([step, losses]) => ({ step, loss: mean(losses) }))
    .sort((a, b) => a.step - b.step);
}

/** What the toggles currently describe, and whether the sweep ever measured it. */
export type Selection =
  | { kind: "baseline" }
  | { kind: "arm"; name: string }
  | { kind: "combination"; name: "modern-stack" }
  | { kind: "unmeasured"; names: string[] };

export function resolveSelection(selected: string[]): Selection {
  if (selected.length === 0) return { kind: "baseline" };
  if (selected.length === 1) return { kind: "arm", name: selected[0]! };
  const set = new Set(selected);
  const isModern =
    set.size === MODERN_STACK.length && MODERN_STACK.every((name) => set.has(name));
  if (isModern) return { kind: "combination", name: "modern-stack" };
  return { kind: "unmeasured", names: selected };
}
