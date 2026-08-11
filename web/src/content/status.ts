import { MEASURED } from "./measured";

/**
 * The honest-status table, mirroring README.md's.
 *
 * A deliberate constraint: the site must never claim more than the README. The
 * README states each row as a bolded verdict plus its detail — "**Done** — 3.0503
 * val loss" — which splits here into the tag and the state, in the README's own
 * words. When a row moves there, it moves here, and nowhere else.
 *
 * The prose is mirrored by hand; the figures inside it are not. Every number below is
 * interpolated from `measured.ts`, because those were the parts that actually rotted —
 * this table said "223 tests green" and claimed the scaling run was still pending long
 * after it had produced 95.1%. A row can now go stale in its wording, which a reader
 * would notice, but no longer in its arithmetic, which a reader would not.
 *
 * Last checked against README.md on 2026-08-11, after the scaling report landed.
 */
export type StatusRow = {
  pillar: string;
  state: string;
  status: "done" | "pending" | "not started";
};

const EIGHT_GPU = MEASURED.scaling.points.find((p) => p.worldSize === 8)!;

export const STATUS: StatusRow[] = [
  {
    pillar: "Package, config system, data pipeline, trainer, CI",
    state: `${MEASURED.tests.python} tests green, end-to-end verified`,
    status: "done",
  },
  {
    pillar: "Modern architecture (RoPE, RMSNorm, SwiGLU, GQA, KV cache)",
    state: "Hand-implemented, property-tested",
    status: "done",
  },
  {
    pillar: "GPT-2 124M reproduction on FineWeb-Edu",
    state: `${MEASURED.reproduction.loss.toFixed(4)} val loss, docs/reproduction.md`,
    status: "done",
  },
  {
    pillar: `Ablation study (${MEASURED.ablations.arms} arms × ${MEASURED.ablations.seeds} seeds)`,
    state: `docs/ablations.md, ${MEASURED.ablations.runs} runs, ${MEASURED.ablations.gpuHours.toFixed(1)} GPU-h`,
    status: "done",
  },
  {
    pillar: "Efficiency benchmarks (throughput, memory, KV cache)",
    state: "H100 training, 4090 inference; the cache sweep found a 30% bug",
    status: "done",
  },
  {
    pillar: "Quantization + speculative decoding",
    // The artifact records the device as torch names it, lowercase; the README says CUDA.
    state: `docs/efficiency.md, measured on ${MEASURED.quantization.device.toUpperCase()}`,
    status: "done",
  },
  {
    pillar: "Fault-tolerance design doc",
    state: "docs/fault-tolerance.md",
    status: "done",
  },
  {
    pillar: "Multi-GPU scaling report",
    state: `docs/scaling.md, ${(EIGHT_GPU.efficiency * 100).toFixed(1)}% efficiency on 8 GPUs`,
    status: "done",
  },
  {
    pillar: "Interactive attention visualization",
    state: "Live, auto-deployed from CI",
    status: "done",
  },
  {
    pillar: "Interactive site (explainer, RoPE explorer, ablation playground)",
    state: "Live; the playground renders the published sweep",
    status: "done",
  },
];

export const TAG_CLASS: Record<StatusRow["status"], string> = {
  done: "tag tag-accent",
  pending: "tag tag-accent-2",
  "not started": "tag tag-neutral",
};

/**
 * The preview of what the ablation page will say, invented so the layout can be
 * judged before the GPU bill. It renders only while `results/ablations.json` is
 * absent, so it took itself off the page the moment the sweep published — which is
 * the only reason it was ever safe to write.
 */
export const ILLUSTRATIVE: Array<{
  arm: string;
  axis: string;
  delta: string;
  verdict: string;
  better: boolean;
}> = [
  { arm: "norm-rmsnorm", axis: "LayerNorm → RMSNorm", delta: "−0.0121 ± 0.0043", verdict: "better", better: true },
  { arm: "pos-rope", axis: "learned → RoPE", delta: "−0.0286 ± 0.0051", verdict: "better", better: true },
  { arm: "mlp-swiglu", axis: "GELU → SwiGLU", delta: "−0.0074 ± 0.0089", verdict: "within noise", better: false },
  { arm: "gqa-2", axis: "MHA → 2 KV heads", delta: "+0.0035 ± 0.0061", verdict: "within noise", better: false },
  { arm: "modern-stack", axis: "all five at once", delta: "−0.0402 ± 0.0067", verdict: "better", better: true },
];
