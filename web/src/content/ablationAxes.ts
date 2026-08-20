/**
 * The ablation arms' axis labels, from `llmfs.ablation.report.AXIS`, and the
 * modern stack's composition, computed from `configs/ablations/*.yaml`.
 *
 * The site used to hold byte-identical copies of both, kept identical by a
 * comment. A label edited in the report, or a sixth arm added to the stack,
 * now reaches the page by regeneration instead of by someone remembering.
 *
 * Do not hand-edit — regenerate with `llmfs-export-web`. `tests/test_web_export.py`
 * asserts this file is still what the generator emits, so a stale copy fails CI
 * rather than shipping.
 */
export const ABLATION_AXES = {
  "axis": {
    "baseline": "\u2014",
    "norm-rmsnorm": "LayerNorm \u2192 RMSNorm",
    "pos-rope": "learned positions \u2192 RoPE",
    "pos-none": "learned positions \u2192 none",
    "mlp-swiglu": "GELU \u2192 SwiGLU (param-matched)",
    "untied-embeddings": "tied \u2192 untied embeddings",
    "no-bias": "bias \u2192 no bias",
    "gqa-2": "8 KV heads \u2192 2 (GQA)",
    "sched-wsd": "cosine \u2192 WSD schedule",
    "wd-zero": "weight decay 0.1 \u2192 0",
    "lr-3e-4": "lr 1e-3 \u2192 3e-4",
    "lr-3e-3": "lr 1e-3 \u2192 3e-3",
    "modern-stack": "all modern components"
  },
  "modernStack": [
    "gqa-2",
    "mlp-swiglu",
    "no-bias",
    "norm-rmsnorm",
    "pos-rope"
  ]
} as const;
