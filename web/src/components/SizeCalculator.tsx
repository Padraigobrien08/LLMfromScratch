import { useState } from "react";

import PlateNumeral from "./PlateNumeral";
import {
  GPT2_124M,
  LLAMA_124M,
  type SizeConfig,
  flopsPerToken,
  formatBytes,
  formatCount,
  headDim,
  kvCacheBytes,
  mlpHidden,
  parameters,
} from "../lib/modelsize";

const PRESETS: Array<[string, SizeConfig]> = [
  ["GPT-2 124M", GPT2_124M],
  ["Llama-style 124M", LLAMA_124M],
  ["Ablation baseline", { ...GPT2_124M, nLayer: 8, nHead: 8, nKvHead: 8, nEmbd: 512, blockSize: 512 }],
];

/** Where the budget goes, in the order the model spends it. */
const PARTS = [
  ["tokenEmbedding", "Token embedding", "var(--color-accent)"],
  ["positionEmbedding", "Position embedding", "var(--color-accent-2)"],
  ["attention", "Attention", "var(--color-accent-700)"],
  ["feedForward", "Feed-forward", "var(--color-process-yellow)"],
  ["norms", "Norms", "var(--color-neutral-500)"],
  ["lmHead", "Output head", "var(--color-accent-2-700)"],
] as const;

/** n_embd must divide by n_head, and n_head by n_kv_head — same rules as ModelConfig. */
const divisorsOf = (n: number, max: number) =>
  Array.from({ length: max }, (_, i) => i + 1).filter((d) => n % d === 0);

export default function SizeCalculator() {
  const [cfg, setCfg] = useState<SizeConfig>(GPT2_124M);
  const [seqLen, setSeqLen] = useState(1024);
  const [batch, setBatch] = useState(1);

  const set = <K extends keyof SizeConfig>(key: K, value: SizeConfig[K]) =>
    setCfg((c) => {
      const next = { ...c, [key]: value };
      // Keep the shape legal rather than letting the reader build a model the real
      // ModelConfig would reject in __post_init__.
      if (next.nEmbd % next.nHead !== 0) next.nHead = divisorsOf(next.nEmbd, 32).at(-1) ?? 1;
      if (next.nHead % next.nKvHead !== 0) next.nKvHead = next.nHead;
      if (next.nKvHead > next.nHead) next.nKvHead = next.nHead;
      return next;
    });

  const p = parameters(cfg);
  const flops = flopsPerToken(cfg, seqLen);
  const cache = kvCacheBytes(cfg, seqLen, batch);

  const costs: Array<[string, string]> = [
    ["Weights in bf16", formatBytes(p.total * 2)],
    ["KV cache", formatBytes(cache)],
    ["FLOPs / token", `${((flops.dense + flops.attention) / 1e9).toFixed(2)}G`],
    [
      "of which attention",
      `${((flops.attention / (flops.dense + flops.attention)) * 100).toFixed(1)}%`,
    ],
  ];

  return (
    <div className="figure-panel">
      <div className="fig-row">
        {PRESETS.map(([name, preset]) => {
          const current = JSON.stringify(preset) === JSON.stringify(cfg);
          return (
            <button
              key={name}
              className={`btn btn-sm ${current ? "btn-primary" : "btn-secondary"}`}
              onClick={() => setCfg(preset)}
              aria-pressed={current}
            >
              {name}
            </button>
          );
        })}
      </div>

      <p className="fig-note" style={{ margin: "var(--space-2) 0 var(--space-3)" }}>
        Drag layers and width and watch the breakdown at the bottom move. Then drag context length
        and watch only the last two rows move — that is attention's cost, and it is why long context
        is expensive.
      </p>

      <div className="fig-grid">
        <label className="field">
          layers
          <input type="range" min={1} max={48} value={cfg.nLayer}
            onChange={(e) => set("nLayer", Number(e.target.value))} />
          <b style={{ minWidth: 28 }}>{cfg.nLayer}</b>
        </label>
        <label className="field">
          width
          <input type="range" min={128} max={4096} step={64} value={cfg.nEmbd}
            onChange={(e) => set("nEmbd", Number(e.target.value))} />
          <b style={{ minWidth: 44 }}>{cfg.nEmbd}</b>
        </label>
      </div>

      <div className="fig-row fig-row-wide" style={{ marginBottom: "var(--space-4)" }}>
        <label className="field field-inline">
          heads
          <select className="input input-sm" value={cfg.nHead}
            onChange={(e) => set("nHead", Number(e.target.value))}>
            {divisorsOf(cfg.nEmbd, 64).map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="field field-inline">
          kv heads
          <select className="input input-sm" value={cfg.nKvHead}
            onChange={(e) => set("nKvHead", Number(e.target.value))}>
            {divisorsOf(cfg.nHead, cfg.nHead).map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="field field-inline">
          mlp
          <select className="input input-sm" value={cfg.mlp}
            onChange={(e) => set("mlp", e.target.value as SizeConfig["mlp"])}>
            <option value="gelu">GELU</option>
            <option value="swiglu">SwiGLU</option>
          </select>
        </label>
        <label className="field field-inline">
          positions
          <select className="input input-sm" value={cfg.posEmb}
            onChange={(e) => set("posEmb", e.target.value as SizeConfig["posEmb"])}>
            <option value="learned">learned</option>
            <option value="rope">RoPE</option>
            <option value="none">none</option>
          </select>
        </label>
        <label className="field field-inline">
          <input type="checkbox" checked={cfg.tieEmbeddings}
            onChange={(e) => set("tieEmbeddings", e.target.checked)} />
          tie embeddings
        </label>
        <label className="field field-inline">
          <input type="checkbox" checked={cfg.bias}
            onChange={(e) => set("bias", e.target.checked)} />
          bias terms
        </label>
        <label className="field field-inline">
          <input type="checkbox" checked={cfg.norm === "rmsnorm"}
            onChange={(e) => set("norm", e.target.checked ? "rmsnorm" : "layernorm")} />
          RMSNorm
        </label>
      </div>

      <p className="eyebrow">Total parameters</p>
      <div className="param-total">
        <PlateNumeral value={formatCount(p.total)} />
      </div>
      <p className="param-detail">
        {p.total.toLocaleString()} · head dim {headDim(cfg)} · feed-forward width {mlpHidden(cfg)}
      </p>

      <div className="param-bar">
        {PARTS.map(([key, label, color]) =>
          p[key] > 0 ? (
            <div key={key} title={`${label}: ${formatCount(p[key])}`}
              style={{ width: `${(p[key] / p.total) * 100}%`, background: color }} />
          ) : null,
        )}
      </div>

      <table className="table" style={{ marginTop: "var(--space-3)" }}>
        <tbody>
          {PARTS.map(([key, label, color]) => (
            <tr key={key}>
              <td style={{ width: 20 }}>
                <span className="param-swatch" style={{ background: color }} />
              </td>
              <td style={{ fontSize: 16 }}>{label}</td>
              <td className="mono num">{formatCount(p[key])}</td>
              <td className="mono num" style={{ color: "var(--color-neutral-700)" }}>
                {((p[key] / p.total) * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="fig-grid" style={{ marginTop: "var(--space-4)", marginBottom: 0 }}>
        <label className="field">
          context
          <input type="range" min={128} max={32768} step={128} value={seqLen}
            onChange={(e) => setSeqLen(Number(e.target.value))} />
          <b style={{ minWidth: 50 }}>{seqLen}</b>
        </label>
        <label className="field">
          batch
          <input type="range" min={1} max={64} value={batch}
            onChange={(e) => setBatch(Number(e.target.value))} />
          <b style={{ minWidth: 28 }}>{batch}</b>
        </label>
      </div>

      <div className="fig-stats" style={{ marginTop: "var(--space-4)" }}>
        {costs.map(([label, value]) => (
          <div key={label}>
            <p className="eyebrow">{label}</p>
            <div className="readout readout-sm">{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
