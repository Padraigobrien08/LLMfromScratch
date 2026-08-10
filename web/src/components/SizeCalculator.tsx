import { useState } from "react";

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

const PARTS = [
  ["tokenEmbedding", "Token embedding", "var(--accent)"],
  ["positionEmbedding", "Position embedding", "var(--accent-2)"],
  ["attention", "Attention", "var(--good)"],
  ["feedForward", "Feed-forward", "var(--warn)"],
  ["norms", "Norms", "var(--muted)"],
  ["lmHead", "Output head", "#8b5cf6"],
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

  return (
    <div className="card">
      <div className="controls" style={{ marginBottom: 14 }}>
        {PRESETS.map(([name, preset]) => (
          <button
            key={name}
            onClick={() => setCfg(preset)}
            style={{ fontSize: 13 }}
            aria-pressed={JSON.stringify(preset) === JSON.stringify(cfg)}
          >
            {name}
          </button>
        ))}
      </div>

      <div className="grid2" style={{ gap: 12, marginBottom: 16 }}>
        <label className="field">
          layers
          <input type="range" min={1} max={48} value={cfg.nLayer}
            onChange={(e) => set("nLayer", Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontFamily: "var(--mono)", minWidth: 26 }}>{cfg.nLayer}</b>
        </label>
        <label className="field">
          width
          <input type="range" min={128} max={4096} step={64} value={cfg.nEmbd}
            onChange={(e) => set("nEmbd", Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontFamily: "var(--mono)", minWidth: 42 }}>{cfg.nEmbd}</b>
        </label>
      </div>

      <div className="controls" style={{ marginBottom: 16 }}>
        <label className="field">
          heads
          <select value={cfg.nHead} onChange={(e) => set("nHead", Number(e.target.value))}>
            {divisorsOf(cfg.nEmbd, 64).map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="field">
          kv heads
          <select value={cfg.nKvHead} onChange={(e) => set("nKvHead", Number(e.target.value))}>
            {divisorsOf(cfg.nHead, cfg.nHead).map((d) => (
              <option key={d} value={d}>{d}</option>
            ))}
          </select>
        </label>
        <label className="field">
          mlp
          <select value={cfg.mlp} onChange={(e) => set("mlp", e.target.value as SizeConfig["mlp"])}>
            <option value="gelu">GELU</option>
            <option value="swiglu">SwiGLU</option>
          </select>
        </label>
        <label className="field">
          positions
          <select value={cfg.posEmb}
            onChange={(e) => set("posEmb", e.target.value as SizeConfig["posEmb"])}>
            <option value="learned">learned</option>
            <option value="rope">RoPE</option>
            <option value="none">none</option>
          </select>
        </label>
        <label className="field">
          <input type="checkbox" checked={cfg.tieEmbeddings}
            onChange={(e) => set("tieEmbeddings", e.target.checked)} />
          tie embeddings
        </label>
        <label className="field">
          <input type="checkbox" checked={cfg.bias}
            onChange={(e) => set("bias", e.target.checked)} />
          bias terms
        </label>
        <label className="field">
          <input type="checkbox" checked={cfg.norm === "rmsnorm"}
            onChange={(e) => set("norm", e.target.checked ? "rmsnorm" : "layernorm")} />
          RMSNorm
        </label>
      </div>

      <p className="eyebrow">Total parameters</p>
      <div className="readout">{formatCount(p.total)}</div>
      <p className="small muted" style={{ margin: "2px 0 14px" }}>
        {p.total.toLocaleString()} · head dim {headDim(cfg)} · feed-forward width {mlpHidden(cfg)}
      </p>

      <div style={{ display: "flex", height: 26, borderRadius: 6, overflow: "hidden", gap: 1 }}>
        {PARTS.map(([key, label, color]) =>
          p[key] > 0 ? (
            <div key={key} title={`${label}: ${formatCount(p[key])}`}
              style={{ width: `${(p[key] / p.total) * 100}%`, background: color }} />
          ) : null,
        )}
      </div>
      <table style={{ marginTop: 12 }}>
        <tbody>
          {PARTS.map(([key, label, color]) => (
            <tr key={key}>
              <td style={{ width: 18 }}>
                <span style={{ display: "inline-block", width: 10, height: 10, borderRadius: 2,
                  background: color }} />
              </td>
              <td>{label}</td>
              <td className="num">{formatCount(p[key])}</td>
              <td className="num muted">{((p[key] / p.total) * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
      </table>

      <div className="grid2" style={{ marginTop: 18, gap: 12 }}>
        <label className="field">
          context
          <input type="range" min={128} max={32768} step={128} value={seqLen}
            onChange={(e) => setSeqLen(Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontFamily: "var(--mono)", minWidth: 48 }}>{seqLen}</b>
        </label>
        <label className="field">
          batch
          <input type="range" min={1} max={64} value={batch}
            onChange={(e) => setBatch(Number(e.target.value))} style={{ flex: 1 }} />
          <b style={{ fontFamily: "var(--mono)", minWidth: 26 }}>{batch}</b>
        </label>
      </div>

      <div className="statrow" style={{ marginTop: 12 }}>
        <div>
          <p className="eyebrow">Weights in bf16</p>
          <div className="readout sm">{formatBytes(p.total * 2)}</div>
        </div>
        <div>
          <p className="eyebrow">KV cache</p>
          <div className="readout sm">{formatBytes(cache)}</div>
        </div>
        <div>
          <p className="eyebrow">FLOPs / token</p>
          <div className="readout sm">
            {((flops.dense + flops.attention) / 1e9).toFixed(2)}G
          </div>
        </div>
        <div>
          <p className="eyebrow">of which attention</p>
          <div className="readout sm">
            {((flops.attention / (flops.dense + flops.attention)) * 100).toFixed(1)}%
          </div>
        </div>
      </div>
      <p className="small muted" style={{ margin: "10px 0 0" }}>
        Drag the context slider and watch the last two move: attention's cost grows with
        sequence length while everything else stays flat, which is why long context is hard
        and why the KV cache — the thing grouped-query attention shrinks — is what decides
        how many users fit on one GPU.
      </p>
    </div>
  );
}
