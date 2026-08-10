import { useMemo, useState } from "react";

import bigram from "../data/bigram.json";
import { scoreCandidates, sample } from "../lib/sampling";

const entries = bigram.entries;

export default function SamplingDemo() {
  const [contextIndex, setContextIndex] = useState(0);
  const [temperature, setTemperature] = useState(1);
  const [topK, setTopK] = useState<number | null>(null);
  const [topP, setTopP] = useState<number | null>(null);
  const [draws, setDraws] = useState<string[]>([]);
  const [seed, setSeed] = useState(1);

  const entry = entries[contextIndex]!;
  const scored = useMemo(
    () => scoreCandidates(entry.candidates, { temperature, topK, topP }),
    [entry, temperature, topK, topP],
  );

  const ordered = [...scored].sort((a, b) => b.raw - a.raw);
  const maxProb = Math.max(...ordered.map((s) => s.prob), 0.001);
  const keptCount = scored.filter((s) => s.kept).length;

  const draw = () => {
    // Seeded so the same settings give the same sequence — a demo that cannot be
    // reproduced is a demo you cannot reason about.
    let s = (seed * 1103515245 + 12345) & 0x7fffffff;
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    setSeed(s);
    const picked = sample(scored, (s % 100000) / 100000);
    if (picked) setDraws((d) => [...d.slice(-11), picked.text]);
  };

  return (
    <div className="card">
      <div className="controls" style={{ marginBottom: 14 }}>
        <label className="field">
          after the token
          <select
            value={contextIndex}
            onChange={(e) => {
              setContextIndex(Number(e.target.value));
              setDraws([]);
            }}
            style={{ fontFamily: "var(--mono)" }}
          >
            {entries.map((e, i) => (
              <option key={e.context} value={i}>
                {JSON.stringify(e.context)} · {e.occurrences}×
              </option>
            ))}
          </select>
        </label>
        <button onClick={draw} className="primary">
          Draw a token
        </button>
        {draws.length > 0 && <button onClick={() => setDraws([])}>Clear</button>}
      </div>

      <div className="controls" style={{ marginBottom: 6 }}>
        <label className="field" style={{ flex: "1 1 210px" }}>
          temperature
          <input
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <b style={{ fontFamily: "var(--mono)", minWidth: 34 }}>{temperature.toFixed(2)}</b>
        </label>
        <label className="field" style={{ flex: "1 1 190px" }}>
          top-k
          <input
            type="range"
            min={0}
            max={entry.candidates.length}
            step={1}
            value={topK ?? 0}
            onChange={(e) => setTopK(Number(e.target.value) === 0 ? null : Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <b style={{ fontFamily: "var(--mono)", minWidth: 30 }}>{topK ?? "off"}</b>
        </label>
        <label className="field" style={{ flex: "1 1 190px" }}>
          top-p
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={topP ?? 1}
            onChange={(e) => setTopP(Number(e.target.value) === 1 ? null : Number(e.target.value))}
            style={{ flex: 1 }}
          />
          <b style={{ fontFamily: "var(--mono)", minWidth: 30 }}>{topP?.toFixed(2) ?? "off"}</b>
        </label>
      </div>

      <p className="small muted" style={{ margin: "0 0 12px" }}>
        {keptCount} of {scored.length} candidates survive the cutoffs.{" "}
        {entry.distinct_followers > entry.candidates.length && (
          <>
            This token had <b>{entry.distinct_followers}</b> distinct followers in the corpus; the{" "}
            {entry.candidates.length} shown cover {(entry.covered * 100).toFixed(0)}% of its
            occurrences.
          </>
        )}
      </p>

      {draws.length > 0 && (
        <p
          style={{
            font: "15px var(--mono)",
            background: "var(--panel-alt)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: "10px 12px",
            margin: "0 0 14px",
            whiteSpace: "pre-wrap",
          }}
        >
          <span className="muted">{entry.context}</span>
          {draws.join("")}
        </p>
      )}

      <div style={{ display: "grid", gap: 3 }}>
        {ordered.map((s) => (
          <div
            key={s.id}
            style={{
              display: "grid",
              gridTemplateColumns: "116px 1fr 62px",
              alignItems: "center",
              gap: 10,
              opacity: s.kept ? 1 : 0.32,
            }}
          >
            <code
              style={{
                fontSize: 12,
                background: "none",
                border: "none",
                padding: 0,
                whiteSpace: "pre",
                overflow: "hidden",
                textOverflow: "ellipsis",
              }}
            >
              {JSON.stringify(s.text)}
            </code>
            <div style={{ background: "var(--grid-line)", borderRadius: 3, height: 15 }}>
              <div
                style={{
                  width: `${(s.prob / maxProb) * 100}%`,
                  height: "100%",
                  background: s.kept ? "var(--accent)" : "var(--muted)",
                  borderRadius: 3,
                  transition: "width .12s",
                }}
              />
            </div>
            <span
              className="small"
              style={{ fontFamily: "var(--mono)", textAlign: "right", color: "var(--muted)" }}
            >
              {(s.prob * 100).toFixed(1)}%
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
