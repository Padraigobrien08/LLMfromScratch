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
    <div className="figure-panel">
      <div className="fig-row fig-row-wide">
        <label className="field field-inline">
          after the token
          <select
            className="input input-sm mono"
            style={{ fontSize: 14 }}
            value={contextIndex}
            onChange={(e) => {
              setContextIndex(Number(e.target.value));
              setDraws([]);
            }}
          >
            {entries.map((e, i) => (
              <option key={e.context} value={i}>
                {JSON.stringify(e.context)} · {e.occurrences}×
              </option>
            ))}
          </select>
        </label>
        <button className="btn btn-primary" onClick={draw}>
          Draw a token
        </button>
        <button className="btn btn-ghost" onClick={() => setDraws([])}>
          Clear
        </button>
      </div>

      {/* Numbered, and in the order the real sampler applies them.
          The reader meets three sliders at once and has no way to know that two of them
          are a different kind of thing: temperature reshapes the whole distribution, where
          top-k and top-p delete from it. Both cutoffs start off, so the first thing anyone
          moves acts on temperature alone — the numbering says that is deliberate rather
          than leaving them to discover it. Nothing is hidden: a control the reader cannot
          see is a control they cannot learn from. */}
      <p className="sampler-stage">
        <span className="sampler-stage-num">1</span> Reshape the distribution
      </p>
      <div className="fig-grid fig-grid-narrow" style={{ marginBottom: "var(--space-3)" }}>
        <label className="field">
          temperature
          <input type="range" min={0} max={2} step={0.05} value={temperature}
            onChange={(e) => setTemperature(Number(e.target.value))} />
          <b style={{ minWidth: 38 }}>{temperature.toFixed(2)}</b>
        </label>
      </div>

      <p className="sampler-stage">
        <span className="sampler-stage-num">2</span> Then cut it down — both off until you move them
      </p>
      <div className="fig-grid fig-grid-narrow" style={{ marginBottom: "var(--space-2)" }}>
        <label className="field">
          top-k
          <input type="range" min={0} max={entry.candidates.length} step={1} value={topK ?? 0}
            onChange={(e) => setTopK(Number(e.target.value) === 0 ? null : Number(e.target.value))} />
          <b style={{ minWidth: 32 }}>{topK ?? "off"}</b>
        </label>
        <label className="field">
          top-p
          <input type="range" min={0} max={1} step={0.01} value={topP ?? 1}
            onChange={(e) => setTopP(Number(e.target.value) === 1 ? null : Number(e.target.value))} />
          <b style={{ minWidth: 32 }}>{topP?.toFixed(2) ?? "off"}</b>
        </label>
      </div>

      <p className="fig-note" style={{ margin: "0 0 var(--space-3)" }}>
        Watch the bars below: temperature moves probability mass between them, the cutoffs grey
        candidates out entirely. {keptCount} of {scored.length} candidates survive.{" "}
        {entry.distinct_followers > entry.candidates.length && (
          <>
            This token had {entry.distinct_followers} distinct followers in the corpus; the{" "}
            {entry.candidates.length} shown cover {(entry.covered * 100).toFixed(0)}% of its
            occurrences.
          </>
        )}
      </p>

      {/* The one control on the page whose entire output is a side effect: pressing "draw"
          appends a word here and says nothing. Announcing the text the button produced is
          what the button is for — without it the figure is inoperable by anyone not
          watching this line. `polite` so a reader drawing several does not get interrupted
          mid-announcement. */}
      <p className="draws" role="status" aria-live="polite">
        <span style={{ color: "var(--color-neutral-700)" }}>{entry.context}</span>
        {draws.join("")}
      </p>

      <div className="candidates">
        {ordered.map((s) => (
          <div key={s.id} className={`candidate ${s.kept ? "" : "candidate-dropped"}`}>
            <code>{JSON.stringify(s.text)}</code>
            <div className="candidate-track">
              <div className="candidate-bar" style={{ width: `${(s.prob / maxProb) * 100}%` }} />
            </div>
            <span className="candidate-pct">{(s.prob * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}
