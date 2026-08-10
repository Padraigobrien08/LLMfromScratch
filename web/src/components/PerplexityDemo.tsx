import { useState } from "react";

import { perplexity, uniformLoss } from "../lib/sampling";

const VOCAB = 50257;
const CHANCE = uniformLoss(VOCAB);

/** Every number here is either arithmetic or documented in the repository. */
const MARKS: Array<{ loss: number; label: string; note: string }> = [
  { loss: CHANCE, label: "Guessing", note: "ln(50,257) — a model that has learned nothing" },
  { loss: 6.0, label: "Early training", note: "word frequencies learned, not much else" },
  { loss: 3.29, label: "GPT-2 124M", note: "this repository's pre-registered target on FineWeb-Edu" },
  { loss: 2.6, label: "Bigger models", note: "roughly where a few-billion-parameter model lands" },
];

export default function PerplexityDemo() {
  const [loss, setLoss] = useState(3.29);
  const ppl = perplexity(loss);

  return (
    <div className="card">
      <label className="field" style={{ width: "100%", marginBottom: 16 }}>
        validation loss
        <input
          type="range"
          min={1}
          max={11}
          step={0.01}
          value={loss}
          onChange={(e) => setLoss(Number(e.target.value))}
          style={{ flex: 1 }}
        />
        <b style={{ fontFamily: "var(--mono)", minWidth: 44 }}>{loss.toFixed(2)}</b>
      </label>

      <p className="eyebrow">What that means</p>
      <div className="readout">
        {ppl < 10000 ? ppl.toFixed(ppl < 100 ? 1 : 0) : ppl.toExponential(1)}
      </div>
      <p style={{ margin: "6px 0 0", maxWidth: "60ch" }}>
        The model is as uncertain about the next token as if it were choosing uniformly between{" "}
        <b>{ppl < 10000 ? Math.round(ppl).toLocaleString() : ppl.toExponential(1)}</b> equally likely
        options — out of a vocabulary of {VOCAB.toLocaleString()}.
      </p>

      <div style={{ display: "grid", gap: 3, marginTop: 18 }}>
        {MARKS.map((m) => (
          <button
            key={m.label}
            onClick={() => setLoss(Number(m.loss.toFixed(2)))}
            style={{
              display: "grid",
              gridTemplateColumns: "130px 60px 1fr",
              gap: 10,
              alignItems: "baseline",
              textAlign: "left",
              border: "none",
              background: Math.abs(loss - m.loss) < 0.02 ? "var(--panel-alt)" : "transparent",
              padding: "7px 9px",
              borderRadius: 7,
            }}
          >
            <span style={{ fontWeight: 550, fontSize: 14 }}>{m.label}</span>
            <span style={{ fontFamily: "var(--mono)", fontSize: 13, color: "var(--muted)" }}>
              {m.loss.toFixed(2)}
            </span>
            <span className="small muted">{m.note}</span>
          </button>
        ))}
      </div>

      <p className="small muted" style={{ margin: "14px 0 0" }}>
        Note how compressed the useful range is. The whole distance between "knows nothing"
        and "reproduces GPT-2" is about 7.5 in loss — and the last tenth of that is harder to
        win than the first five. It is also why an ablation arguing over 0.02 needs the
        paired-seed machinery: at this scale, 0.02 is a real effect that raw run-to-run noise
        would bury.
      </p>
    </div>
  );
}
