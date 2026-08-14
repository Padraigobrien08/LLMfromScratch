import { useState } from "react";

import PlateNumeral from "./PlateNumeral";
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
  const shown = ppl < 10000 ? ppl.toFixed(ppl < 100 ? 1 : 0) : ppl.toExponential(1);

  return (
    <div className="figure-panel">
      <label className="field" style={{ marginBottom: "var(--space-2)" }}>
        validation loss
        <input type="range" min={1} max={11} step={0.01} value={loss}
          onChange={(e) => setLoss(Number(e.target.value))} />
        <b style={{ minWidth: 48 }}>{loss.toFixed(2)}</b>
      </label>

      <p className="fig-note" style={{ margin: "0 0 var(--space-4)" }}>
        Drag the loss and watch the number below it — that is how many equally likely options the
        model is effectively choosing between. The marks are the anchors worth knowing.
      </p>

      <p className="eyebrow">What that means</p>
      <div className="ppl-total">
        <PlateNumeral value={shown} />
      </div>
      <p className="ppl-sentence">
        The model is as uncertain about the next token as if it were choosing uniformly between{" "}
        <b>{ppl < 10000 ? Math.round(ppl).toLocaleString() : ppl.toExponential(1)}</b> equally
        likely options — out of a vocabulary of {VOCAB.toLocaleString()}.
      </p>

      {/* `aria-pressed` because the tint was the only thing saying which anchor the slider
          is sitting on — invisible to a screen reader, and to anyone who cannot separate
          the tint from the paper. The rule the CSS adds in the margin carries the same
          information without relying on colour. */}
      <div className="loss-marks">
        {MARKS.map((m) => {
          const current = Math.abs(loss - m.loss) < 0.02;
          return (
            <button
              key={m.label}
              className={`loss-mark ${current ? "loss-mark-current" : ""}`}
              aria-pressed={current}
              onClick={() => setLoss(Number(m.loss.toFixed(2)))}
            >
              <span className="loss-mark-label">{m.label}</span>
              <span className="loss-mark-loss">{m.loss.toFixed(2)}</span>
              <span className="loss-mark-note">{m.note}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
