import { useEffect, useMemo, useState } from "react";

import { type Token, type Tokenizer, loadTokenizer } from "../lib/tokenizer";

const EXAMPLES = [
  "Dorothy lived in the midst of the great Kansas prairies.",
  "The Scarecrow wanted a brain, the Tin Woodman a heart.",
  "unbelievable antidisestablishmentarianism",
  "3.14159 costs $1,234,567.89 — 100% guaranteed!",
  "def train(model, batch): return model(batch).loss",
];

/** Distinct enough to see the boundaries, quiet enough to still read as a sentence. */
const HUES = [210, 340, 150, 40, 270, 190];

export default function TokenizerDemo() {
  const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);
  const [failed, setFailed] = useState(false);
  const [text, setText] = useState(EXAMPLES[0]!);
  const [showIds, setShowIds] = useState(false);

  useEffect(() => {
    loadTokenizer().then(setTokenizer, () => setFailed(true));
  }, []);

  const tokens: Token[] = useMemo(
    () => (tokenizer ? tokenizer.tokenize(text) : []),
    [tokenizer, text],
  );

  if (failed) {
    return (
      <div className="card">
        <p className="small" style={{ margin: 0 }}>
          The vocabulary failed to load, so this demo is unavailable.
        </p>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="controls" style={{ marginBottom: 10 }}>
        {EXAMPLES.map((e, i) => (
          <button key={e} onClick={() => setText(e)} style={{ fontSize: 13, padding: "5px 10px" }}>
            {["A sentence", "Two characters", "A long word", "Numbers", "Code"][i]}
          </button>
        ))}
      </div>

      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={2}
        spellCheck={false}
        aria-label="Text to tokenize"
        style={{
          width: "100%",
          font: "14px var(--mono)",
          padding: "10px 12px",
          color: "var(--text)",
          background: "var(--bg)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          resize: "vertical",
        }}
      />

      {!tokenizer ? (
        <p className="small muted" style={{ margin: "12px 0 0" }}>
          Loading the 50,257-token vocabulary…
        </p>
      ) : (
        <>
          <div className="strip" style={{ marginTop: 14 }}>
            {tokens.map((t, i) => (
              <span
                key={i}
                title={`id ${t.id}`}
                style={{
                  font: "13px var(--mono)",
                  padding: "3px 1px",
                  borderRadius: 4,
                  background: `hsl(${HUES[i % HUES.length]} 70% 50% / 0.18)`,
                  boxShadow: `inset 0 -2px 0 hsl(${HUES[i % HUES.length]} 70% 50% / 0.5)`,
                  whiteSpace: "pre-wrap",
                }}
              >
                {showIds ? t.id : t.text === "\n" ? "⏎\n" : t.text}
              </span>
            ))}
            {tokens.length === 0 && <span className="small muted">No tokens — the input is empty.</span>}
          </div>

          <div className="statrow" style={{ marginTop: 16 }}>
            <div>
              <p className="eyebrow">Characters</p>
              <div className="readout sm">{[...text].length}</div>
            </div>
            <div>
              <p className="eyebrow">Tokens</p>
              <div className="readout sm">{tokens.length}</div>
            </div>
            <div>
              <p className="eyebrow">Chars per token</p>
              <div className="readout sm">
                {tokens.length ? ([...text].length / tokens.length).toFixed(2) : "—"}
              </div>
            </div>
            <label className="field" style={{ alignSelf: "end" }}>
              <input
                type="checkbox"
                checked={showIds}
                onChange={(e) => setShowIds(e.target.checked)}
              />
              show ids
            </label>
          </div>
        </>
      )}
    </div>
  );
}
