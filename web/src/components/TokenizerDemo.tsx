import { useEffect, useMemo, useState } from "react";

import { type Token, type Tokenizer, loadTokenizer } from "../lib/tokenizer";

const EXAMPLES: Array<[string, string]> = [
  ["A sentence", "Dorothy lived in the midst of the great Kansas prairies."],
  ["Two characters", "The Scarecrow wanted a brain, the Tin Woodman a heart."],
  ["A long word", "unbelievable antidisestablishmentarianism"],
  ["Numbers", "3.14159 costs $1,234,567.89 — 100% guaranteed!"],
  ["Code", "def train(model, batch): return model(batch).loss"],
];

/** Six tints, cycled — distinct enough to see the boundaries, quiet enough to read. */
const TINTS = 6;

export default function TokenizerDemo() {
  const [tokenizer, setTokenizer] = useState<Tokenizer | null>(null);
  const [failed, setFailed] = useState(false);
  const [text, setText] = useState(EXAMPLES[0]![1]);
  const [showIds, setShowIds] = useState(false);

  useEffect(() => {
    loadTokenizer().then(setTokenizer, () => setFailed(true));
  }, []);

  const tokens: Token[] = useMemo(
    () => (tokenizer ? tokenizer.tokenize(text) : []),
    [tokenizer, text],
  );

  const chars = [...text].length;

  const stats: Array<[string, string]> = [
    ["Characters", String(chars)],
    ["Tokens", String(tokens.length)],
    ["Chars per token", tokens.length ? (chars / tokens.length).toFixed(2) : "—"],
  ];

  return (
    <div className="figure-panel">
      <div className="fig-row">
        {EXAMPLES.map(([label, example]) => (
          <button
            key={label}
            className="btn btn-secondary btn-sm"
            style={{ whiteSpace: "nowrap" }}
            onClick={() => setText(example)}
          >
            {label}
          </button>
        ))}
      </div>

      <textarea
        className="input token-input"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={2}
        spellCheck={false}
        aria-label="Text to tokenize"
      />

      {/* What to manipulate, what to watch, and what it shows — the three things a figure
          has to answer without a caption explaining it. */}
      <p className="fig-note" style={{ margin: "var(--space-2) 0" }}>
        Type anything above and watch the vocabulary cut it up. Each shaded run is one token; the
        count below is what the model actually receives.
      </p>

      <div className="token-strip">
        {tokens.map((t, i) => (
          <span key={i} className={`token token-${i % TINTS}`} title={`id ${t.id}`}>
            {showIds ? t.id : t.text === "\n" ? "⏎\n" : t.text}
          </span>
        ))}
      </div>

      {(!tokenizer || tokens.length === 0) && (
        <p className="fig-note">
          {failed
            ? "The vocabulary failed to load, so this figure is unavailable."
            : !tokenizer
              ? "Loading the 50,257-token vocabulary…"
              : "No tokens — the input is empty."}
        </p>
      )}

      <div className="fig-stats" style={{ marginTop: "var(--space-4)" }}>
        {stats.map(([label, value]) => (
          <div key={label}>
            <p className="eyebrow">{label}</p>
            <div className="readout">{value}</div>
          </div>
        ))}
        <label className="field field-inline">
          <input type="checkbox" checked={showIds} onChange={(e) => setShowIds(e.target.checked)} />
          show ids
        </label>
      </div>
    </div>
  );
}
