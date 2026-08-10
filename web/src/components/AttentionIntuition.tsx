import { useState } from "react";

/**
 * Why a token needs to look at other tokens.
 *
 * Deliberately *not* model output — these are linguistic dependencies chosen to make
 * the problem obvious, and the page says so. Showing invented attention weights and
 * calling them the model's would be the exact dishonesty the rest of the site avoids;
 * what the model actually attended to is one link away, in the real explorer.
 */
type Sentence = {
  words: string[];
  /** word index → the earlier words that resolve it, plus why. */
  links: Record<number, { on: number[]; why: string }>;
  /** Which link to open with. Object key order would otherwise pick the lowest index. */
  initial: number;
};

const SENTENCES: Sentence[] = [
  {
    words: ["The", "trophy", "didn't", "fit", "in", "the", "suitcase", "because", "it", "was", "too", "big"],
    links: {
      8: { on: [1], why: "“it” is the trophy — swap “big” for “small” and it becomes the suitcase." },
      11: { on: [1, 6], why: "“big” only makes sense as a comparison between the two objects." },
      3: { on: [1, 6], why: "what fits in what — the verb needs both nouns." },
    },
    initial: 8,
  },
  {
    words: ["Dorothy", "lived", "in", "Kansas", ",", "and", "she", "missed", "her", "home"],
    links: {
      6: { on: [0], why: "“she” refers back to Dorothy, four tokens earlier." },
      8: { on: [0], why: "“her” too — pronouns are the clearest case for looking back." },
      9: { on: [3], why: "“home” means Kansas here, and only because Kansas was mentioned." },
    },
    initial: 6,
  },
  {
    words: ["The", "key", "to", "the", "cabinets", "was", "rusty", "and", "would", "not", "turn"],
    links: {
      5: { on: [1], why: "“was”, not “were” — the verb agrees with “key”, not with “cabinets”." },
      10: { on: [1], why: "the thing that turns is the key, despite “cabinets” sitting closer." },
    },
    initial: 5,
  },
];

export default function AttentionIntuition() {
  const [which, setWhich] = useState(0);
  const sentence = SENTENCES[which]!;
  const targets = Object.keys(sentence.links).map(Number);
  const [selected, setSelected] = useState<number | null>(sentence.initial);

  const link = selected !== null ? sentence.links[selected] : undefined;
  const highlighted = new Set(link?.on ?? []);

  return (
    <div className="card">
      <div className="controls" style={{ marginBottom: 14 }}>
        {SENTENCES.map((_, i) => (
          <button
            key={i}
            onClick={() => {
              setWhich(i);
              setSelected(SENTENCES[i]!.initial);
            }}
            aria-pressed={which === i}
            style={{
              fontSize: 13,
              background: which === i ? "var(--accent)" : "var(--bg)",
              borderColor: which === i ? "var(--accent)" : "var(--border)",
              color: which === i ? "#fff" : "var(--text)",
            }}
          >
            {["Ambiguous “it”", "Pronouns", "Agreement"][i]}
          </button>
        ))}
      </div>

      <p className="small muted" style={{ margin: "0 0 10px" }}>
        The underlined words cannot be understood on their own. Click one.
      </p>

      <p style={{ fontSize: 20, lineHeight: 2, margin: "0 0 14px", maxWidth: "none" }}>
        {sentence.words.map((w, i) => {
          const isTarget = targets.includes(i);
          const isSelected = selected === i;
          const isSource = highlighted.has(i);
          return (
            <span key={i}>
              <span
                onClick={() => isTarget && setSelected(i)}
                role={isTarget ? "button" : undefined}
                tabIndex={isTarget ? 0 : undefined}
                onKeyDown={(e) => {
                  if (isTarget && (e.key === "Enter" || e.key === " ")) {
                    e.preventDefault();
                    setSelected(i);
                  }
                }}
                style={{
                  padding: "2px 5px",
                  borderRadius: 5,
                  cursor: isTarget ? "pointer" : "default",
                  textDecoration: isTarget ? "underline dotted" : "none",
                  textUnderlineOffset: 5,
                  background: isSelected
                    ? "var(--accent)"
                    : isSource
                      ? "color-mix(in srgb, var(--accent) 22%, transparent)"
                      : "transparent",
                  color: isSelected ? "#fff" : "inherit",
                  fontWeight: isSelected || isSource ? 600 : 400,
                  transition: "background .12s",
                }}
              >
                {w}
              </span>
              {i < sentence.words.length - 1 && " "}
            </span>
          );
        })}
      </p>

      {link && (
        <div className="callout" style={{ margin: 0 }}>
          <p className="small" style={{ margin: 0 }}>
            {link.why}
          </p>
        </div>
      )}

      <p className="small muted" style={{ margin: "14px 0 0" }}>
        A model that reads strictly left to right with no memory cannot do this — by the time
        it reaches “it”, the trophy is gone. Attention is the mechanism that lets every token
        pull information from any earlier token, and learn <i>which</i> ones are worth pulling
        from. These particular links are linguistic illustrations, not model output; what this
        repository's model actually attended to is in the attention explorer.
      </p>
    </div>
  );
}
