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
  label: string;
  words: string[];
  /** word index → the earlier words that resolve it, plus why. */
  links: Record<number, { on: number[]; why: string }>;
  /** Which link to open with. Object key order would otherwise pick the lowest index. */
  initial: number;
};

const SENTENCES: Sentence[] = [
  {
    label: "Ambiguous “it”",
    words: ["The", "trophy", "didn't", "fit", "in", "the", "suitcase", "because", "it", "was", "too", "big"],
    links: {
      8: { on: [1], why: "“it” is the trophy — swap “big” for “small” and it becomes the suitcase." },
      11: { on: [1, 6], why: "“big” only makes sense as a comparison between the two objects." },
      3: { on: [1, 6], why: "what fits in what — the verb needs both nouns." },
    },
    initial: 8,
  },
  {
    label: "Pronouns",
    words: ["Dorothy", "lived", "in", "Kansas", ",", "and", "she", "missed", "her", "home"],
    links: {
      6: { on: [0], why: "“she” refers back to Dorothy, four tokens earlier." },
      8: { on: [0], why: "“her” too — pronouns are the clearest case for looking back." },
      9: { on: [3], why: "“home” means Kansas here, and only because Kansas was mentioned." },
    },
    initial: 6,
  },
  {
    label: "Agreement",
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
    <div className="figure-panel">
      <div className="fig-row" style={{ marginBottom: "var(--space-4)" }}>
        {SENTENCES.map((s, i) => (
          <button
            key={s.label}
            className={`btn btn-sm ${which === i ? "btn-primary" : "btn-secondary"}`}
            onClick={() => {
              setWhich(i);
              setSelected(SENTENCES[i]!.initial);
            }}
            aria-pressed={which === i}
          >
            {s.label}
          </button>
        ))}
      </div>

      <p className="fig-note" style={{ margin: "0 0 var(--space-3)" }}>
        The underlined words cannot be understood on their own. Click one.
      </p>

      <p className="sentence">
        {sentence.words.map((w, i) => {
          const isTarget = targets.includes(i);
          const isSelected = selected === i;
          const isSource = highlighted.has(i);
          const classes = [
            "sentence-word",
            isTarget ? "sentence-word-target" : "",
            isSelected ? "sentence-word-selected" : isSource ? "sentence-word-source" : "",
          ]
            .filter(Boolean)
            .join(" ");
          return (
            <span key={i}>
              <span
                className={classes}
                onClick={() => isTarget && setSelected(i)}
                role={isTarget ? "button" : undefined}
                tabIndex={isTarget ? 0 : undefined}
                onKeyDown={(e) => {
                  if (isTarget && (e.key === "Enter" || e.key === " ")) {
                    e.preventDefault();
                    setSelected(i);
                  }
                }}
              >
                {w}
              </span>
              {i < sentence.words.length - 1 && " "}
            </span>
          );
        })}
      </p>

      {link && <p className="sentence-why">{link.why}</p>}
    </div>
  );
}
