import { useState } from "react";

/**
 * Why a token needs to look at other tokens.
 *
 * Deliberately *not* model output — these are linguistic dependencies chosen to make the
 * problem obvious, and the page says so. Showing invented attention weights and calling
 * them the model's would be the exact dishonesty the rest of the site avoids; what the
 * model actually attended to is one link away, in the real explorer.
 *
 * The first sentence carries a swap, and it is the point of the whole chapter. "The
 * trophy didn't fit in the suitcase because it was too big" — change `big` to `small` and
 * "it" stops meaning the trophy and starts meaning the suitcase. Nothing about the word
 * "it" changed. That is the entire argument for context in one keystroke, and it is worth
 * an interaction rather than a sentence claiming it happens.
 */
type Link = { on: number[]; why: string };

type Sentence = {
  label: string;
  words: string[];
  /** word index → the earlier words that resolve it, plus why. */
  links: Record<number, Link>;
  /** Which link to open with. Object key order would otherwise pick the lowest index. */
  initial: number;
  /**
   * One word the reader can exchange, and the links that change when they do. The word
   * itself is substituted into `words`, so the sentence reads normally either way.
   */
  swap?: {
    at: number;
    options: [string, string];
    links: Record<string, Record<number, Link>>;
  };
};

const SENTENCES: Sentence[] = [
  {
    label: "Ambiguous “it”",
    words: ["The", "trophy", "didn't", "fit", "in", "the", "suitcase", "because", "it", "was", "too", "big"],
    swap: {
      at: 11,
      options: ["big", "small"],
      links: {
        big: { 8: { on: [1], why: "“it” is the trophy — the trophy is the thing that is too big." } },
        small: {
          8: {
            on: [6],
            why: "One word changed at the end of the sentence, and “it” is now the suitcase. The word “it” is identical in both.",
          },
        },
      },
    },
    links: {
      11: { on: [1, 6], why: "the adjective only means anything as a comparison between the two objects." },
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
  const [option, setOption] = useState(sentence.swap?.options[0] ?? "");
  const [selected, setSelected] = useState<number | null>(sentence.initial);

  const swapped = sentence.swap?.links[option] ?? {};
  const links: Record<number, Link> = { ...sentence.links, ...swapped };
  const targets = Object.keys(links).map(Number);
  const words = sentence.swap
    ? sentence.words.map((w, i) => (i === sentence.swap!.at ? option : w))
    : sentence.words;

  const link = selected !== null ? links[selected] : undefined;
  const highlighted = new Set(link?.on ?? []);

  const choose = (i: number) => {
    setWhich(i);
    setOption(SENTENCES[i]!.swap?.options[0] ?? "");
    setSelected(SENTENCES[i]!.initial);
  };

  return (
    <div className="intuition">
      <div className="fig-row" style={{ marginBottom: "var(--space-4)" }}>
        <div className="seg seg-paper" role="radiogroup" aria-label="Sentence">
          {SENTENCES.map((s, i) => (
            <label key={s.label} className="seg-opt">
              <input
                type="radio"
                name="intuition-sentence"
                value={s.label}
                checked={which === i}
                onChange={() => choose(i)}
              />
              {s.label}
            </label>
          ))}
        </div>
      </div>

      <p className="fig-note" style={{ margin: "0 0 var(--space-3)" }}>
        The underlined words cannot be understood on their own. Click one to see what resolves it.
      </p>

      <p className="sentence">
        {words.map((w, i) => {
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
              {isTarget ? (
                <button type="button" className={classes} onClick={() => setSelected(i)}>
                  {w}
                </button>
              ) : (
                <span className={classes}>{w}</span>
              )}
              {i < words.length - 1 && " "}
            </span>
          );
        })}
      </p>

      {link && <p className="sentence-why">{link.why}</p>}

      {/* The swap. A radio pair rather than two buttons, because it is one choice out of
          two and the reader should see both words at once — the whole demonstration is
          that the sentence differs by exactly this. */}
      {sentence.swap && (
        <div className="sentence-swap">
          <span className="sentence-swap-lead">Change the last word</span>
          <div className="seg seg-paper" role="radiogroup" aria-label="Final adjective">
            {sentence.swap.options.map((o) => (
              <label key={o} className="seg-opt">
                <input
                  type="radio"
                  name="intuition-swap"
                  value={o}
                  checked={option === o}
                  onChange={() => {
                    setOption(o);
                    setSelected(sentence.initial);
                  }}
                />
                too {o}
              </label>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
