import type { ReactNode } from "react";

import AttentionIntuition from "../components/AttentionIntuition";
import Caveat, { Provenance } from "../components/Caveat";
import PerplexityDemo from "../components/PerplexityDemo";
import PlateNumeral from "../components/PlateNumeral";
import SamplingDemo from "../components/SamplingDemo";
import SizeCalculator from "../components/SizeCalculator";
import TokenizerDemo from "../components/TokenizerDemo";
import { chapter } from "./chapters";
import { href } from "../router";

/**
 * The eight chapters' copy.
 *
 * Each body opens on the difficulty behind its page's question, reaches the chapter's
 * own title as the answer, and then shows the thing that settles it. The title is
 * printed by `Answer` rather than written out here, so the sentence a chapter is named
 * for cannot drift from the sentence the page resolves to.
 *
 * Figures are introduced by their label and then left alone: the components carry their
 * own logic, and the only thing a chapter decides is where one sits in the argument.
 */

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";
const attentionExplorer = () => `${import.meta.env.BASE_URL}attention/`;

function FigureLabel({ n, children }: { n: number; children: ReactNode }) {
  return (
    <p className="figure-label">
      Figure {n} · {children}
    </p>
  );
}

/** The chapter's title, printed where its argument arrives at it. */
function Answer({ n }: { n: number }) {
  return <h2 className="chapter-answer">{chapter(n).title}</h2>;
}

/**
 * A question asked of the reader before the page answers it.
 *
 * No input, deliberately. The value is in being wrong for a second, and a text field
 * with a submit button would turn a moment of thought into a form to fill in — which is
 * the point at which a publication starts behaving like courseware. Where the answer is
 * a number the reader can check against the figure below, that is enough.
 */
function Predict({ children }: { children: ReactNode }) {
  return (
    <p className="chapter-predict">
      <span className="chapter-predict-lead">Before you look</span> {children}
    </p>
  );
}

function Chapter1() {
  return (
    <>
      <p className="prose">
        A neural network multiplies numbers together. There is no operation in it that takes a
        letter, or a word, or a sentence — so something has to happen to your text before the model
        can do anything at all with it.
      </p>

      <Answer n={1} />

      <p className="prose">
        Text is cut into <b>tokens</b> — chunks drawn from a fixed vocabulary of 50,257 entries,
        learned by finding the most common character sequences in a large pile of text. Common words
        are one token. Rare ones get split. Each token has an id, and the ids are what reaches the
        model.
      </p>

      <Predict>
        how many tokens is the sentence below? The obvious guess is one per word — try it before you
        read the count.
      </Predict>

      <FigureLabel n={1}>the real GPT-2 vocabulary, in your browser</FigureLabel>
      <TokenizerDemo />

      <p className="prose-secondary">
        Three things worth noticing. A leading space belongs to the token, so <code>&nbsp;the</code>{" "}
        and <code>the</code> are different entries. Long or unusual words shatter into fragments,
        which is why models are worse at rare names. And the ids are the only thing the model ever
        sees — everything downstream operates on those.
      </p>
      <Provenance>
        This runs the same byte-pair merges as the Python tokenizer, pinned to it by a{" "}
        <a href={`${REPO}/tests/test_tokenizer.py`}>fixture asserted from both sides</a> — exact on
        all 14 cases, including emoji, newlines and leading spaces.
      </Provenance>
    </>
  );
}

function Chapter2() {
  return (
    <>
      <p className="prose">
        Token 5,432 is not five thousand of anything. It is a name, and you cannot average two names
        and get a word between them.
      </p>

      <Answer n={2} />

      <p className="prose">
        So the id is used as a row number. The model looks it up in a big table — the{" "}
        <b>embedding matrix</b> — and gets back a vector of a few hundred numbers, which is the
        token's meaning as far as the model is concerned. It is learned, not designed.
      </p>

      <Predict>
        where do you expect most of a 124M-parameter model to live — in the layers that do the
        computing, or somewhere else? Set the sliders to a 124M model and look at the breakdown.
      </Predict>

      <FigureLabel n={2}>parameter, memory and FLOP budget</FigureLabel>
      <SizeCalculator />

      <p className="prose">
        That lookup table is 50,304 × 768 numbers — about <b>31% of the entire model</b>, spent
        before a single layer of actual computation. Drag the context slider and watch the last two
        figures move while everything else stays flat: attention's cost grows with sequence length,
        which is why long context is hard, and why the KV cache — the thing grouped-query attention
        shrinks — is what decides how many users fit on one GPU.
      </p>
      <Provenance>
        Every count is computed from the same shapes the real model builds, and checked against
        parameter counts dumped from{" "}
        <a href={`${REPO}/src/llmfs/model/transformer.py`}>
          the actual <code>Transformer</code>
        </a>{" "}
        for twelve configurations, so the arithmetic cannot quietly drift from the code.
      </Provenance>
    </>
  );
}

function Chapter3() {
  return (
    <>
      <p className="prose">
        Some words carry no meaning of their own. They borrow it from words that came earlier,
        sometimes much earlier.
      </p>

      <Predict>
        in the sentence below, what does <b>it</b> refer to? Then change the last word and watch the
        answer move.
      </Predict>

      <FigureLabel n={3}>an illustration, not model output</FigureLabel>
      <AttentionIntuition />

      <Answer n={3} />

      <p className="prose">
        The word did not change — the sentence around it did, and that was enough to move what it
        points at. So whatever the model holds for <i>it</i> cannot be a property of the word alone,
        and a lookup table can only ever give it that.
      </p>
      <p className="prose-secondary">
        These links are linguistic illustrations, not model output. What this repository's model
        actually attended to is in the attention explorer, one chapter on.
      </p>
    </>
  );
}

function Chapter4() {
  return (
    <>
      <p className="prose">
        And which earlier tokens matter depends entirely on the sentence, so it cannot be wired in
        advance.
      </p>

      <Answer n={4} />

      <p className="prose">
        Every token asks a question and every token advertises what it has. The questions are
        compared against the advertisements, which gives a score for each earlier token — how much
        this one should care about that one — and the token then takes a blend of what the
        high-scoring ones hold, weighted by those scores.
      </p>
      <p className="prose">
        The question is called a <i>query</i>, the advertisement a <i>key</i>, and the thing being
        blended a <i>value</i>. None of the three is hand-written: the model learns what to ask for
        and what to offer, and different <i>heads</i> learn to look for different things — one may
        track which noun a pronoun refers to, another simply the previous word.
      </p>

      <FigureLabel n={4}>every weight from a trained model, per layer and per head</FigureLabel>
      <p className="chapter-handoff">
        <a href={attentionExplorer()}>→ Open the attention explorer</a>
      </p>
      <p className="prose-secondary">
        Click a token and watch which earlier ones it drew from — the real learned weights, in a
        single self-contained HTML file with no build step, no CDN and no backend. Measured, not
        illustrated.
      </p>

      <Predict>
        one thing attention cannot do. Shuffle the tokens and take the same weighted average — does
        the answer change?
      </Predict>

      <p className="prose">
        It does not. A weighted average has no sense of order, so "the cat sat" and "sat cat the"
        are identical to it. Which is the next problem, and the next chapter.
      </p>

      <Provenance>
        Three properties of this implementation are asserted rather than assumed: that perturbing
        token <i>t</i> leaves every position before it bitwise unchanged, across all ten
        architecture variants; that with <code>n_kv_head == n_head</code> grouped-query attention is
        numerically identical to plain multi-head; and that the eager path used to export weights
        matches the fused SDPA kernel, so the visualizer cannot show weights the model never used.
      </Provenance>
    </>
  );
}

function Chapter5() {
  return (
    <>
      <p className="prose">
        Nothing so far records where a token sits: the embedding of "cat" is the same vector whether
        it is the first word or the ninth.
      </p>

      <Answer n={5} />

      <p className="prose">
        Position has to be injected deliberately. The old approach was a second lookup table, one
        learned vector per slot — 786,432 parameters that do nothing but say "you are eighth". The
        modern one, and the one this repository implements, is <b>rotary embeddings</b>: rotate each
        query and key by an angle proportional to its position, so that when a query meets a key the
        rotation leaves behind exactly one thing — the distance between them.
      </p>
      <p className="prose">
        It sounds like it shouldn't work. There is a whole page here where you can move two tokens
        around and watch it hold to fifteen decimal places.
      </p>
      <p className="chapter-handoff">
        <a href={href({ kind: "rope" })}>→ Open the RoPE explorer</a>
      </p>
      <Provenance>
        An off-by-one in a position table, a double-rotated key, or the adjacent-pair convention
        used where the split-half one was meant — none of these crash. The model trains, emits
        plausible English, and is quietly worse. That is why the property is asserted numerically in{" "}
        <a href={`${REPO}/tests/test_rope.py`}>
          <code>tests/test_rope.py</code>
        </a>{" "}
        rather than trusted.
      </Provenance>
    </>
  );
}

function Chapter6() {
  return (
    <>
      <p className="prose">
        Always taking the most likely token gives flat, repetitive text. Sampling freely gives
        incoherence.
      </p>

      <Answer n={6} />

      <Predict>
        raising the temperature — what do you expect it to do to the bars? Move it first, then read
        the explanation underneath.
      </Predict>

      <FigureLabel n={4}>temperature, then top-k, then top-p — the sampler's own order</FigureLabel>
      <SamplingDemo />

      <p className="prose-secondary">
        <b>Temperature</b> scales the scores before they become probabilities: below 1 sharpens
        toward the favourite, above 1 flattens toward chaos. <b>Top-k</b> keeps only the k best.{" "}
        <b>Top-p</b> keeps the smallest set covering p of the probability mass, so it stays wide
        when the model is unsure and narrow when it is confident. They are applied in that order
        because that is the order the real sampler applies them — including the off-by-one that
        makes <code>top_p = 0</code> sample from nothing.
      </p>
      <p className="prose-secondary">
        Now draw a few tokens. The text wanders, because this model only ever looks at the single
        previous token — it has no idea what it said two words ago. That gap, between counting pairs
        and understanding a sentence, is the entire reason for chapters three to five.
      </p>
      <Provenance>
        The distribution here is real, and simpler than a transformer on purpose: it is the actual
        count of what followed each token in{" "}
        <a href={`${REPO}/data/wizard_of_oz.txt`}>the repository's corpus</a> — a bigram model, the
        simplest language model there is, and the one this project began as.
      </Provenance>
    </>
  );
}

function Chapter7() {
  return (
    <>
      <p className="prose">
        Training shows the model real text and asks, at every position, what comes next. When it
        puts low probability on the true token, that is <b>loss</b> — and the loss is pushed down by
        nudging every weight slightly. Repeat ten billion tokens' worth.
      </p>

      <Answer n={7} />

      <Predict>
        this model reached a validation loss of 3.05. Is that good? The number is unreadable on its
        own — commit to a guess, then find it on the scale.
      </Predict>

      <FigureLabel n={5}>a loss, translated</FigureLabel>
      <PerplexityDemo />

      <p className="prose">
        Loss is a logarithm, so its exponential — <b>perplexity</b> — is the readable form: how many
        equally likely options the model is effectively choosing between. Note how compressed the
        useful range is. The whole distance between "knows nothing" and "reproduces GPT-2" is about
        7.5 in loss, and the last tenth of that is harder to win than the first five.
      </p>
      <p className="prose-secondary">
        It is also why an ablation arguing over 0.02 needs the paired-seed machinery: at this scale
        0.02 is a real effect that raw run-to-run noise would bury.
      </p>
    </>
  );
}

/**
 * The sweep's own numbers, from `docs/ablations.md`. The handoff was written while
 * these were still an estimate on a runbook; they are now a measurement, so they say
 * what was run rather than what it would cost.
 */
const SWEEP_FACTS = [
  { value: "12", label: "arms — eleven varying a single axis, one combining all five modern components" },
  { value: "3", label: "seeds per arm, so every comparison can be paired" },
  { value: "39", label: "runs in total, the baseline repeated at every seed" },
  { value: "7.6h", label: "of H100 time for the whole sweep" },
];

function Chapter8() {
  return (
    <>
      <p className="prose">
        Two runs differing only in random seed do not reach the same loss, and most architecture
        effects are smaller than that spread. So an unpaired comparison has to clear the whole
        spread before it can claim anything.
      </p>

      <Answer n={8} />

      <p className="prose">
        Only if the comparison is paired. Every arm runs at the same three seeds, and each is
        differenced against the baseline run that saw its data <i>in the same order</i>, cancelling
        the batch-ordering variance the two share. An arm counts as a result only when the range of
        its per-seed deltas does not straddle zero: every seed agreed on the direction. It is a
        blunt rule, and three seeds do not support a sharper one.
      </p>

      <div className="sweep-facts">
        {SWEEP_FACTS.map((fact) => (
          <div key={fact.value}>
            <PlateNumeral value={fact.value} />
            <p className="sweep-fact-label">{fact.label}</p>
          </div>
        ))}
      </div>

      <p className="prose-secondary">
        Five arms improved at every seed and five hurt at every seed; RMSNorm and zero weight decay
        came out indistinguishable from the baseline at this scale, which is a finding rather than a
        blank. The arm expected to diverge, <code>lr-3e-3</code>, turned out to be the largest
        improvement in the study.
      </p>
      <p className="chapter-handoff">
        <a href={href({ kind: "ablations" })}>→ Open the ablation playground</a>
      </p>
      <Caveat narrow>
        The sweep ran at ablation scale — 51M parameters, not the 124M of the reproduction — so its
        conclusions transfer in direction rather than in magnitude.
      </Caveat>
    </>
  );
}

export const CHAPTER_BODIES: Array<() => React.ReactElement> = [
  Chapter1,
  Chapter2,
  Chapter3,
  Chapter4,
  Chapter5,
  Chapter6,
  Chapter7,
  Chapter8,
];
