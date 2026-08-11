import AttentionIntuition from "../components/AttentionIntuition";
import PerplexityDemo from "../components/PerplexityDemo";
import PlateNumeral from "../components/PlateNumeral";
import SamplingDemo from "../components/SamplingDemo";
import SizeCalculator from "../components/SizeCalculator";
import TokenizerDemo from "../components/TokenizerDemo";
import { href } from "../router";

/**
 * The eight chapters' copy, adapted from the page this redesign replaces.
 *
 * Figures are introduced by their label and then left alone: the components carry
 * their own logic, and the only thing a chapter decides is where one sits in the
 * argument.
 */

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";
const attentionExplorer = () => `${import.meta.env.BASE_URL}attention/`;

function FigureLabel({ n, children }: { n: number; children: React.ReactNode }) {
  return (
    <p className="figure-label">
      Figure {n} · {children}
    </p>
  );
}

function Chapter1() {
  return (
    <>
      <p className="prose">
        The first surprise is that "the cat sat" is not what reaches the model. Text is cut into{" "}
        <b>tokens</b> — chunks drawn from a fixed vocabulary of 50,257 entries, learned by finding
        the most common character sequences in a large pile of text. Common words are one token.
        Rare ones get split.
      </p>

      <FigureLabel n={1}>the real GPT-2 vocabulary, in your browser</FigureLabel>
      <TokenizerDemo />

      <p className="prose-secondary">
        Three things worth noticing. A leading space belongs to the token, so <code>&nbsp;the</code>{" "}
        and <code>the</code> are different entries. Long or unusual words shatter into fragments,
        which is why models are worse at rare names. And the ids are the only thing the model ever
        sees — everything downstream operates on those.
      </p>
      <p className="prose-caveat">
        This runs the same byte-pair merges as the Python tokenizer, pinned to it by a{" "}
        <a href={`${REPO}/tests/test_tokenizer.py`}>fixture asserted from both sides</a> — exact on
        all 14 cases, including emoji, newlines and leading spaces.
      </p>
    </>
  );
}

function Chapter2() {
  return (
    <>
      <p className="prose">
        A token id is just an index. To do anything with it, the model looks it up in a big table —
        the <b>embedding matrix</b> — and gets back a vector of a few hundred numbers. That vector
        is the token's meaning as far as the model is concerned, and it is learned, not designed.
      </p>
      <p className="prose">
        That table is enormous. For a 124M-parameter model it is 50,304 × 768 numbers, about{" "}
        <b>31% of the entire model</b> — before a single layer of actual computation. Drag the
        sliders and watch where the budget goes.
      </p>

      <FigureLabel n={2}>parameter, memory and FLOP budget</FigureLabel>
      <SizeCalculator />

      <p className="prose-secondary">
        Drag the context slider and watch the last two figures move while everything else stays
        flat. Attention's cost grows with sequence length; that is why long context is hard, and
        why the KV cache — the thing grouped-query attention shrinks — is what decides how many
        users fit on one GPU.
      </p>
      <p className="prose-caveat">
        Every count is computed from the same shapes the real model builds, and checked against
        parameter counts dumped from{" "}
        <a href={`${REPO}/src/llmfs/model/transformer.py`}>
          the actual <code>Transformer</code>
        </a>{" "}
        for twelve configurations, so the arithmetic cannot quietly drift from the code.
      </p>
    </>
  );
}

function Chapter3() {
  return (
    <>
      <p className="prose">
        Here is the problem that makes language hard. Some words carry no meaning on their own —
        they borrow it from words that came earlier, sometimes much earlier.
      </p>

      <FigureLabel n={3}>an illustration, not model output</FigureLabel>
      <AttentionIntuition />

      <p className="prose-secondary">
        A model that reads strictly left to right with no memory cannot do this — by the time it
        reaches "it", the trophy is gone. Attention is the mechanism that lets every token pull
        information from any earlier token, and learn <i>which</i> ones are worth pulling from.
        These particular links are linguistic illustrations, not model output; what this
        repository's model actually attended to is in the attention explorer.
      </p>
    </>
  );
}

function Chapter4() {
  return (
    <>
      <p className="prose">
        <b>Attention</b> is the answer. Each token produces a <i>query</i> — what am I looking
        for? — and every token produces a <i>key</i> — what do I offer? Compare a query against all
        the keys and you get a score for every earlier token: how much this token should care about
        that one. Normalise the scores into weights, and take a weighted average of what those
        tokens hold.
      </p>
      <p className="prose">
        Nothing about that is hand-written. The model learns what to ask for and what to offer, and
        different <i>heads</i> learn to look for different things — one may track which noun a
        pronoun refers to, another simply the previous word.
      </p>

      <div className="two-col">
        <div>
          <h3 className="column-title">
            <a href={attentionExplorer()}>See the real weights ↗</a>
          </h3>
          <p className="prose-secondary">
            Every weight from a trained model, per layer and per head, in a single self-contained
            HTML file — no build step, no CDN, no backend, and a test that asserts no external
            resource is ever referenced. Click a token and watch which others it drew from.
            Measured, not illustrated.
          </p>
        </div>
        <div>
          <h3 className="column-title">One catch</h3>
          <p className="prose-secondary">
            A weighted average has no sense of order — shuffle the tokens and it returns the same
            answer. So "the cat sat" and "sat cat the" would be identical to it. Which is the next
            problem, and the next chapter.
          </p>
        </div>
      </div>

      <p className="prose-caveat">
        Three properties of this implementation are asserted rather than assumed: that perturbing
        token <i>t</i> leaves every position before it bitwise unchanged, across all ten
        architecture variants; that with <code>n_kv_head == n_head</code> grouped-query attention is
        numerically identical to plain multi-head; and that the eager path used to export weights
        matches the fused SDPA kernel, so the visualizer cannot show weights the model never used.
      </p>
    </>
  );
}

function Chapter5() {
  return (
    <>
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
      <p className="prose-caveat">
        An off-by-one in a position table, a double-rotated key, or the adjacent-pair convention
        used where the split-half one was meant — none of these crash. The model trains, emits
        plausible English, and is quietly worse. That is why the property is asserted numerically in{" "}
        <a href={`${REPO}/tests/test_rope.py`}>
          <code>tests/test_rope.py</code>
        </a>{" "}
        rather than trusted.
      </p>
    </>
  );
}

function Chapter6() {
  return (
    <>
      <p className="prose">
        After a stack of these layers the model produces one score per vocabulary entry, and those
        become probabilities. Then something has to <i>choose</i>. Always taking the most likely
        token gives flat, repetitive text; sampling freely gives incoherence. The knobs below are
        how that trade is made.
      </p>
      <p className="prose-caveat" style={{ marginBottom: "var(--space-4)" }}>
        The distribution here is real, and simpler than a transformer on purpose: it is the actual
        count of what followed each token in{" "}
        <a href={`${REPO}/data/wizard_of_oz.txt`}>the repository's corpus</a> — a bigram model, the
        simplest language model there is, and the one this project began as.
      </p>

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
        and understanding a sentence, is the entire reason for everything in chapters three to five.
      </p>
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
      <p className="prose">
        The loss number on its own means little. Its exponential — <b>perplexity</b> — is readable:
        it is how many equally likely options the model is effectively choosing between.
      </p>

      <FigureLabel n={5}>a loss, translated</FigureLabel>
      <PerplexityDemo />

      <p className="prose-secondary">
        Note how compressed the useful range is. The whole distance between "knows nothing" and
        "reproduces GPT-2" is about 7.5 in loss — and the last tenth of that is harder to win than
        the first five. It is also why an ablation arguing over 0.02 needs the paired-seed
        machinery: at this scale 0.02 is a real effect that raw run-to-run noise would bury.
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
        Every chapter above hid a decision. RMSNorm or LayerNorm; rotary positions or learned; one
        activation or another; how many key/value heads. The papers all report improvements. The
        honest question is whether those improvements survive being measured carefully, at a scale
        you can afford, against the run-to-run noise of simply changing the random seed.
      </p>
      <p className="prose">
        Two runs differing only in seed do not reach the same loss, so an unpaired comparison has to
        clear that entire spread before it can claim anything — and most architecture effects are
        smaller than it. So every arm runs at the same three seeds, and each is differenced against
        the baseline run that saw its data <i>in the same order</i>, cancelling the batch-ordering
        variance the two share.
      </p>
      <p className="prose">
        An arm counts as a result only when the range of its per-seed deltas does not straddle zero:
        every seed agreed on the direction. A deliberately blunt rule rather than a p-value — with
        three seeds nothing stronger would be honest. An ablation table without that check is worse
        than no table, because it reads as authoritative while recommending changes that do nothing.
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
        The sweep has run. Five arms improved at every seed and five hurt at every seed; RMSNorm and
        zero weight decay came out indistinguishable from the baseline at this scale, which is a
        finding rather than a blank. The arm expected to diverge, <code>lr-3e-3</code>, turned out
        to be the largest improvement in the study.
      </p>
      <p className="chapter-handoff">
        <a href={href({ kind: "ablations" })}>→ Open the ablation playground</a>
      </p>
      <p className="prose-caveat">
        One caveat the table carries with it: the sweep ran at ablation scale — 51M parameters, not
        the 124M of the reproduction — so its conclusions transfer in direction rather than in
        magnitude.
      </p>
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
