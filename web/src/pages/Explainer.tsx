import AttentionIntuition from "../components/AttentionIntuition";
import PerplexityDemo from "../components/PerplexityDemo";
import SamplingDemo from "../components/SamplingDemo";
import SizeCalculator from "../components/SizeCalculator";
import TokenizerDemo from "../components/TokenizerDemo";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";

function Stop({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <section style={{ marginTop: 52 }}>
      <p className="eyebrow">Step {n}</p>
      <h2 style={{ marginTop: 0 }}>{title}</h2>
      {children}
    </section>
  );
}

export default function Explainer() {
  return (
    <>
      <p className="eyebrow">Start here</p>
      <h1>How a language model actually works</h1>
      <p className="lede">
        No prior knowledge assumed. Eight steps from a sentence you type to a model that
        predicts what comes next — each one something you can poke at rather than take on
        faith. Everything on this page runs in your browser, and where a number is measured,
        it says what measured it.
      </p>

      <Stop n={1} title="A model never sees your text">
        <p>
          The first surprise is that "the cat sat" is not what reaches the model. Text is cut
          into <b>tokens</b> — chunks drawn from a fixed vocabulary of 50,257 entries, learned
          by finding the most common character sequences in a large pile of text. Common words
          are one token. Rare ones get split.
        </p>
        <TokenizerDemo />
        <p className="small muted">
          Three things worth noticing. A leading space belongs to the token, so{" "}
          <code> the</code> and <code>the</code> are different entries. Long or unusual words
          shatter into fragments, which is why models are worse at rare names. And the numbers
          are the only thing the model ever sees — everything downstream operates on those ids.
        </p>
        <p className="small muted">
          This is the real GPT-2 vocabulary, running the same byte-pair merges as the Python
          tokenizer, pinned to it by a fixture asserted from both sides.
        </p>
      </Stop>

      <Stop n={2} title="Each token becomes a list of numbers">
        <p>
          A token id is just an index. To do anything with it, the model looks it up in a big
          table — the <b>embedding matrix</b> — and gets back a vector of a few hundred numbers.
          That vector is the token's meaning as far as the model is concerned, and it is
          learned, not designed.
        </p>
        <p>
          That table is enormous. For a 124M-parameter model it is 50,304 × 768 numbers, which
          is about <b>31% of the entire model</b> — before a single layer of actual computation.
          Drag the sliders and watch where the budget goes.
        </p>
        <SizeCalculator />
        <p className="small muted">
          Every count here is computed from the same shapes the real model builds, and checked
          against parameter counts dumped from{" "}
          <a href={`${REPO}/src/llmfs/model/transformer.py`}>the actual <code>Transformer</code></a>{" "}
          for twelve configurations — so the arithmetic cannot quietly drift from the code.
        </p>
      </Stop>

      <Stop n={3} title="Words only mean things in context">
        <p>
          Here is the problem that makes language hard. Some words carry no meaning on their
          own — they borrow it from words that came earlier, sometimes much earlier.
        </p>
        <AttentionIntuition />
      </Stop>

      <Stop n={4} title="Attention: letting every token look back">
        <p>
          <b>Attention</b> is the answer. Each token produces a <i>query</i> ("what am I looking
          for?") and every token produces a <i>key</i> ("what do I offer?"). Compare a query
          against all the keys, and you get a score for every earlier token — how much this
          token should care about that one. Normalise the scores into weights, and take a
          weighted average of what those tokens hold.
        </p>
        <p>
          Nothing about that is hand-written. The model learns what to ask for and what to
          offer, and different <i>heads</i> learn to look for different things — one may track
          which noun a pronoun refers to, another simply the previous word.
        </p>
        <div className="grid2">
          <a
            className="card"
            href={`${import.meta.env.BASE_URL}attention/`}
            style={{ display: "block", color: "inherit", margin: 0 }}
          >
            <h3>See real attention weights ↗</h3>
            <p className="small muted" style={{ margin: 0 }}>
              Every weight from a trained model, per layer and per head. Click a token and watch
              which others it drew from. This is measured, not illustrated.
            </p>
          </a>
          <div className="card" style={{ margin: 0 }}>
            <h3>One catch</h3>
            <p className="small muted" style={{ margin: 0 }}>
              A weighted average has no sense of order — shuffle the tokens and it returns the
              same answer. So "the cat sat" and "sat cat the" would be identical to it. Which is
              the next problem.
            </p>
          </div>
        </div>
      </Stop>

      <Stop n={5} title="Telling the model where each token sits">
        <p>
          Position has to be injected deliberately. The old approach was a second lookup table,
          one learned vector per slot. The modern one — and the one this repository implements —
          is <b>rotary embeddings</b>: rotate each query and key by an angle proportional to its
          position, so that when a query meets a key, the rotation leaves behind exactly one
          thing, the distance between them.
        </p>
        <p>
          It sounds like it shouldn't work. There is a page here where you can move two tokens
          around and watch it hold.
        </p>
        <p>
          <a href="#/rope">→ Open the RoPE explorer</a>
        </p>
      </Stop>

      <Stop n={6} title="Out comes a probability for every token">
        <p>
          After a stack of these layers, the model produces one score per vocabulary entry, and
          those become probabilities. Then something has to <i>choose</i>. Always taking the
          most likely token gives flat, repetitive text; sampling freely gives incoherence. The
          knobs below are how that trade is made.
        </p>
        <p className="small muted">
          The distribution here is real, and simpler than a transformer on purpose: it is the
          actual count of what followed each token in{" "}
          <a href={`${REPO}/data/wizard_of_oz.txt`}>this repository's corpus</a> — a bigram
          model, the simplest language model there is, and the one this project began as.
        </p>
        <SamplingDemo />
        <p className="small muted">
          <b>Temperature</b> scales the scores before they become probabilities: below 1
          sharpens toward the favourite, above 1 flattens toward chaos. <b>Top-k</b> keeps only
          the k best. <b>Top-p</b> keeps the smallest set covering p of the probability mass, so
          it stays wide when the model is unsure and narrow when it is confident. They are
          applied in that order here because that is the order{" "}
          <a href={`${REPO}/src/llmfs/model/transformer.py`}>the real sampler</a> applies them.
        </p>
        <p>
          Now click "Draw a token" a few times. The text wanders, because this model only ever
          looks at the single previous token — it has no idea what it said two words ago. That
          gap, between counting pairs and understanding a sentence, is the entire reason for
          everything in steps 3 to 5.
        </p>
      </Stop>

      <Stop n={7} title="Training is just making the right token less surprising">
        <p>
          Training shows the model real text and asks, at every position, what comes next. When
          it puts low probability on the true token, that is <b>loss</b> — and the loss is
          pushed down by nudging every weight slightly. Repeat ten billion tokens' worth.
        </p>
        <p>
          The loss number on its own means little. Its exponential — <b>perplexity</b> — is
          readable: it is how many equally likely options the model is effectively choosing
          between.
        </p>
        <PerplexityDemo />
      </Stop>

      <Stop n={8} title="So does any of the design actually matter?">
        <p>
          Every step above hid a decision. RMSNorm or LayerNorm; rotary positions or learned;
          one activation or another; how many key/value heads. The papers all report
          improvements. The honest question is whether those improvements survive being measured
          carefully, at a scale you can afford, against the run-to-run noise of simply changing
          the random seed.
        </p>
        <p>
          That is what the ablation study here is for — every arm run at the same three seeds so
          comparisons can be paired, and a rule that an arm counts only if every seed agreed on
          the direction.
        </p>
        <p>
          <a href="#/ablations">→ Open the ablation playground</a>
        </p>
      </Stop>

      <hr style={{ border: 0, borderTop: "1px solid var(--border)", margin: "44px 0 22px" }} />
      <p className="small muted">
        Everything on this page is either arithmetic you can check, or a number measured by
        code in this repository and pinned to it by a test. Where something is an illustration
        rather than a measurement — the linguistic examples in step 3 — it says so.
      </p>
    </>
  );
}
