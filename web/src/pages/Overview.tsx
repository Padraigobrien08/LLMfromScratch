const REPO = "https://github.com/Padraigobrien08/LLMfromScratch";

/**
 * Status is duplicated from the README on purpose, and kept in the same words.
 * A site that claims more than the README is the failure mode worth designing
 * against — every row here says what has actually been run.
 */
const STATUS: Array<[string, string, "done" | "pending" | "not-started"]> = [
  ["Package, config system, data pipeline, trainer, CI", "222 tests green, end-to-end verified", "done"],
  ["Modern architecture (RoPE, RMSNorm, SwiGLU, GQA, KV cache)", "Hand-implemented, property-tested", "done"],
  ["Interactive attention explorer", "Live, rebuilt by CI on every push", "done"],
  ["Fault-tolerance design doc", "Written, with a prioritised gap list", "done"],
  ["Ablation study (13 arms × 3 seeds)", "Runner and report built; GPU run in progress", "pending"],
  ["GPT-2 124M reproduction on FineWeb-Edu", "Configured; GPU run pending", "pending"],
  ["Efficiency benchmarks (throughput, memory, KV cache)", "Built; GPU run pending", "pending"],
  ["Quantization + speculative decoding", "Not started", "not-started"],
];

const BADGE: Record<string, { text: string; color: string }> = {
  done: { text: "done", color: "var(--good)" },
  pending: { text: "pending", color: "var(--warn)" },
  "not-started": { text: "not started", color: "var(--muted)" },
};

export default function Overview() {
  return (
    <>
      <h1>A language model built from scratch, and the evidence it works</h1>
      <p className="lede">
        A decoder-only transformer written by hand — rotary embeddings, RMSNorm, SwiGLU,
        grouped-query attention, a KV cache — with a GPT-2 124M reproduction, a paired-seed ablation
        study, and efficiency benchmarks. This site is the part you can click.
      </p>

      <a
        className="card"
        href="#/explainer"
        style={{ display: "block", color: "inherit", marginTop: 26 }}
      >
        <p className="eyebrow">Start here · no prior knowledge assumed</p>
        <h3>How a language model actually works</h3>
        <p className="small muted" style={{ margin: 0 }}>
          Eight steps from a sentence you type to a model that predicts what comes next —
          tokenization, embeddings, attention, position, sampling and loss, each one something
          you can poke at rather than take on faith. Every number is either arithmetic you can
          check or measured by code in this repository.
        </p>
      </a>

      <div className="grid2">
        <a className="card" href="#/rope" style={{ display: "block", color: "inherit" }}>
          <p className="eyebrow">Explorable explanation</p>
          <h3>What rotary embeddings actually do</h3>
          <p className="small muted" style={{ margin: 0 }}>
            Drag two tokens along a sequence and watch the attention logit between them refuse to
            change, as long as the gap between them does not. The defining property of RoPE, running
            live in the browser against a port pinned to the model's own code.
          </p>
        </a>
        <a className="card" href="#/ablations" style={{ display: "block", color: "inherit" }}>
          <p className="eyebrow">Measured results</p>
          <h3>The ablation playground</h3>
          <p className="small muted" style={{ margin: 0 }}>
            Thirteen arms, three seeds each, every comparison paired against the baseline run that
            saw the same data in the same order. Toggle a design decision and see what it actually
            bought — including when the honest answer is "nothing measurable".
          </p>
        </a>
      </div>

      <div className="card" style={{ display: "block" }}>
        <p className="eyebrow">Separate page</p>
        <h3>
          <a href={`${import.meta.env.BASE_URL}attention/`}>The attention explorer ↗</a>
        </h3>
        <p className="small muted" style={{ margin: 0 }}>
          Every attention weight, per layer and per head, in a single self-contained HTML file with
          no build step, no CDN and no backend — built by CI from a model CI trains, so it always
          reflects the current code. It stays standalone precisely because that claim is worth
          keeping true.
        </p>
      </div>

      <h2>Honest status</h2>
      <p className="small muted" style={{ marginTop: 0 }}>
        No result appears on this site that has not been measured. Rows that are pending say so.
      </p>
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Pillar</th>
              <th>State</th>
              <th style={{ width: 100 }}>Status</th>
            </tr>
          </thead>
          <tbody>
            {STATUS.map(([pillar, state, status]) => (
              <tr key={pillar}>
                <td>{pillar}</td>
                <td className="muted small">{state}</td>
                <td>
                  <span className="chip" style={{ color: BADGE[status]!.color }}>
                    {BADGE[status]!.text}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <h2>What makes this more than a tutorial reproduction</h2>
      <p>
        The tests assert defining mathematical properties, not output shapes: that RoPE's logit
        depends only on relative position; that perturbing token <i>t</i> leaves every earlier
        position bitwise unchanged, across all ten architecture variants; that incremental KV-cache
        decoding reproduces a full forward pass at every position. Shape checks catch typos. These
        catch the bugs that otherwise survive into a training run and surface only as a mysteriously
        worse loss.
      </p>
      <p>
        And the ablation study is built around the question most ablation tables skip — how big a
        difference is big enough to mean anything? Every arm runs at the same three seeds so
        comparisons can be paired, and an arm counts as a result only when every seed agreed on the
        direction.
      </p>
      <p className="small">
        <a href={REPO}>Source on GitHub ↗</a>
      </p>
    </>
  );
}
