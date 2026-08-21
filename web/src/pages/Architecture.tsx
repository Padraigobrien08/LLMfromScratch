import { useState } from "react";

import DataTable from "../components/DataTable";
import Caveat from "../components/Caveat";
import { ARCHITECTURES } from "../content/architecture";
import { BLOCKS, SIZES, type Variant } from "../content/blocks";
import { MEASURED } from "../content/measured";
import { PROJECT } from "../content/projectState";
import { formatCount, parameters } from "../lib/modelsize";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";

const VARIANTS: Array<{ id: Variant; label: string; note: string }> = [
  { id: "gpt2", label: "GPT-2", note: "the reproduction baseline" },
  { id: "llama", label: "Llama-style", note: "the modern stack" },
];

export default function Architecture() {
  const [variant, setVariant] = useState<Variant>("gpt2");
  const [selected, setSelected] = useState<string>(BLOCKS[0]!.id);

  const block = BLOCKS.find((b) => b.id === selected) ?? BLOCKS[0]!;
  const total = parameters(SIZES[variant]).total;
  const source = ARCHITECTURES[variant].source;

  return (
    <div className="shell page">
      <p className="kicker">The deep end · the stack</p>
      <h1 className="page-headline">One class, two architectures, decided by config</h1>
      <p className="page-standfirst">
        There is a single <code>Transformer</code> in this repository. Whether you get the GPT-2
        baseline or the modern Llama-style stack is decided entirely by YAML, which is deliberate:
        an ablation that swaps LayerNorm for RMSNorm differs from its baseline by one line, so the
        two cannot silently drift apart. Click a block for its shape, its share of the parameter
        budget, and what holds it.
      </p>

      <div className="fig-row fig-row-wide" style={{ marginBottom: "var(--space-4)" }}>
        {/* Radios in labels, which is the design system's own segmented-control markup —
            `.seg-opt` styles its selected state from `:has(input:checked)`, so buttons
            carrying `aria-pressed` got no visual selection at all. It is also the right
            control: this is one choice out of two, so a radio group gives arrow-key
            navigation and the correct announcement for free, where `role="tab"` would
            have promised a `tabpanel` that does not exist. */}
        <div className="seg seg-paper" role="radiogroup" aria-label="Architecture">
          {VARIANTS.map((v) => (
            <label key={v.id} className="seg-opt">
              <input
                type="radio"
                name="architecture"
                value={v.id}
                checked={variant === v.id}
                onChange={() => setVariant(v.id)}
              />
              {v.label}
            </label>
          ))}
        </div>
        <span className="fig-note" style={{ margin: 0 }}>
          {VARIANTS.find((v) => v.id === variant)!.note} · <code>{source}</code>, resolved through
          the repository's own config loader. <b>{formatCount(total)}</b> parameters.
        </span>
      </div>

      <div className="stack-layout">
        <ol className="stack-column">
          {BLOCKS.map((b) => {
            const params = b.params(variant);
            const repeat = b.repeat?.(variant);
            return (
              <li key={b.id}>
                <button
                  className="stack-block"
                  aria-current={b.id === selected ? "true" : undefined}
                  onClick={() => setSelected(b.id)}
                >
                  <span className="stack-block-head">
                    <span className="stack-block-title">{b.title}</span>
                    {repeat != null && <span className="stack-block-repeat">×{repeat}</span>}
                  </span>
                  <span className="stack-block-summary">{b.summary}</span>
                  <span className="stack-block-params">
                    {params == null ? (
                      <span className="stack-block-noparams">no parameters</span>
                    ) : (
                      <>
                        {formatCount(params)}
                        <span className="stack-block-share">
                          {((params / total) * 100).toFixed(1)}% of the budget
                        </span>
                      </>
                    )}
                  </span>
                </button>
              </li>
            );
          })}
        </ol>

        <aside className="stack-detail">
          <div className="stack-detail-inner">
            <p className="eyebrow">{block.title}</p>
            <p className="stack-detail-shape mono">{block.shape(variant)}</p>
            <p className="stack-detail-what">{block.what}</p>

            {block.differs?.(variant) && (
              <>
                <p className="eyebrow" style={{ marginTop: "var(--space-3)" }}>
                  In this configuration
                </p>
                <p className="stack-detail-what">{block.differs(variant)}</p>
              </>
            )}

            <p className="eyebrow" style={{ marginTop: "var(--space-3)" }}>
              What holds it
            </p>
            {block.pins ? (
              <p className="stack-detail-pin">
                <code className="mono">{block.pins.test}</code> asserts {block.pins.claim}
              </p>
            ) : (
              <p className="stack-detail-pin stack-detail-unpinned">
                No property test of its own. The suite checks this block's shapes and its
                contribution to the parameter count, but nothing here asserts a mathematical
                invariant the way RoPE's or causality's tests do. Saying so beats borrowing a
                neighbour's test to fill the row.
              </p>
            )}
          </div>
        </aside>
      </div>

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">Where the budget actually goes</h2>
      <DataTable label="Parameter budget by block">
        <thead>
          <tr>
            <th>Block</th>
            <th className="num">GPT-2</th>
            <th className="num">Llama-style</th>
            <th className="num">Difference</th>
          </tr>
        </thead>
        <tbody>
          {BLOCKS.filter((b) => b.params("gpt2") != null || b.params("llama") != null).map((b) => {
            const g = b.params("gpt2") ?? 0;
            const l = b.params("llama") ?? 0;
            return (
              <tr key={b.id}>
                <td>{b.title}</td>
                <td className="num mono">{g ? formatCount(g) : "—"}</td>
                <td className="num mono">{l ? formatCount(l) : "—"}</td>
                <td className="num mono">
                  {g === l ? "same" : `${l > g ? "+" : "−"}${formatCount(Math.abs(l - g))}`}
                </td>
              </tr>
            );
          })}
          <tr>
            <td>
              <b>Total</b>
            </td>
            <td className="num mono">
              <b>{formatCount(parameters(SIZES.gpt2).total)}</b>
            </td>
            <td className="num mono">
              <b>{formatCount(parameters(SIZES.llama).total)}</b>
            </td>
            <td className="num mono">
              {formatCount(
                Math.abs(parameters(SIZES.llama).total - parameters(SIZES.gpt2).total),
              )}
            </td>
          </tr>
        </tbody>
      </DataTable>
      <Caveat columns>
        <b>Worth reading the difference column rather than the totals.</b> The one place the two
        configs were deliberately matched is the feed-forward: SwiGLU has three projections rather
        than two, so its hidden width carries a 2/3 correction, and the row comes out within{" "}
        {formatCount(
          Math.abs(
            (BLOCKS.find((b) => b.id === "feed-forward")!.params("llama") ?? 0) -
              (BLOCKS.find((b) => b.id === "feed-forward")!.params("gpt2") ?? 0),
          ),
        )}{" "}
        ; without that correction a GELU-versus-SwiGLU comparison would be measuring the parameter
        budget instead of the activation.
        <br />
        <br />
        Everywhere else the modern stack is genuinely smaller, by{" "}
        {formatCount(Math.abs(parameters(SIZES.llama).total - parameters(SIZES.gpt2).total))} in
        total, and the savings are the point rather than an accident: RoPE removes the position
        table outright, dropping the biases removes one vector per projection, and grouped-query
        attention shrinks the KV half of every QKV matrix by three quarters. The last of those is
        the largest single line here and it buys a 3× smaller cache at inference, which is a
        memory result rather than a quality one. The ablation study is where the quality cost gets
        measured, and it came to{" "}
        {/* Read from the export, not typed. The line thirty rows down promises nothing on this
            page is arithmetic anyone did by hand, and a hand-copied delta here would have
            been the one thing on it that was. */}
        {MEASURED.ablations.armDeltas["gqa-2"]!.delta! > 0 ? "+" : ""}
        {MEASURED.ablations.armDeltas["gqa-2"]!.delta!.toFixed(4)} validation loss.
      </Caveat>

      <div className="rule-heavy" style={{ margin: "var(--space-6) 0 var(--space-4)" }} />
      <div className="closing-cols">
        <p className="closing-lead">
          <b>Nothing on this page is arithmetic anyone did by hand.</b> Every parameter count comes
          from the same calculator the explainer uses, pinned exactly, not approximately, to the
          real <code>Transformer</code> across twelve configurations. Every config value is
          generated by resolving the shipped YAML through{" "}
          <a href={`${REPO}/src/llmfs/config.py`}>
            <code>llmfs.config.load_config</code>
          </a>
          , because <code>llama-124m.yaml</code> never states <code>n_layer</code> and{" "}
          <code>gpt2-124m.yaml</code> never states <code>n_kv_head</code>; only the resolved config
          has the fields a reader would check.
        </p>
        <p className="closing-note">
          The "what holds it" lines name real tests, and each was written by reading that test
          rather than its name. Two blocks say they have no property test, which is the honest
          answer: a page whose argument is <i>this is pinned</i> is the worst place to be vague
          about what is. A test asserts every name here still exists. All{" "}
          {PROJECT.pythonTests.toLocaleString()} of them run on every push.
        </p>
      </div>
    </div>
  );
}
