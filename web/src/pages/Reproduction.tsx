import { useState } from "react";

import DataTable from "../components/DataTable";
import LossCurve from "../components/LossCurve";
import PlateNumeral from "../components/PlateNumeral";
import PlateFoot from "../components/PlateFoot";
import { MEASURED } from "../content/measured";
import { plateHeadline, plateKicker } from "../content/path";
import { type Curve, valAt } from "../lib/reproductionCurve";
import { useResult } from "../lib/resultsData";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";
const R = MEASURED.reproduction;

export default function Reproduction() {
  const state = useResult<Curve>("reproduction-curve.json");
  const [step, setStep] = useState<number | null>(null);

  const curve = state.status === "ready" ? state.data : null;
  const at = curve ? (step ?? curve.finalStep) : 0;
  const here = curve ? valAt(curve.val, at) : null;
  const met = here != null && curve != null && here.loss <= curve.targetLoss;

  return (
    <div className="shell page">
      <p className="kicker">{plateKicker("reproduction")}</p>
      <h1 className="page-headline">{plateHeadline("reproduction")}</h1>
      <p className="page-standfirst">
        A model that generates fluent-looking text proves very little. A model that lands on a
        validation loss stated <i>before</i> the run, and then clears a public benchmark against a
        published figure, is very hard to get with a broken training pipeline. That is the only
        claim this page makes, and everything below is the evidence for it.
      </p>

      <div className="figure-strip">
        <figure>
          <PlateNumeral value={R.loss.toFixed(4)} />
          <figcaption>validation loss, against a target of {R.targetLoss} fixed before the run</figcaption>
        </figure>
        <figure>
          <PlateNumeral value={R.hellaswag.accNorm.toFixed(4)} />
          <figcaption>HellaSwag acc_norm, against the published GPT-2 124M {R.hellaswag.reference}</figcaption>
        </figure>
        <figure>
          <PlateNumeral value={`${(R.mfuMean * 100).toFixed(1)}%`} />
          <figcaption>mean model-FLOPs utilisation, flat after the first logged step</figcaption>
        </figure>
      </div>

      <div className="rule-heavy" />
      <h2 className="section-h2">The run, and the line it had to get under</h2>
      <p className="note-wide">
        Drag anywhere on the chart. The dashed line is the target, written into{" "}
        <a href={`${REPO}/configs/gpt2-124m.yaml`}>
          <code>configs/gpt2-124m.yaml</code>
        </a>{" "}
        before any of this was run — the tolerance was not chosen after seeing the result. The pale
        line is training loss every ten steps; the heavy one is validation, which is what the claim
        rests on.
      </p>

      {state.status === "loading" && <p className="fig-note">Loading the run…</p>}
      {state.status === "error" && (
        <p className="fig-note">Could not load the curve: {state.message}</p>
      )}
      {state.status === "absent" && (
        <p className="fig-note">
          The curve has not been published to the site. The scalar figures above come from{" "}
          <code>results/reproduction.json</code> and are unaffected.
        </p>
      )}

      {curve && (
        <div className="figure-panel">
          <h3 className="figure-title">Validation loss against training step</h3>
          <LossCurve curve={curve} step={at} onStep={setStep} />

          <input
            className="scrub"
            type="range"
            min={0}
            max={curve.finalStep}
            step={10}
            value={at}
            onChange={(e) => setStep(Number(e.target.value))}
            aria-label="Training step"
          />

          <div className="readouts" style={{ marginTop: "var(--space-3)" }}>
            <div>
              <p className="eyebrow">At step {at.toLocaleString()}</p>
              <div className="readout-xl">{here ? here.loss.toFixed(4) : "—"}</div>
              <div className="readout-pair">
                <div>
                  <p className="eyebrow">Through the run</p>
                  <div className="readout-md">{((at / curve.finalStep) * 100).toFixed(0)}%</div>
                </div>
                <div>
                  <p className="eyebrow">Perplexity</p>
                  <div className="readout-md">{here ? here.perplexity.toFixed(1) : "—"}</div>
                </div>
              </div>
            </div>

            <div>
              <p className="eyebrow" style={{ color: met ? "var(--color-accent-700)" : "var(--color-accent-2-700)" }}>
                {met ? "Target met" : "Target not yet met"}
              </p>
              <div className={`readout-xl ${met ? "spread-holding" : "spread-moving"}`}>
                {here ? (here.loss - curve.targetLoss >= 0 ? "+" : "") + (here.loss - curve.targetLoss).toFixed(4) : "—"}
              </div>
              <p className="tracker-sentence">
                {curve.crossing ? (
                  <>
                    The run first met {curve.targetLoss} at step{" "}
                    <b>{curve.crossing.step.toLocaleString()}</b> —{" "}
                    <b>{(curve.crossing.fractionOfRun * 100).toFixed(0)}%</b> of the way in — and
                    then kept improving for the remaining{" "}
                    {(100 - curve.crossing.fractionOfRun * 100).toFixed(0)}%, finishing{" "}
                    <b>{(curve.targetLoss - R.loss).toFixed(4)}</b> below it. A target hit on the
                    last step would be a target chosen to be hit.
                  </>
                ) : (
                  <>The target was not met during this run.</>
                )}
              </p>
            </div>
          </div>
        </div>
      )}

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">Why the loss alone would not be enough</h2>
      <div className="two-col">
        <p className="prose">
          Validation loss is measured on a split chosen for this project, with a tokenizer configured
          for it. A
          mismatch in either would move the number without looking wrong — a slightly easier held-out
          split, or a tokenizer that fragments differently, and the figure improves for reasons that
          have nothing to do with the model. Self-consistency is cheap.
        </p>
        <p className="prose">
          HellaSwag is the independent check: a fixed public set, scored against a published figure
          for the model being reproduced. Clearing both chance ({R.hellaswag.chance}) and GPT-2
          124M's {R.hellaswag.reference} is what makes the loss mean something. Near chance, the loss
          would have meant nothing at all, and the honest reading would have been that something in
          the evaluation was wrong.
        </p>
      </div>

      <DataTable label="Reproduction measures against their references" style={{ marginTop: "var(--space-4)" }}>
        <thead>
          <tr>
            <th>Measure</th>
            <th className="num">Achieved</th>
            <th className="num">Reference</th>
            <th className="num">Margin</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td>Validation loss, full {(R.tokensEvaluated / 1e6).toFixed(0)}M-token split</td>
            <td className="num mono">{R.loss.toFixed(4)}</td>
            <td className="num mono">≤ {R.targetLoss}</td>
            <td className="num mono">{(R.loss - R.targetLoss).toFixed(4)}</td>
          </tr>
          <tr>
            <td>Perplexity</td>
            <td className="num mono">{R.perplexity.toFixed(2)}</td>
            <td className="num mono">—</td>
            <td className="num mono">—</td>
          </tr>
          <tr>
            <td>
              HellaSwag <code>acc_norm</code>, all {R.hellaswag.nEvaluated.toLocaleString()} items
            </td>
            <td className="num mono">{R.hellaswag.accNorm.toFixed(4)}</td>
            <td className="num mono">{R.hellaswag.reference}</td>
            <td className="num mono">+{(R.hellaswag.accNorm - R.hellaswag.reference).toFixed(4)}</td>
          </tr>
          <tr>
            <td>
              HellaSwag <code>acc</code>
            </td>
            <td className="num mono">{R.hellaswag.acc.toFixed(4)}</td>
            <td className="num mono">{R.hellaswag.chance} (chance)</td>
            <td className="num mono">+{(R.hellaswag.acc - R.hellaswag.chance).toFixed(4)}</td>
          </tr>
        </tbody>
      </DataTable>

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">What it cost, and on what</h2>
      <p className="note-wide">
        {(R.tokensTrained / 1e9).toFixed(1)}B tokens of FineWeb-Edu on one {R.gpu}, at a mean{" "}
        {(R.mfuMean * 100).toFixed(1)}% of the card's usable FLOPs. After the first logged step —
        which came in at {(R.mfuWarmup * 100).toFixed(0)}%, paying for compilation and allocator
        warmup once — it never left {(R.mfuMin * 100).toFixed(1)}–{(R.mfuMax * 100).toFixed(1)}%.
        Flat utilisation for seven hours is the part worth noticing: a run that starts fast and
        degrades is the usual signature of a data loader falling behind or memory fragmenting, and
        neither happened here.
      </p>

      <div className="rule-heavy" style={{ margin: "var(--space-6) 0 var(--space-4)" }} />
      <div className="closing-cols">
        <p className="closing-lead">
          <b>Every figure on this page is read from an artifact.</b> The scalars come from{" "}
          <code>results/reproduction.json</code> and <code>results/hellaswag.json</code>; the curve
          is the run's own <code>metrics.jsonl</code>, lifted into{" "}
          <code>results/reproduction-curve.json</code>. A test asserts the site's copy of them is
          still what the generator produces, and a second asserts the target appears in the config
          that pre-registered it.
        </p>
        <p className="closing-note">
          The full protocol — target provenance, hardware, the evaluation harness, and sample
          generations — is in{" "}
          <a href={`${REPO}/docs/reproduction.md`}>
            <code>docs/reproduction.md</code>
          </a>
          . It was written before the run and has not been edited since, which is the only reason
          the word "pre-registered" is doing any work.
        </p>
      </div>

      <PlateFoot current="reproduction" />
    </div>
  );
}
