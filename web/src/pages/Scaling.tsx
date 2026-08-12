import { useState } from "react";

import DataTable from "../components/DataTable";
import AccumCurve from "../components/AccumCurve";
import PlateNumeral from "../components/PlateNumeral";
import PlateFoot from "../components/PlateFoot";
import Caveat from "../components/Caveat";
import { MEASURED } from "../content/measured";
import { plateHeadline, plateKicker } from "../content/path";
import { efficiencyAt, residual } from "../lib/amortisation";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";
const S = MEASURED.scaling;
const ACC = MEASURED.accumulation;
const { a, b } = ACC.fit;

/** The slider moves in log2(accum), so each halving gets the same travel. */
const STEPS = 48;
const toAccum = (i: number) => 2 ** ((i / STEPS) * 3);
const nearest = (accum: number) =>
  ACC.points.find((p) => Math.abs(Math.log2(p.accum) - Math.log2(accum)) < 0.06) ?? null;

export default function Scaling() {
  const [slider, setSlider] = useState(STEPS);
  const [revealed, setRevealed] = useState(false);

  const accum = toAccum(slider);
  const measured = nearest(accum);
  const predicted = efficiencyAt(a, b, accum) * 100;
  const eight = S.points.find((p) => p.worldSize === 8)!;

  return (
    <div className="shell page">
      <p className="kicker">{plateKicker("scaling")}</p>
      <h1 className="page-headline">{plateHeadline("scaling")}</h1>
      <p className="page-standfirst">
        {(eight.efficiency * 100).toFixed(1)}% scaling efficiency on eight cards with{" "}
        <b>no NVLink</b> — over a dual-socket box where half the GPUs sit on the other NUMA node, so
        the eight-way all-reduce crosses the inter-socket link. That number is easy to report and
        hard to interpret on its own, so the second half of this page is an experiment that says
        what it is made of.
      </p>

      <div className="figure-strip">
        <figure>
          <PlateNumeral value={`${(eight.efficiency * 100).toFixed(1)}%`} />
          <figcaption>of linear scaling at 8 GPUs, {S.interconnect} only</figcaption>
        </figure>
        <figure>
          <PlateNumeral value={`${(eight.tokensPerSec / 1e6).toFixed(2)}M`} />
          <figcaption>tokens per second, the real trainer under torchrun</figcaption>
        </figure>
        <figure>
          <PlateNumeral value={eight.maxLossDeltaVs1Gpu.toExponential(0).replace("e-", "e−")} />
          <figcaption>largest loss divergence from a single GPU over 50 steps</figcaption>
        </figure>
      </div>

      <div className="rule-heavy" />
      <h2 className="section-h2">Throughput is the easy half</h2>
      <DataTable label="Scaling by world size">
        <thead>
          <tr>
            <th className="num">GPUs</th>
            <th className="num">Grad accum</th>
            <th className="num">Tokens/sec</th>
            <th className="num">Per GPU</th>
            <th className="num">Efficiency</th>
            <th className="num">Max Δloss vs 1 GPU</th>
          </tr>
        </thead>
        <tbody>
          {S.points.map((p) => (
            <tr key={p.worldSize}>
              <td className="num mono">{p.worldSize}</td>
              <td className="num mono">{p.gradAccum}</td>
              <td className="num mono">{p.tokensPerSec.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td className="num mono">{p.tokensPerSecPerGpu.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
              <td className="num mono">{p.worldSize === 1 ? "—" : `${(p.efficiency * 100).toFixed(1)}%`}</td>
              <td className="num mono">
                {p.worldSize === 1 ? "baseline" : p.maxLossDeltaVs1Gpu.toExponential(1)}
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      <Caveat columns>
        The claim worth checking is the last column, not the third. Eight GPUs take the{" "}
        <i>same optimisation steps</i> as one: <code>tokens_per_step</code> is fixed in tokens and
        the accumulation is derived from it, so the world size changes how the batch is gathered and
        nothing about what is optimised. The loss at step one is identical to sixteen significant
        figures at 1, 2 and 4 GPUs, and the largest divergence over fifty steps does not grow with
        the world size — that is floating-point reduction order, not drift. Measured with the real
        trainer under <code>torchrun</code>, not a synthetic loop, because a hand-written loop would
        omit the two things most likely to spoil scaling: the gradient all-reduce and the optimiser
        step.
      </Caveat>

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">What the batch was hiding</h2>
      <p className="note-wide">
        Gradient accumulation means one all-reduce per <i>optimiser</i> step rather than per
        micro-batch, so the communication is spread over however much compute accumulates into it.
        Hold the world size at eight, shrink the batch, and the cost stops being hidden. Two of the
        four points below were used to fit the curve; the other two were <b>predicted before they
        were measured</b>.
      </p>

      <div className="figure-panel">
        <h3 className="figure-title">Efficiency at 8 GPUs against gradient accumulation</h3>
        <AccumCurve accum={accum} revealed={revealed} />

        <input
          className="scrub"
          type="range"
          min={0}
          max={STEPS}
          value={slider}
          onChange={(e) => setSlider(Number(e.target.value))}
          aria-label="Gradient accumulation steps"
        />

        <div className="fig-row fig-row-wide" style={{ marginTop: "var(--space-2)" }}>
          <button
            className={revealed ? "btn btn-secondary" : "btn btn-primary"}
            onClick={() => setRevealed((r) => !r)}
          >
            {revealed ? "Hide the two it did not see" : "Reveal the two it did not see"}
          </button>
          <span className="fig-note" style={{ margin: 0 }}>
            Fitted to accum {ACC.fit.fittedFrom.join(" and ")} only —{" "}
            <code className="mono">
              loss = {a.toFixed(3)} + {b.toFixed(3)}/accum
            </code>{" "}
            percentage points.
          </span>
        </div>

        <div className="readouts" style={{ marginTop: "var(--space-3)" }}>
          <div>
            <p className="eyebrow">The model says, at accum {accum.toFixed(accum < 2 ? 2 : 1)}</p>
            <div className="readout-xl">{predicted.toFixed(2)}%</div>
            <p className="tracker-sentence">
              The curve answers for any accumulation. The repository only ran four, which is the
              difference between a model and a table — and the reason the two hidden points are
              worth anything.
            </p>
          </div>
          <div>
            <p className="eyebrow">Measured here</p>
            <div className="readout-xl">
              {measured ? `${(measured.efficiency * 100).toFixed(2)}%` : "not run"}
            </div>
            <p className="tracker-sentence">
              {measured ? (
                <>
                  {measured.predicted ? (
                    <>
                      <b>Predicted, then measured.</b> The fit never saw this point and missed it by{" "}
                      <b>{Math.abs(residual(a, b, measured.accum, measured.efficiency)).toFixed(2)}</b>{" "}
                      percentage points, across a further {8 / measured.accum}× of range.
                    </>
                  ) : (
                    <>
                      <b>One of the two points the fit was given.</b> It passes through this exactly,
                      which is arithmetic rather than evidence — the evidence is what happens at the
                      other two.
                    </>
                  )}{" "}
                  {measured.tokensPerSec.toLocaleString(undefined, { maximumFractionDigits: 0 })}{" "}
                  tokens/sec at {measured.tokensPerStep.toLocaleString()} tokens per step.
                </>
              ) : (
                <>
                  No run at this accumulation. Slide to 1, 2, 4 or 8 to compare the model against a
                  measurement instead of against itself.
                </>
              )}
            </p>
          </div>
        </div>
      </div>

      <Caveat columns>
        With one all-reduce per micro-batch, efficiency falls to{" "}
        {(ACC.points.find((p) => p.accum === 1)!.efficiency * 100).toFixed(1)}%. So communication is
        exactly what the accumulation was hiding — and the split says how much of it there is: about{" "}
        <b>{a.toFixed(1)} points</b> that do not amortise at all, and{" "}
        <b>{b.toFixed(1)} points</b> divided by however much compute each all-reduce is spread
        across. At the reproduction's configuration a perfect interconnect could therefore recover
        roughly {a.toFixed(1)} points, which is why the planned NVLink comparison was dropped in
        favour of this experiment: two different machines would have confounded interconnect with
        architecture, memory bandwidth and NCCL version to chase a three-point effect.
      </Caveat>

      <div className="rule-heavy" style={{ margin: "var(--space-6) 0 var(--space-4)" }} />
      <div className="closing-cols">
        <p style={{ font: "400 17px/1.6 var(--font-body)" }}>
          <b>A mechanism you can be wrong about is worth more than a number you cannot.</b> The fit
          is two parameters from two points, which is the weakest possible model — and that is what
          makes landing on the other two, across a further fourfold range, mean something. Fitting
          all four would have produced a better-looking curve and no evidence at all.
        </p>
        <p style={{ font: "400 16px/1.6 var(--font-body)", color: "var(--color-neutral-700)" }}>
          The topology is recorded in the results file rather than described:{" "}
          <code>nvidia-smi topo -m</code> shows no NVLink and two NUMA nodes. Full report, including
          why the MFU column is deliberately empty on this hardware, in{" "}
          <a href={`${REPO}/docs/scaling.md`}>
            <code>docs/scaling.md</code>
          </a>
          .
        </p>
      </div>

      <PlateFoot current="scaling" />
    </div>
  );
}
