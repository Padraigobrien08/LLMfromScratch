import { useMemo, useState } from "react";

import DataTable from "../components/DataTable";
import CurvePlot, { type Series } from "../components/CurvePlot";
import SeedDeltas from "../components/SeedDeltas";
import PlateNumeral from "../components/PlateNumeral";
import PlateFoot from "../components/PlateFoot";
import Caveat from "../components/Caveat";
import { MEASURED } from "../content/measured";
import { plateKicker } from "../content/path";
import {
  AXIS,
  MODERN_STACK,
  type Comparison,
  type Payload,
  compare,
  groupByName,
  meanCurve,
  resolveSelection,
} from "../lib/ablations";
import { useResult } from "../lib/resultsData";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";
const A = MEASURED.ablations;

/** The single-axis arms, in the order a reader would want to think about them. */
const TOGGLES: Array<{ name: string; label: string }> = [
  { name: "norm-rmsnorm", label: "RMSNorm" },
  { name: "pos-rope", label: "RoPE" },
  { name: "pos-none", label: "No positions" },
  { name: "mlp-swiglu", label: "SwiGLU" },
  { name: "gqa-2", label: "GQA (2 KV heads)" },
  { name: "no-bias", label: "No bias" },
  { name: "untied-embeddings", label: "Untied embeddings" },
  { name: "sched-wsd", label: "WSD schedule" },
  { name: "wd-zero", label: "No weight decay" },
  { name: "lr-3e-4", label: "lr 3e-4" },
  { name: "lr-3e-3", label: "lr 3e-3" },
];

export default function Ablations() {
  const state = useResult<Payload>("ablations.json");
  const [selected, setSelected] = useState<string[]>([]);

  const payload = state.status === "ready" ? state.data : null;
  const analysis = useMemo(() => (payload ? compare(payload) : null), [payload]);
  const byName = useMemo(
    () => (payload ? groupByName(payload.arms) : new Map()),
    [payload],
  );

  const selection = resolveSelection(selected);
  const toggle = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));

  const shownArm =
    selection.kind === "arm" || selection.kind === "combination" ? selection.name : null;
  const row: Comparison | undefined = analysis?.rows.find((r) => r.name === shownArm);

  const series: Series[] = [];
  if (analysis) {
    const base = meanCurve(byName.get("baseline") ?? []);
    if (base.length) {
      series.push({ label: "baseline", color: "var(--color-neutral-500)", dashed: true, points: base });
    }
    if (shownArm) {
      const arm = meanCurve(byName.get(shownArm) ?? []);
      if (arm.length) series.push({ label: shownArm, color: "var(--color-accent)", points: arm });
    }
  }

  // One shared scale for the seed strip, so switching arms moves the ticks rather than
  // rescaling the axis under them. Comparing two arms is the whole point of the toggles,
  // and an axis that silently rescales makes every arm look the same size.
  const scale = useMemo(() => {
    const worst = Math.max(0.02, ...(analysis?.rows ?? []).map((r) => Math.abs(r.delta ?? 0)));
    return Math.min(worst * 1.15, 0.2);
  }, [analysis]);

  return (
    <div className="shell page">
      <p className="kicker">{plateKicker("ablations")}</p>
      <h1 className="page-headline">What actually matters, and what only sounds like it does</h1>
      <p className="page-standfirst">
        {A.arms} design decisions, each varied against a shared baseline, each run at the same{" "}
        {A.seeds} seeds. The finding is not the one the architecture literature would lead you to
        expect: <b>the optimiser dominates the architecture</b>. Learning rate and schedule move
        validation loss more than every architecture change in the study combined.
      </p>

      {/* The plate's opening trio, in the grammar the other three plates use. The cost of
          the study, and the number every delta on this page has to clear to mean anything
          — which the prose below refers to repeatedly and nothing was printing. */}
      <div className="figure-strip">
        <figure>
          <PlateNumeral value={String(A.runs)} />
          <figcaption>
            runs, {A.arms} arms and their baseline at {A.seeds} seeds each
          </figcaption>
        </figure>
        <figure>
          <PlateNumeral value={A.gpuHours.toFixed(1)} />
          <figcaption>GPU-hours, the whole sweep</figcaption>
        </figure>
        <figure>
          <PlateNumeral value={A.noiseFloor.toFixed(4)} />
          <figcaption>
            the noise floor: seed-to-seed spread on the baseline, and the bar every delta here
            has to clear
          </figcaption>
        </figure>
      </div>

      <div className="rule-heavy" />
      <h2 className="section-h2">Change one thing</h2>
      <div className="figure-panel">
        <div className="fig-row fig-row-wide" style={{ marginBottom: "var(--space-2)" }}>
          {TOGGLES.map((t) => {
            const on = selected.includes(t.name);
            return (
              <button
                key={t.name}
                className={on ? "btn btn-primary btn-sm" : "btn btn-secondary btn-sm"}
                onClick={() => toggle(t.name)}
                aria-pressed={on}
              >
                {t.label}
              </button>
            );
          })}
          {selected.length > 0 && (
            <button className="btn btn-ghost btn-sm" onClick={() => setSelected([])}>
              Reset
            </button>
          )}
        </div>
        <p className="fig-note">
          {selection.kind === "baseline"
            ? "Nothing changed — this is the baseline: LayerNorm, learned positions, GELU, tied embeddings, bias, 8 KV heads."
            : selection.kind === "arm"
              ? AXIS[selection.name]
              : selection.kind === "combination"
                ? "All five modern components at once — the one combination the sweep actually ran."
                : "More than one axis at a time."}
        </p>

        {state.status === "loading" && <p className="fig-note">Loading the sweep…</p>}
        {state.status === "error" && (
          <p className="fig-note">Could not load results: {state.message}</p>
        )}
        {state.status === "absent" && (
          <p className="fig-note">
            This page reads <code>results/ablations.json</code>, which CI copies into the site when
            it exists. Until the sweep has run there is nothing measured to show, and inventing a
            placeholder table would defeat the point of the study.
          </p>
        )}

        {selection.kind === "unmeasured" && payload && (
          <Caveat style={{ marginTop: "var(--space-3)" }}>
            <b>That combination was never measured.</b> The sweep varies <i>one</i> axis at a time,
            so there is no run for {selection.names.map((n) => AXIS[n] ?? n).join(" + ")}. Predicting
            it by adding the individual deltas assumes the components do not interact — which is an
            assumption, not a result. The one combination that <i>was</i> run is{" "}
            <button className="btn btn-ghost btn-sm" onClick={() => setSelected([...MODERN_STACK])}>
              modern-stack
            </button>
            , and it exists precisely to check whether that addition holds.
          </Caveat>
        )}

        {analysis && (
          <div className="readouts" style={{ marginTop: "var(--space-3)" }}>
            <div>
              <p className="eyebrow">
                {shownArm ? "Δ validation loss vs baseline" : "Baseline validation loss"}
              </p>
              {shownArm && row?.delta != null ? (
                <>
                  <div
                    className="readout-xl"
                    style={{
                      color: row.significant
                        ? row.delta < 0
                          ? "var(--color-accent-700)"
                          : "var(--color-accent-2-700)"
                        : "var(--color-neutral-700)",
                    }}
                  >
                    {row.delta >= 0 ? "+" : ""}
                    {row.delta.toFixed(4)}
                    {row.halfRange > 0 && (
                      <span style={{ fontSize: 18, color: "var(--color-neutral-700)" }}>
                        {" "}
                        ± {row.halfRange.toFixed(4)}
                      </span>
                    )}
                  </div>
                  <p className="tracker-sentence">
                    {row.significant ? (
                      <b>Every seed agreed on the direction.</b>
                    ) : (
                      <b>Not a result — the per-seed deltas straddle zero.</b>
                    )}{" "}
                    {row.paired ? "Paired" : "Unpaired"} over {row.nSeeds}{" "}
                    {row.nSeeds === 1 ? "seed" : "seeds"}
                    {row.deltas.length > 0 && `: ${row.deltas.map((d) => d.toFixed(4)).join(", ")}`}.
                  </p>
                </>
              ) : (
                <>
                  <div className="readout-xl">
                    {analysis.baseline.mean !== null ? analysis.baseline.mean.toFixed(4) : "—"}
                  </div>
                  <p className="tracker-sentence">
                    Pick an axis above to difference an arm against it. Every arm is compared to the
                    baseline run that saw its data <i>in the same order</i>, which is what cancels
                    the batch-ordering noise the two share.
                  </p>
                </>
              )}
            </div>

            <div>
              <p className="eyebrow">Noise floor across {analysis.baseline.n} baseline seeds</p>
              <div className="readout-xl">{analysis.baseline.spread.toFixed(4)}</div>
              <p className="tracker-sentence">
                The full range two runs differing <i>only</i> in seed can occupy — the spread any
                unpaired claim would first have to clear. Most architecture effects in this study are
                smaller than it, which is why the comparisons are paired and why an unpaired table
                would have found almost nothing.
              </p>
            </div>
          </div>
        )}
      </div>

      {row && row.deltas.length > 0 && (
        <div className="figure-panel" style={{ marginTop: "var(--space-4)" }}>
          <h3 className="figure-title">The verdict is whether these three ticks straddle zero</h3>
          <p className="fig-note" style={{ margin: "0 0 var(--space-2)" }}>
            One tick per seed, on a shared axis so switching arms moves the ticks rather than
            rescaling under them. The dot is the mean; the bar is the range.
          </p>
          <SeedDeltas row={row} scale={scale} />
          <p className="fig-note">
            {row.significant
              ? `All ${row.nSeeds} seeds fell on the same side of zero, so ${row.name} counts as a result.`
              : `The seeds disagree about the sign, so ${row.name} is reported as no effect rather than as a small one.`}
          </p>
        </div>
      )}

      {analysis && series.length > 0 && (
        <div className="figure-panel" style={{ marginTop: "var(--space-4)" }}>
          <h3 className="figure-title">Validation loss through the run, averaged over seeds</h3>
          <CurvePlot series={series} />
        </div>
      )}

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">Every arm</h2>
      {analysis && (
        <DataTable label="Every ablation arm">
          <thead>
            <tr>
              <th>Arm</th>
              <th>Axis</th>
              <th className="num">Val loss</th>
              <th className="num">Δ</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {analysis.rows.map((r) => (
              <tr key={r.name}>
                <td className="mono">{r.name}</td>
                <td className="status-state">{r.axis}</td>
                <td className="num mono">{r.valLoss?.toFixed(4) ?? "—"}</td>
                <td className="num mono">
                  {r.delta == null ? "—" : `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(4)}`}
                </td>
                <td>
                  {r.status !== "completed" ? (
                    <span className="tag tag-accent-2">{r.status}</span>
                  ) : r.name === "baseline" ? (
                    <span className="tag tag-neutral">reference</span>
                  ) : r.significant ? (
                    <span className={r.delta! < 0 ? "verdict-better" : "verdict-noise"}>
                      {r.delta! < 0 ? "better" : "worse"}
                    </span>
                  ) : (
                    <span className="verdict-noise">within noise</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </DataTable>
      )}

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">Three things worth pulling out</h2>
      <div className="two-col">
        <p className="prose">
          <b>The optimiser dominates the architecture.</b> Learning rate and schedule move loss more
          than every architecture change combined. RMSNorm versus LayerNorm is worth less than the
          noise floor; the learning rate is worth two orders of magnitude more. A study that varied
          only architecture would have concluded that nothing matters.
        </p>
        <p className="prose">
          <b>The components are additive.</b> Summing the five individual modern-stack parts predicts
          almost exactly what the combined arm measured — within a third of the noise floor — so they
          compose without interacting. That is a real finding and it is also why the playground
          refuses to add deltas for any <i>other</i> combination: this one was checked, and the rest
          were not.
        </p>
      </div>
      <Caveat>
        <b>The study's largest is a prediction that was wrong.</b> The <code>lr-3e-3</code>{" "}
        arm was expected to diverge. It won — which means every other arm was measured at a learning
        rate now known to be suboptimal. That does not invalidate the paired comparisons, since every
        arm shares the same baseline, but it does mean the absolute losses are all worse than they
        needed to be, and it is stated as such in the write-up rather than left for a reader to
        notice.
      </Caveat>

      <div className="rule-heavy" style={{ margin: "var(--space-6) 0 var(--space-4)" }} />
      <div className="closing-cols">
        <p style={{ font: "400 17px/1.6 var(--font-body)" }}>
          <b>An arm counts as a result only when its per-seed deltas do not straddle zero.</b> A
          deliberately blunt rule rather than a p-value: with three seeds nothing stronger would be
          honest, and it is exactly what the strip above draws. An ablation table without such a
          check is worse than no table, because it reads as authoritative while recommending changes
          that do nothing.
        </p>
        <p style={{ font: "400 16px/1.6 var(--font-body)", color: "var(--color-neutral-700)" }}>
          The analysis here mirrors{" "}
          <a href={`${REPO}/src/llmfs/ablation/report.py`}>
            <code>src/llmfs/ablation/report.py</code>
          </a>
          , so the page cannot report something the repository's own report would refuse to. The arms
          live in <a href={`${REPO}/configs/ablations`}>configs/ablations/</a>, where a test asserts
          each one differs from the shared baseline in its own named axis and nothing else.
        </p>
      </div>

      <PlateFoot current="ablations" />
    </div>
  );
}
