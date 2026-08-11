import { useEffect, useMemo, useState } from "react";

import CurvePlot, { type Series } from "../components/CurvePlot";
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
import { fetchAblations } from "../lib/ablationsData";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";

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

type LoadState =
  | { status: "loading" }
  | { status: "absent" }
  | { status: "error"; message: string }
  | { status: "ready"; payload: Payload };

export default function Ablations() {
  const [state, setState] = useState<LoadState>({ status: "loading" });
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    // The load itself lives in lib/ablationsData.ts, because the front page's
    // illustrative preview keys off exactly the same answer.
    fetchAblations()
      .then((payload: Payload | null) =>
        setState(payload ? { status: "ready", payload } : { status: "absent" }),
      )
      .catch((e: unknown) =>
        setState({ status: "error", message: e instanceof Error ? e.message : String(e) }),
      );
  }, []);

  const analysis = useMemo(
    () => (state.status === "ready" ? compare(state.payload) : null),
    [state],
  );
  const byName = useMemo(
    () => (state.status === "ready" ? groupByName(state.payload.arms) : new Map()),
    [state],
  );

  const selection = resolveSelection(selected);
  const toggle = (name: string) =>
    setSelected((s) => (s.includes(name) ? s.filter((x) => x !== name) : [...s, name]));

  const shownArm =
    selection.kind === "arm"
      ? selection.name
      : selection.kind === "combination"
        ? selection.name
        : null;

  const row: Comparison | undefined = analysis?.rows.find((r) => r.name === shownArm);

  const series: Series[] = [];
  if (analysis) {
    const base = meanCurve(byName.get("baseline") ?? []);
    if (base.length) series.push({ label: "baseline", color: "var(--muted)", dashed: true, points: base });
    if (shownArm) {
      const arm = meanCurve(byName.get(shownArm) ?? []);
      if (arm.length) series.push({ label: shownArm, color: "var(--accent)", points: arm });
    }
  }

  return (
    <>
      <p className="eyebrow">Measured results</p>
      <h1>The ablation playground</h1>
      <p className="lede">
        Thirteen arms against a shared baseline. Eleven vary exactly one design decision;{" "}
        <code>modern-stack</code> combines five of them to test whether the individual deltas
        actually add up. Every arm runs at the same three seeds, and every comparison is paired
        against the baseline run that saw its data in the same order.
      </p>

      <div className="card">
        <p className="eyebrow">Change one thing</p>
        <div className="controls" style={{ gap: 8 }}>
          {TOGGLES.map((t) => {
            const on = selected.includes(t.name);
            return (
              <button
                key={t.name}
                onClick={() => toggle(t.name)}
                aria-pressed={on}
                style={{
                  background: on ? "var(--accent)" : "var(--bg)",
                  borderColor: on ? "var(--accent)" : "var(--border)",
                  color: on ? "#fff" : "var(--text)",
                  fontWeight: on ? 550 : 400,
                }}
              >
                {t.label}
              </button>
            );
          })}
          {selected.length > 0 && <button onClick={() => setSelected([])}>Reset</button>}
        </div>
        <p className="small muted" style={{ margin: "10px 0 0" }}>
          {selection.kind === "baseline"
            ? "Nothing changed — this is the baseline: LayerNorm, learned positions, GELU, tied embeddings, bias, 8 KV heads."
            : selection.kind === "arm"
              ? AXIS[selection.name]
              : selection.kind === "combination"
                ? "All five modern components at once — the one combination the sweep actually ran."
                : "More than one axis at a time."}
        </p>
      </div>

      {state.status === "loading" && (
        <div className="card">
          <p className="small muted" style={{ margin: 0 }}>
            Loading results…
          </p>
        </div>
      )}

      {state.status === "error" && (
        <div className="card">
          <p className="small" style={{ margin: 0 }}>
            Could not load results: {state.message}
          </p>
        </div>
      )}

      {state.status === "absent" && (
        <div className="callout warn">
          <h3>The sweep has not published results yet</h3>
          <p className="small" style={{ margin: 0 }}>
            This page reads <code>results/ablations.json</code>, which CI copies into the site when
            it exists. Until the GPU run finishes there is nothing measured to show, and inventing a
            placeholder table would defeat the point of the study. The controls above are live —
            they just have no data behind them yet.
          </p>
        </div>
      )}

      {selection.kind === "unmeasured" && state.status === "ready" && (
        <div className="callout warn">
          <h3>That combination was never measured</h3>
          <p className="small" style={{ margin: 0 }}>
            The sweep varies <b>one</b> axis at a time, so there is no run for{" "}
            {selection.names.map((n) => AXIS[n] ?? n).join(" + ")}. Predicting it by adding the
            individual deltas assumes the components do not interact — which is an assumption, not a
            result. The one combination that <i>was</i> run is{" "}
            <button
              style={{ padding: "2px 8px", fontSize: 13 }}
              onClick={() => setSelected([...MODERN_STACK])}
            >
              modern-stack
            </button>
            , and it exists precisely to check whether that addition holds.
          </p>
        </div>
      )}

      {state.status === "ready" && analysis && (
        <>
          <div className="grid2">
            <div className="card">
              <p className="eyebrow">
                {shownArm ? `Δ validation loss vs baseline` : "Baseline validation loss"}
              </p>
              {shownArm && row?.delta != null ? (
                <>
                  <div
                    className="readout"
                    style={{ color: row.significant ? (row.delta < 0 ? "var(--good)" : "var(--accent-2)") : "var(--muted)" }}
                  >
                    {row.delta >= 0 ? "+" : ""}
                    {row.delta.toFixed(4)}
                    {row.halfRange > 0 && (
                      <span style={{ fontSize: 17, color: "var(--muted)" }}> ± {row.halfRange.toFixed(4)}</span>
                    )}
                  </div>
                  <p className="small" style={{ margin: "6px 0 0" }}>
                    {row.significant ? (
                      <b style={{ color: row.delta < 0 ? "var(--good)" : "var(--accent-2)" }}>
                        Every seed agreed on the direction.
                      </b>
                    ) : (
                      <b className="muted">Not a result — the per-seed deltas straddle zero.</b>
                    )}{" "}
                    <span className="muted">
                      {row.paired ? "Paired" : "Unpaired"} over {row.nSeeds}{" "}
                      {row.nSeeds === 1 ? "seed" : "seeds"}
                      {row.deltas.length > 0 &&
                        `: ${row.deltas.map((d) => d.toFixed(4)).join(", ")}`}
                      .
                    </span>
                  </p>
                </>
              ) : (
                <div className="readout">
                  {analysis.baseline.mean !== null ? analysis.baseline.mean.toFixed(4) : "—"}
                </div>
              )}
            </div>

            <div className="card">
              <p className="eyebrow">Noise floor</p>
              <div className="readout">{analysis.baseline.spread.toFixed(4)}</div>
              <p className="small muted" style={{ margin: "6px 0 0" }}>
                Full range of the baseline across {analysis.baseline.n} seeds — the spread any
                unpaired claim would have to clear. Pairing is what lets an effect smaller than this
                still be visible, and it is the reason every arm repeats the same seeds instead of
                only the baseline.
              </p>
            </div>
          </div>

          <div className="card">
            <CurvePlot series={series} />
          </div>

          <h2>Every arm</h2>
          <div className="card">
            <table>
              <thead>
                <tr>
                  <th>Arm</th>
                  <th>Axis</th>
                  <th style={{ textAlign: "right" }}>Val loss</th>
                  <th style={{ textAlign: "right" }}>Δ</th>
                  <th>Verdict</th>
                </tr>
              </thead>
              <tbody>
                {analysis.rows.map((r) => (
                  <tr key={r.name}>
                    <td>
                      <code>{r.name}</code>
                    </td>
                    <td className="muted small">{r.axis}</td>
                    <td className="num">{r.valLoss?.toFixed(4) ?? "—"}</td>
                    <td className="num">
                      {r.delta == null
                        ? "—"
                        : `${r.delta >= 0 ? "+" : ""}${r.delta.toFixed(4)}`}
                    </td>
                    <td className="small">
                      {r.status !== "completed" ? (
                        <span style={{ color: "var(--warn)" }}>{r.status}</span>
                      ) : r.name === "baseline" ? (
                        <span className="muted">reference</span>
                      ) : r.significant ? (
                        <span style={{ color: r.delta! < 0 ? "var(--good)" : "var(--accent-2)" }}>
                          {r.delta! < 0 ? "better" : "worse"}
                        </span>
                      ) : (
                        <span className="muted">within noise</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h2>How to read this</h2>
      <p>
        An arm counts as a result only when the range of its per-seed deltas does not straddle zero
        — every seed agreed on the direction. That is a deliberately blunt rule rather than a
        p-value: with three seeds nothing stronger would be honest, and it is exactly what the error
        bars show. An ablation table without such a check is worse than no table, because it reads
        as authoritative while recommending changes that do nothing.
      </p>
      <p className="small muted">
        Analysis logic mirrors{" "}
        <a href={`${REPO}/src/llmfs/ablation/report.py`}>src/llmfs/ablation/report.py</a>; the arms
        are defined in <a href={`${REPO}/configs/ablations`}>configs/ablations/</a>, where a test
        asserts each one differs from the shared baseline in its own named axis and nothing else.
      </p>
    </>
  );
}
