import { useEffect, useState } from "react";

import DataTable from "../components/DataTable";
import PlateNumeral from "../components/PlateNumeral";
import { frontFigures } from "../content/frontFigures";
import { pathStops } from "../content/path";
import { ILLUSTRATIVE, STATUS, TAG_CLASS } from "../content/status";
import { fetchAblations } from "../lib/ablationsData";

export default function Front() {
  /**
   * The illustrative block below is invented numbers, and exists only to show what
   * the ablation page would look like before the sweep had run. It renders while
   * `results/ablations.json` is absent and disappears by itself once CI publishes
   * it — and on any error it stays hidden, because the one outcome worth ruling out
   * is invented figures appearing beside measured ones.
   */
  const [sweepPublished, setSweepPublished] = useState(true);
  useEffect(() => {
    let live = true;
    fetchAblations().then(
      (payload) => live && setSweepPublished(payload !== null),
      () => live && setSweepPublished(true),
    );
    return () => {
      live = false;
    };
  }, []);

  const figures = frontFigures();
  const stops = pathStops(`${import.meta.env.BASE_URL}attention/`);

  return (
    <div className="shell page">
      <p className="kicker kicker-2">Start here · no prior knowledge assumed</p>
      <h1 className="front-headline">
        A language model built by hand, and the evidence that it works
      </h1>

      <div className="standfirst-grid">
        <p className="standfirst">
          A decoder-only transformer written from nothing — rotary embeddings, RMSNorm, SwiGLU,
          grouped-query attention, a static KV cache — with a GPT-2 124M reproduction, a
          paired-seed ablation study and efficiency benchmarks behind it.
        </p>
        <p className="standfirst-secondary">
          This is the part you can click. Thirteen stops, in order, from a sentence you type to
          the question of whether any of the architecture decisions were worth making. Every
          number here is either arithmetic you can check or a measurement pinned to the
          repository by a test — and where a run has not happened yet, it says so.
        </p>
      </div>

      <div className="figure-strip">
        {figures.map((figure) => (
          <figure key={figure.label}>
            <PlateNumeral value={figure.value} />
            <figcaption>{figure.label}</figcaption>
          </figure>
        ))}
      </div>

      <div className="rule-heavy" />
      <h2 className="section-label">The path</h2>
      <ol className="path">
        {stops.map((stop) => (
          <li className="path-item" key={stop.title}>
            <span className="path-num">
              <PlateNumeral value={stop.numeral} />
            </span>
            <div>
              <p className="kicker path-kicker">{stop.kicker}</p>
              <h3 className="path-title">
                <a
                  href={stop.href}
                  {...(stop.external ? { target: "_blank", rel: "noopener" } : {})}
                >
                  {stop.title}
                </a>
              </h3>
              <p className="path-blurb">{stop.blurb}</p>
            </div>
            <a
              className="path-cta"
              href={stop.href}
              {...(stop.external ? { target: "_blank", rel: "noopener" } : {})}
            >
              {stop.cta}
            </a>
          </li>
        ))}
      </ol>

      <div className="rule-heavy" />
      <h2 className="section-label">Honest status</h2>
      <p className="status-note">
        No result appears anywhere on this site that has not been measured. The one row still
        waiting on a run says so, and so does every claim inside a row that is otherwise done.
      </p>
      <DataTable label="Pillar status" className="status-table">
        <thead>
          <tr>
            <th>Pillar</th>
            <th>State</th>
            <th style={{ width: 130 }}>Status</th>
          </tr>
        </thead>
        <tbody>
          {STATUS.map((row) => (
            <tr key={row.pillar}>
              <td className="status-pillar">{row.pillar}</td>
              <td className="status-state">{row.state}</td>
              <td>
                <span className={TAG_CLASS[row.status]}>{row.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </DataTable>

      {!sweepPublished && (
        <div className="illustrative">
          <p className="kicker kicker-2">Illustrative — not measured</p>
          <h3 className="illustrative-title">What the ablation page will say when the sweep lands</h3>
          <p className="illustrative-note">
            Shape of the finished thing, with invented numbers, so the layout can be judged
            before the GPU bill. Nothing in this block is a result.
          </p>
          <DataTable label="Illustrative ablation preview">
            <thead>
              <tr>
                <th>Arm</th>
                <th>Axis</th>
                <th style={{ textAlign: "right" }}>Δ val loss</th>
                <th>Verdict</th>
              </tr>
            </thead>
            <tbody>
              {ILLUSTRATIVE.map((row) => (
                <tr key={row.arm}>
                  <td className="mono">{row.arm}</td>
                  <td className="status-state">{row.axis}</td>
                  <td className="mono num">{row.delta}</td>
                  <td className={row.better ? "verdict-better" : "verdict-noise"}>{row.verdict}</td>
                </tr>
              ))}
            </tbody>
          </DataTable>
        </div>
      )}
    </div>
  );
}
