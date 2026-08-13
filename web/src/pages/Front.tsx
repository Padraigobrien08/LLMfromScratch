import { Fragment, useEffect, useState } from "react";

import DataTable from "../components/DataTable";
import PlateNumeral from "../components/PlateNumeral";
import StackFigure from "../components/StackFigure";
import { frontFigures } from "../content/frontFigures";
import { GROUP_HEADING, pathStops } from "../content/path";
import { ILLUSTRATIVE, STATUS, TAG_CLASS } from "../content/status";
import { fetchAblations } from "../lib/ablationsData";
import { numberWord, numberWordCapitalized } from "../lib/numberWord";

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
  /* One expression for the attention explorer's URL: the path list links to it and so
     does Figure V's attention block, and a second copy is how they would come to
     disagree. */
  const attentionHref = `${import.meta.env.BASE_URL}attention/`;
  const stops = pathStops(attentionHref);

  /**
   * Every pillar is done, so the Status column would print the same tag ten times.
   * Fold it away — but derive the fold, so the day a row goes back to pending the
   * column returns with it rather than hiding the one thing the table exists to say.
   */
  const allDone = STATUS.every((row) => row.status === "done");
  const statusNote = allDone
    ? `All ${numberWord(STATUS.length)} pillars below are done, and every claim inside a row is still pinned by a test.`
    : "The rows still waiting on a run say so, and so does every claim inside a row that is otherwise done.";

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
          This is the part you can click. {numberWordCapitalized(stops.length)} stops, in order,
          from a sentence you type to the question of whether any of the architecture decisions
          were worth making. Every
          number here is either arithmetic you can check or a measurement pinned to the
          repository by a test — and where a run has not happened yet, it says so.
        </p>
      </div>

      {/* Each figure is the door to the page that proves it: the numeral is the link,
          printed in the page's own ink, on the `.path-title` precedent. */}
      <div className="figure-strip">
        {figures.map((figure) => (
          <figure key={figure.label}>
            <a
              className="figure-link"
              href={figure.href}
              aria-label={`${figure.value} ${figure.label}`}
            >
              <PlateNumeral value={figure.value} />
            </a>
            <figcaption>{figure.label}</figcaption>
          </figure>
        ))}
      </div>

      {/* The centrepiece, between the readouts and the path: the four figures above say
          what was built, and this is the thing itself. It carries a fixed height rather
          than a fluid one so the first path row stays above the fold on a laptop — the
          path is how a reader gets anywhere, and burying it would cost more than the
          figure gains. */}
      <StackFigure attentionHref={attentionHref} />

      <div className="rule-heavy" />
      <h2 className="section-label">The path</h2>
      {/* One ordered list, because it is still one path — but broken into its three
          named runs, so a reader sees eight chapters, four plates and three explorers
          rather than sixteen equal rows. A group that prints a heading drops the
          per-row kicker the heading now carries. */}
      <ol className="path">
        {stops.map((stop, i) => {
          const heading =
            stop.group === stops[i - 1]?.group ? undefined : GROUP_HEADING[stop.group];
          return (
            <Fragment key={stop.title}>
              {heading && (
                <li className="path-break">
                  <div className="rule-heavy" />
                  {/* h2, like "The path" above it: these are its sibling sections, not
                      subsections of it — "The path" is the chapters' own heading. */}
                  <h2 className="section-label">{heading}</h2>
                </li>
              )}
              <li className="path-item">
                <span className="path-num">
                  <PlateNumeral value={stop.numeral} />
                </span>
                <div>
                  {!GROUP_HEADING[stop.group] && (
                    <p className="kicker path-kicker">{stop.kicker}</p>
                  )}
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
            </Fragment>
          );
        })}
      </ol>

      <div className="rule-heavy" />
      <h2 className="section-label">Honest status</h2>
      <p className="status-note">
        No result appears anywhere on this site that has not been measured.{" "}
        {statusNote}
      </p>
      <DataTable label="Pillar status" className="status-table">
        <thead>
          <tr>
            <th>Pillar</th>
            <th>State</th>
            {!allDone && <th style={{ width: 130 }}>Status</th>}
          </tr>
        </thead>
        <tbody>
          {STATUS.map((row) => (
            <tr key={row.pillar}>
              <td className="status-pillar">{row.pillar}</td>
              <td className="status-state">{row.state}</td>
              {!allDone && (
                <td>
                  <span className={TAG_CLASS[row.status]}>{row.status}</span>
                </td>
              )}
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
