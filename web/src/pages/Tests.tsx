import { Fragment } from "react";

import PlateNumeral from "../components/PlateNumeral";
import { PROJECT } from "../content/projectState";
import { TEST_SHOWCASE } from "../content/testShowcase";
import { claimHeadline } from "../lib/claim";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";

/** `tests/test_kv_cache.py` → `kv cache`, which is what a reader is scanning for. */
const area = (file: string) =>
  file.replace(/^tests\/test_/, "").replace(/\.py$/, "").replace(/_/g, " ");

export default function Tests() {
  const cases = TEST_SHOWCASE.reduce((n, row) => n + row.cases, 0);

  return (
    <div className="shell page">
      <p className="kicker">The deep end · what the suite asserts</p>
      <h1 className="page-headline">A test count is not evidence. This is what they check.</h1>
      <p className="page-standfirst">
        {PROJECT.pythonTests.toLocaleString()} tests pass on every push, and that sentence is worth
        almost nothing on its own — it is a claim about effort, and no reader can check it. What is
        worth showing is a handful of the tests themselves: what each one asserts, and the specific
        bug it exists to catch. Most of these describe mistakes that would otherwise survive all
        the way into a training run and surface only as a mysteriously worse loss.
      </p>

      <div className="figure-strip">
        <figure>
          <PlateNumeral value={String(TEST_SHOWCASE.length)} />
          <figcaption>claims shown here, marked in the suite for this page</figcaption>
        </figure>
        <figure>
          <PlateNumeral value={String(cases)} />
          <figcaption>parametrised runs behind them, across {PROJECT.archVariants} architecture variants</figcaption>
        </figure>
      </div>
      {/* The suite total is deliberately not a third numeral here. This page's headline
          disowns the test count as evidence, and printing it at 58px alongside the two
          figures that are evidence made the layout argue the opposite. It is still on the
          page twice — the standfirst opens on it, and the closing pair subtracts it — which
          is what "declines to lead with" can honestly mean. */}

      <div className="rule-heavy" />

      {/* Grouped by the file the claims came from, so the area is stated once instead of
          once per row — and the headline starts at the claim's own subject, because
          twelve headlines that all began "It asserts that" were twelve headlines whose
          first three words carried nothing. The page's own headline says it once. */}
      <ol className="claims">
        {TEST_SHOWCASE.map((row, i) => {
          const heading =
            row.file === TEST_SHOWCASE[i - 1]?.file ? undefined : area(row.file);
          return (
            <Fragment key={`${row.file}::${row.name}`}>
              {heading && (
                <li className="claim-break">
                  <div className="rule-hair" />
                  <h2 className="section-label">{heading}</h2>
                </li>
              )}
              <li className="claim">
                <h3 className="claim-pins">{claimHeadline(row.pins)}</h3>
                <p className="claim-why">{row.why}</p>
                <p className="claim-source mono">
                  <a href={`${REPO}/${row.file}`}>
                    {row.file}::{row.name}
                  </a>
                  {row.cases > 1 && (
                    <span className="tag tag-neutral claim-cases">{row.cases} cases</span>
                  )}
                </p>
              </li>
            </Fragment>
          );
        })}
      </ol>

      <div className="rule-heavy" style={{ margin: "var(--space-6) 0 var(--space-4)" }} />
      <div className="closing-cols">
        <p className="closing-lead">
          <b>These rows are collected, not typed.</b> Each test above carries a{" "}
          <code>@pytest.mark.showcase(pins=…, why=…)</code>, and the site's export runs pytest's own
          collection to gather them. Rename one and the committed export stops matching; delete one
          and its row disappears.
        </p>
        <p className="closing-note">
          Curation stays deliberate: only marked tests appear, so this is a chosen argument rather
          than a directory listing. The other{" "}
          {(PROJECT.pythonTests - cases).toLocaleString()} cases are config validation, end-to-end
          training, serialisation round-trips and the rest of the ordinary work — necessary, and
          not what anyone came here to read.
        </p>
      </div>
    </div>
  );
}
