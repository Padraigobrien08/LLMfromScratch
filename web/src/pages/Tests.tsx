import { Fragment } from "react";

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
          <span className="cmyk-num plate-num">
            <span className="paper">{TEST_SHOWCASE.length}</span>
            <span className="plate plate-c" aria-hidden="true">{TEST_SHOWCASE.length}</span>
            <span className="plate plate-m" aria-hidden="true">{TEST_SHOWCASE.length}</span>
            <span className="plate plate-y" aria-hidden="true">{TEST_SHOWCASE.length}</span>
          </span>
          <figcaption>claims shown here, chosen rather than listed</figcaption>
        </figure>
        <figure>
          <span className="cmyk-num plate-num">
            <span className="paper">{cases}</span>
            <span className="plate plate-c" aria-hidden="true">{cases}</span>
            <span className="plate plate-m" aria-hidden="true">{cases}</span>
            <span className="plate plate-y" aria-hidden="true">{cases}</span>
          </span>
          <figcaption>parametrised runs behind them, across {PROJECT.archVariants} architecture variants</figcaption>
        </figure>
        <figure>
          <span className="cmyk-num plate-num">
            <span className="paper">{PROJECT.pythonTests}</span>
            <span className="plate plate-c" aria-hidden="true">{PROJECT.pythonTests}</span>
            <span className="plate plate-m" aria-hidden="true">{PROJECT.pythonTests}</span>
            <span className="plate plate-y" aria-hidden="true">{PROJECT.pythonTests}</span>
          </span>
          <figcaption>in the suite in total, the number this page declines to lead with</figcaption>
        </figure>
      </div>

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
        <p style={{ font: "400 17px/1.6 var(--font-body)" }}>
          <b>These rows are collected, not typed.</b> Each test above carries a{" "}
          <code>@pytest.mark.showcase(pins=…, why=…)</code>, and the site's export runs pytest's own
          collection to gather them. Rename one and the committed export stops matching; delete one
          and its row disappears.
        </p>
        <p style={{ font: "400 16px/1.6 var(--font-body)", color: "var(--color-neutral-700)" }}>
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
