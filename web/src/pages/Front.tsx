import StackFigure from "../components/StackFigure";
import { frontFigures } from "../content/frontFigures";
import { destinations } from "../content/path";

/**
 * The front page: a claim, three doors, and the model itself.
 *
 * Set as a front page rather than as an article — banner across the top, then three
 * columns under it: where to go, the thing that was built, and what the thing is made
 * of. The reader is choosing here, not reading, so nothing on this page is longer than
 * a line except the figure's own panel, which is the one place they have asked a
 * question by clicking something.
 *
 * The four project figures used to run as a strip beneath the standfirst. They are
 * already on the dateline rail above — `346 tests green`, `GPT-2 124M · val loss 3.05`,
 * `ablation sweep complete · 39 runs` — printed on every page of the site from the same
 * generated module, so the strip was the same four numbers a second time, in larger
 * type, one scroll lower.
 */
export default function Front() {
  const figures = frontFigures();
  const routes = destinations();
  /* One expression for the attention explorer's URL: the deep-end row names it and so
     does Figure V's attention block, and a second copy is how they would come to
     disagree. */
  const attentionHref = `${import.meta.env.BASE_URL}attention/`;

  return (
    <div className="shell front-shell page">
      {/* Three children, three columns, and the DOM order is the phone's order. Wide:
          routes left, figure centre, and the figure's own panel becomes the third
          column. Narrow: one column, and a reader gets claim → model → where next,
          which is the order the page argues in. */}
      <div className="front-grid">
        <header className="front-lede">
          <h1 className="front-headline">
            A language model built by hand, and the evidence it works
          </h1>
          <p className="standfirst front-standfirst">
            A decoder-only transformer, reproduced at GPT-2 124M scale and checked against a
            public benchmark, with every design choice tested by ablation.{" "}
            <em>Every number here is measured.</em>
          </p>

          {/* The evidence, at reading size, under the claim it supports. Each figure links
              to the page that proves it — that pairing is the whole point of printing them
              here rather than leaving them to the dateline rail, where they are furniture. */}
          <ul className="front-metrics">
            {figures.map((figure) => (
              <li key={figure.label}>
                <a href={figure.href}>
                  <span className="front-metric-value mono">{figure.value}</span>{" "}
                  <span className="front-metric-label">{figure.label}</span>
                </a>
              </li>
            ))}
          </ul>
        </header>

        {/* The centrepiece. Its detail panel is the page's right-hand column — the figure
            owns that split, so the page does not try to place the panel itself.
            Set before the routes in the markup and after them in the grid: on a phone a
            reader should meet the model before being asked to choose, and on a wide screen
            the choices belong in the left-hand column. Named grid areas are what let those
            two orders differ. */}
        <div className="front-figure">
          <StackFigure attentionHref={attentionHref} />
        </div>

        <nav className="front-routes" aria-label="Where to go next">
          <h2 className="section-label">Where do you want to go?</h2>
          <ol className="destinations">
            {routes.map((route) => (
              <li className="destination" key={route.numeral}>
                <p className="destination-num" aria-hidden="true">
                  {route.numeral}
                </p>
                <h3 className="destination-title">
                  <a href={route.href}>{route.title}</a>
                </h3>
                <p className="destination-blurb">{route.blurb}</p>
                {/* A button rather than a link in a rule of type: this column is the
                    page's one set of choices, and a choice should look pressable. */}
                <a className="destination-cta" href={route.href}>
                  {route.cta}
                </a>
              </li>
            ))}
          </ol>
        </nav>
      </div>
    </div>
  );
}
