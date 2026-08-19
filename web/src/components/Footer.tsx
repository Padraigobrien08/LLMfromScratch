import { PROJECT } from "../content/projectState";
import { type Route } from "../router";

export default function Footer({ route }: { route: Route }) {
  return (
    /* The front page sets itself wider than an article, and the masthead already follows
       it. A footer left at the reading measure would rule off under two thirds of the
       page it closes — and its three cells would share 1180px while the content above
       spans 2400. */
    <footer
      className={`shell site-footer${route.kind === "front" ? " front-shell" : ""}`}
    >
      <div className="rule-heavy" />
      {/*
        One row, two cells: where the project came from, and where to get it. It was three
        stacked blocks separated by rules, which ran to nearly 200px — on a front page
        written to be read without scrolling, the footer was the thing that made it scroll.

        A newspaper prints where it came from, and this began as a tutorial reproduction of
        a small character-level GPT. Worth stating on the site rather than only in the
        README: the repository's About box used to point at that write-up, and moving the
        link here is what makes it safe to point the About box at the finished thing
        instead.

        What the line no longer says is how the rewrite relates to that tutorial — that
        `legacy/` holds the original scripts unmodified and shares no logic with anything
        above. That is a claim about the repository rather than about the site, and the
        README makes it at length; here it was a second sentence that took the colophon to
        two lines in a footer being asked to fit a window.
      */}
      <div className="footer-row">
        <p className="colophon">
          This project began as a tutorial reproduction of a character-level GPT —{" "}
          <a
            href="https://app.readytensor.ai/publications/building-a-transformer-based-llm-from-scratch-using-pytorch-HMEzasyetWey"
            target="_blank"
            rel="noopener"
          >
            the original write-up ↗
          </a>{" "}
          and{" "}
          <a href="https://youtu.be/UU1WVnMk4E8" target="_blank" rel="noopener">
            video ↗
          </a>
          .
        </p>
        <p className="footer-meta">
          <a href={PROJECT.repo} target="_blank" rel="noopener">
            Source on GitHub ↗
          </a>{" "}
          · {PROJECT.licence}
        </p>
      </div>
    </footer>
  );
}
