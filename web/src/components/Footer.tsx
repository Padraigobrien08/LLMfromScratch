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
        One row, three cells: the claim, where the project came from, and where to get it.
        It was three stacked blocks separated by rules, which ran to nearly 200px — on a
        front page written to be read without scrolling, the footer was the thing that
        made it scroll.

        The claim used to read "Nothing here is a number typed by hand." It is the site's
        own honesty rule and it stated it as a riddle: a reader has to work out that the
        subject is provenance, and that "typed by hand" means transcribed rather than
        generated from a run. Saying it plainly costs four words.

        The middle cell is the colophon. A newspaper prints where it came from, and this
        began as a tutorial reproduction of a small character-level GPT — the original
        scripts are preserved unmodified in `legacy/` and share no logic with anything
        above. Worth stating on the site rather than only in the README: the repository's
        About box used to point at that write-up, and moving the link here is what makes
        it safe to point the About box at the finished thing instead.
      */}
      <div className="footer-row">
        <p className="footer-claim">
          Every number here is read from a run or from arithmetic you can check.
        </p>
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
          . Everything above the <code>legacy/</code> directory is a rewrite rather than a
          refactor, and shares no logic with it.
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
