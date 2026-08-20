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
        One line: where to read about the project, and where to get it.

        The colophon that used to sit here — what this began as, and that `legacy/` holds
        the original scripts unmodified — is on `#/about` now. Those are claims about the
        repository rather than about whichever page a reader is on, and printing them under
        every page cost the front page the height it needed for the model. A link is the
        honest form of a colophon anyway: the claim is one click away and stated properly
        there, rather than compressed into a sentence nobody has room to finish.
      */}
      <div className="footer-row">
        <p className="colophon">
          <a href="#/about">Where this came from</a>
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
