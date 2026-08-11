import { PROJECT } from "../content/projectState";

export default function Footer() {
  return (
    <footer className="shell site-footer">
      <div className="rule-heavy" />
      <div className="footer-row">
        <span>
          Everything here is arithmetic you can check or a measurement pinned to the repository
          by a test.
        </span>
        <span>
          <a href={PROJECT.repo} target="_blank" rel="noopener">
            Source on GitHub ↗
          </a>{" "}
          · {PROJECT.licence}
        </span>
      </div>

      {/*
        The colophon. A newspaper prints where it came from, and this began as a tutorial
        reproduction of a small character-level GPT — the original scripts are preserved
        unmodified in `legacy/` and share no logic with anything above. Worth stating on
        the site rather than only in the README: the repository's About box used to point
        at that write-up, and moving the link here is what makes it safe to point the
        About box at the finished thing instead.
      */}
      <div className="rule-hair" />
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
    </footer>
  );
}
