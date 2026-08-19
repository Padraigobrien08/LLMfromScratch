import { PROJECT } from "../content/projectState";

/**
 * Where this came from, and what its numbers are worth.
 *
 * The colophon used to run in the footer of every page: what the project began as, that
 * `legacy/` holds the original scripts unmodified, and the line about every number being
 * read from a run. Three claims in a strip of 12.5px type under a front page written to
 * fit a screen — which is where they were competing with the model for height, and losing.
 *
 * They are claims about the repository rather than about whichever page a reader happens
 * to be on, so they belong on a page of their own and the footer keeps a link to it. The
 * front page gets its height back and the claims get room to be stated properly rather
 * than compressed into a sentence each.
 */
export default function About() {
  return (
    <div className="shell page">
      <p className="kicker">About · where this came from</p>
      <h1 className="page-headline">A rewrite of a tutorial, and what that is worth</h1>
      <p className="page-standfirst">
        This project began as a tutorial reproduction of a small character-level GPT. What is
        here now is not that tutorial — it is a decoder-only transformer written from nothing
        and reproduced at GPT-2 124M scale — but the starting point is worth printing rather
        than quietly leaving behind.
      </p>

      <div className="rule-heavy" />

      <h2 className="section-h2">What it was built from</h2>
      <p className="prose">
        The original is a write-up and a video by their author, and both are worth reading on
        their own terms:{" "}
        <a
          href="https://app.readytensor.ai/publications/building-a-transformer-based-llm-from-scratch-using-pytorch-HMEzasyetWey"
          target="_blank"
          rel="noopener"
        >
          the write-up ↗
        </a>{" "}
        and{" "}
        <a href="https://youtu.be/UU1WVnMk4E8" target="_blank" rel="noopener">
          the video ↗
        </a>
        .
      </p>
      <p className="prose">
        The scripts from that reproduction are preserved unmodified in the repository's{" "}
        <code>legacy/</code> directory. Everything above it is a rewrite rather than a
        refactor, and shares no logic with it — the two can be read side by side, which is
        the only reason to keep the old one at all.
      </p>

      <div className="rule-hair" />

      <h2 className="section-h2">What the numbers here are</h2>
      <p className="prose">
        Every number printed on this site is read from a run or from arithmetic you can check.
        The validation loss, the parameter count, the ablation deltas and the benchmark figures
        are pulled from the JSON a run wrote, not typed into the page — and where a figure is
        derived rather than measured, the page says which arithmetic produced it.
      </p>
      <p className="prose">
        That is a claim, so the site is built to let you audit it rather than take it. The
        results pages name the run behind each plate, the architecture explorer names the test
        that pins each block's parameter count, and the tests page shows what the suite actually
        asserts rather than only how many assertions there are. {PROJECT.pythonTests.toLocaleString()}{" "}
        of them run on every push.
      </p>

      <div className="rule-hair" />

      <h2 className="section-h2">The source</h2>
      <p className="prose">
        The repository is public and MIT licensed:{" "}
        <a href={PROJECT.repo} target="_blank" rel="noopener">
          {PROJECT.repo.replace("https://github.com/", "")} ↗
        </a>
        . The site in <code>web/</code> is part of it, so the page you are reading and the model
        it describes are versioned together.
      </p>
    </div>
  );
}
