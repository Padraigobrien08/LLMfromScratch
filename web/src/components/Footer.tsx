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
    </footer>
  );
}
