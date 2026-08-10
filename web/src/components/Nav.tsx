const LINKS = [
  { href: "#/", label: "Overview", match: "" },
  { href: "#/explainer", label: "How it works", match: "explainer" },
  { href: "#/rope", label: "RoPE explorer", match: "rope" },
  { href: "#/ablations", label: "Ablations", match: "ablations" },
];

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch";

export default function Nav({ route }: { route: string }) {
  return (
    <nav className="nav">
      <div className="nav-inner">
        <a className="brand" href="#/">
          LLMfromScratch
        </a>
        <div className="nav-links">
          {LINKS.map((l) => (
            <a key={l.href} href={l.href} className={route === l.match ? "active" : ""}>
              {l.label}
            </a>
          ))}
          {/* Not a route: the attention explorer is a separate self-contained page,
              built by CI from a checkpoint CI trains. Kept standalone on purpose.
              Built from BASE_URL rather than written relative, so it resolves the
              same whether or not the current URL carries a trailing slash. */}
          <a href={`${import.meta.env.BASE_URL}attention/`}>Attention ↗</a>
          <a href={REPO}>GitHub ↗</a>
        </div>
      </div>
    </nav>
  );
}
