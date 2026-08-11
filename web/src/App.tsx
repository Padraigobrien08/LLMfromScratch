import Footer from "./components/Footer";
import Masthead from "./components/Masthead";
import Ablations from "./pages/Ablations";
import Chapter from "./pages/Chapter";
import Efficiency from "./pages/Efficiency";
import Front from "./pages/Front";
import Reproduction from "./pages/Reproduction";
import Rope from "./pages/Rope";
import Scaling from "./pages/Scaling";
import { type Route, useRoute } from "./router";

/**
 * `Legacy` used to live here: a wrapper that re-pointed the pre-redesign token names at
 * Broadsheet's, so `#/ablations` could render in the new ink without being rewritten.
 * It went out with that rewrite, along with the `.legacy-page` block at the foot of
 * `styles.css` — the last of the old design system in the tree.
 */
function Placeholder({ title }: { title: string }) {
  return (
    <div className="shell" style={{ paddingTop: "var(--space-8)" }}>
      <p style={{ font: "400 19px/1.62 var(--font-body)" }}>{title} — not built yet.</p>
    </div>
  );
}

function Page({ route }: { route: Route }) {
  switch (route.kind) {
    case "front":
      return <Front />;
    case "chapter":
      return <Chapter n={route.n} />;
    case "rope":
      return <Rope />;
    case "architecture":
      return <Placeholder title="The architecture page" />;
    case "tests":
      return <Placeholder title="The test-suite page" />;
    case "reproduction":
      return <Reproduction />;
    case "ablations":
      return <Ablations />;
    case "efficiency":
      return <Efficiency />;
    case "scaling":
      return <Scaling />;
  }
}

export default function App() {
  const route = useRoute();

  return (
    <>
      <Masthead route={route} />
      <main>
        <Page route={route} />
      </main>
      <Footer />
    </>
  );
}
