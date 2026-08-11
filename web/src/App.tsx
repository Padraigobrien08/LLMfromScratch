import Footer from "./components/Footer";
import Masthead from "./components/Masthead";
import Ablations from "./pages/Ablations";
import Chapter from "./pages/Chapter";
import Front from "./pages/Front";
import Rope from "./pages/Rope";
import { type Route, useRoute } from "./router";

/**
 * Pages still wearing the pre-redesign layout are wrapped in `.legacy-page`, which
 * re-points the old token names at Broadsheet's. They render in the new ink without
 * being edited — which is what keeps `#/ablations` genuinely untouched while the
 * sweep's page waits its turn.
 */
function Legacy({ children }: { children: React.ReactNode }) {
  return <div className="legacy-page">{children}</div>;
}

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
    case "ablations":
      return (
        <Legacy>
          <Ablations />
        </Legacy>
      );
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
