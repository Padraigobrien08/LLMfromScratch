import Footer from "./components/Footer";
import Masthead from "./components/Masthead";
import Ablations from "./pages/Ablations";
import Architecture from "./pages/Architecture";
import Chapter from "./pages/Chapter";
import Efficiency from "./pages/Efficiency";
import Front from "./pages/Front";
import Reproduction from "./pages/Reproduction";
import Rope from "./pages/Rope";
import Scaling from "./pages/Scaling";
import Tests from "./pages/Tests";
import { type Route, useRoute } from "./router";

/**
 * Two things used to live here and are worth recording as gone.
 *
 * `Legacy` re-pointed the pre-redesign token names at Broadsheet's so `#/ablations`
 * could render in the new ink without being rewritten; it went out with that rewrite.
 * `Placeholder` kept the router honest while `#/architecture` and `#/tests` were
 * outstanding — every route resolves to a real page now, so the switch below is
 * exhaustive over `Route` with nothing standing in for anything.
 */
function Page({ route }: { route: Route }) {
  switch (route.kind) {
    case "front":
      return <Front />;
    case "chapter":
      return <Chapter n={route.n} />;
    case "rope":
      return <Rope />;
    case "architecture":
      return <Architecture />;
    case "tests":
      return <Tests />;
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
