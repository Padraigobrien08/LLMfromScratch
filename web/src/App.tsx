import { Suspense, lazy } from "react";

import Footer from "./components/Footer";
import Masthead from "./components/Masthead";
import Front from "./pages/Front";
import { type Route, useRoute } from "./router";

/**
 * Every page except the front one is loaded on demand.
 *
 * The front page is the landing page and it is cheap: a figure strip, the path, and the
 * status table. Everything behind it is not — the explainer chapters carry a tokenizer
 * demo, a parameter calculator and a sampler, the results plates carry their figures,
 * and before this split a reader who opened the front page and left had downloaded all
 * of it. Splitting is worth roughly two thirds of the initial payload, and the cost is a
 * fetch on the first navigation to each route, which is exactly when the reader has
 * asked for it.
 *
 * `Front` stays eagerly imported on purpose: lazy-loading the page that is already being
 * rendered adds a round trip to the only view whose speed anyone measures.
 *
 * Two things used to live here and are worth recording as gone. `Legacy` re-pointed the
 * pre-redesign token names at Broadsheet's so `#/ablations` could render in the new ink
 * without being rewritten; it went out with that rewrite. `Placeholder` kept the router
 * honest while `#/architecture` and `#/tests` were outstanding — every route resolves to
 * a real page now, so the switch below is exhaustive over `Route` with nothing standing
 * in for anything.
 */
const Chapter = lazy(() => import("./pages/Chapter"));
const Rope = lazy(() => import("./pages/Rope"));
const Architecture = lazy(() => import("./pages/Architecture"));
const Tests = lazy(() => import("./pages/Tests"));
const Reproduction = lazy(() => import("./pages/Reproduction"));
const Ablations = lazy(() => import("./pages/Ablations"));
const Efficiency = lazy(() => import("./pages/Efficiency"));
const Scaling = lazy(() => import("./pages/Scaling"));

/**
 * Deliberately close to empty.
 *
 * These chunks are tens of kilobytes on a connection that has already fetched the shell,
 * so a spinner would appear and vanish inside one frame — which reads as a flicker, not
 * as progress. The live region is for screen readers, who get no visual cue at all that
 * a navigation is in flight.
 */
function PageLoading() {
  return (
    <div className="shell page-loading" role="status" aria-live="polite">
      Loading…
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
      <a className="skip-link" href="#main">
        Skip to content
      </a>
      <Masthead route={route} />
      <main id="main" tabIndex={-1}>
        <Suspense fallback={<PageLoading />}>
          <Page route={route} />
        </Suspense>
      </main>
      <Footer route={route} />
    </>
  );
}
