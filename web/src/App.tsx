import Nav from "./components/Nav";
import Ablations from "./pages/Ablations";
import Explainer from "./pages/Explainer";
import Overview from "./pages/Overview";
import Rope from "./pages/Rope";
import { useRoute } from "./router";

const PAGES: Record<string, () => React.ReactElement> = {
  "": Overview,
  explainer: Explainer,
  rope: Rope,
  ablations: Ablations,
};

export default function App() {
  const route = useRoute();
  const Page = PAGES[route] ?? Overview;

  return (
    <>
      <Nav route={route} />
      <main>
        <Page />
      </main>
    </>
  );
}
