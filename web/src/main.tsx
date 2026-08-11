import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import App from "./App";
// Broadsheet, copied verbatim from the design handoff: the token sheet and component
// layer, then the bundle that inlines the SVG separation filter defs the plate
// treatments print with and drives their registration. Both load before the site's
// own layer, which is built entirely out of their variables.
import "./design-system/styles.css";
import "./design-system/_ds_bundle.js";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
