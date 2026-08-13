import { useEffect, useRef, useState } from "react";

import PlateNumeral from "./PlateNumeral";
import { type Variant } from "../content/blocks";
import { figurePanel } from "../content/stackFigure";
import { formatCount, parameters } from "../lib/modelsize";
import { SIZES } from "../content/blocks";
import { mountStackFigure, type StackEngine } from "./stack/engine";

/**
 * Figure V — the stack, in three dimensions.
 *
 * The division of labour: React owns the plate chrome, the control and the detail
 * panel, because those are content and links and need to be readable by everything
 * that reads a page. `stack/engine.ts` owns the drawing, because it redraws while the
 * reader orbits. The engine reports selection back up so the panel is rendered once,
 * here, from `content/stackFigure.ts` — which resolves nine of the fourteen parts
 * straight out of `blocks.ts`.
 *
 * Nothing in this component knows a parameter count, a shape or a test name.
 */

const VARIANTS: Array<{ id: Variant; label: string }> = [
  { id: "gpt2", label: "GPT-2" },
  { id: "llama", label: "Llama-style" },
];

export default function StackFigure({ attentionHref }: { attentionHref: string }) {
  const [variant, setVariant] = useState<Variant>("gpt2");
  const [selected, setSelected] = useState<string>("whole");
  const [note, setNote] = useState("");

  const wrapRef = useRef<HTMLDivElement>(null);
  const flatRef = useRef<SVGSVGElement>(null);
  const leadersRef = useRef<SVGSVGElement>(null);
  const labelsRef = useRef<HTMLDivElement>(null);
  const engineRef = useRef<StackEngine | null>(null);
  const [announcement, setAnnouncement] = useState("");

  useEffect(() => {
    const engine = mountStackFigure({
      canvasWrap: wrapRef.current!,
      flat: flatRef.current!,
      leaders: leadersRef.current!,
      labelLayer: labelsRef.current!,
      variant,
      selected,
      onSelect: setSelected,
      onAnnounce: setAnnouncement,
      onMode: setNote,
    });
    engineRef.current = engine;
    return () => {
      engine.destroy();
      engineRef.current = null;
    };
    // Mounted once: variant and selection are pushed in through the handle below,
    // because rebuilding the scene on every click would throw away the orbit.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => engineRef.current?.setVariant(variant), [variant]);
  useEffect(() => engineRef.current?.setSelected(selected), [selected]);

  const panel = figurePanel(selected, variant, attentionHref);
  const total = parameters(SIZES[variant]).total;
  /* The whole object is the denominator, so printing its share would say "100.0% of"
     itself. It gets the total alone; everything else gets its slice of it. */
  const share =
    panel.params == null
      ? "No parameters of its own"
      : selected === "whole"
        ? `${formatCount(panel.params)} in total`
        : `${formatCount(panel.params)} · ${((panel.params / total) * 100).toFixed(1)}% of ${formatCount(total)}`;

  useEffect(() => {
    setAnnouncement(`${panel.name} selected`);
  }, [panel.name]);

  return (
    <section className="stack-figure" aria-labelledby="stack-figure-title">
      <div className="rule-hair" />
      <header className="stack-fig-head">
        <span className="stack-fig-numeral" aria-hidden="true">
          <PlateNumeral value="V" />
        </span>
        <div>
          <p className="kicker">Figure · the model this repository builds</p>
          <h2 id="stack-figure-title" className="stack-fig-title">
            The stack, in three dimensions
          </h2>
        </div>
      </header>

      <div className="stack-fig-stage">
        <div className="stack-fig-area">
          <div className="stack-fig-canvas" ref={wrapRef}>
            <svg className="stack-fig-flat" ref={flatRef} aria-hidden="true" />
            <svg className="stack-fig-leaders" ref={leadersRef} aria-hidden="true" />
          </div>
          {/* The engine creates the label buttons in here: they are positioned against a
              projection that moves with the orbit, so React does not own their layout. */}
          <div className="stack-fig-labels" ref={labelsRef} />
        </div>

        <aside className="stack-fig-panel">
          <p className="eyebrow">{selected === "whole" ? "The object" : "Selected block"}</p>
          <h3 className="stack-fig-panel-title">{panel.name}</h3>
          <p className="stack-fig-what">{panel.what}</p>
          {panel.differs && (
            <p className="stack-fig-differs">
              {VARIANTS.find((v) => v.id === variant)!.label}: {panel.differs}
            </p>
          )}

          <p className="eyebrow">Its shape</p>
          <p className="stack-fig-shape mono">{panel.shape}</p>

          <p className="eyebrow">Its share of the budget</p>
          <p className="stack-fig-shape mono">{share}</p>

          <p className="eyebrow">What holds it</p>
          {panel.pins ? (
            <p className="stack-fig-pin">
              <code className="mono">{panel.pins.test}</code> asserts {panel.pins.claim}
            </p>
          ) : (
            <p className="stack-fig-pin stack-fig-unpinned">
              No property test of its own. The suite exercises this wherever it runs a
              forward pass, but nothing asserts an invariant about it — and saying so beats
              borrowing a neighbour's test to fill the row.
            </p>
          )}

          <p className="eyebrow">Read more</p>
          <nav className="stack-fig-links">
            {panel.links.map((link) => (
              <a
                key={link.href}
                href={link.href}
                {...(link.external ? { target: "_blank", rel: "noopener" } : {})}
              >
                {link.text}
                {link.external && " ↗"}
              </a>
            ))}
          </nav>
        </aside>

        <div className="stack-fig-foot">
          <div className="seg" role="radiogroup" aria-label="Architecture">
            {VARIANTS.map((v) => (
              <label key={v.id} className="seg-opt">
                <input
                  type="radio"
                  name="stack-figure-architecture"
                  value={v.id}
                  checked={variant === v.id}
                  onChange={() => setVariant(v.id)}
                />
                {v.label}
              </label>
            ))}
          </div>
          <p className="fig-note">{note}</p>
          <p className="fig-note">
            Slab thickness is that component's share of the parameter budget, summed over all
            layers; the block region is sliced once per layer, so its depth is the layer count
            itself. Parts that hold no weights of their own — the input, the logits, the tied
            head — are drawn at a floor thickness. Colour is role, printed as the four process
            plates: cyan for the token embedding and the head that shares its matrix, magenta
            for attention and its cache, yellow for the feed-forward, ink for the norms and the
            residual stream. Every number and sentence is read from the shipped configs,{" "}
            <code className="mono">blocks.ts</code> and <code className="mono">modelsize.ts</code>.
          </p>
          <div className="rule-hair" />
        </div>
      </div>

      <p className="stack-fig-sr" role="status" aria-live="polite">
        {announcement}
      </p>
    </section>
  );
}
