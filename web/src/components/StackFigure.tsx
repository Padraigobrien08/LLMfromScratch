import { useEffect, useRef, useState } from "react";

import { type Variant } from "../content/blocks";
import { FIGURE_LABELS, figurePanel } from "../content/stackFigure";
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

/**
 * Every part the panel can be showing — the fourteen labels' blocks, plus the whole
 * object, which is what it opens on and which no label points at.
 *
 * Derived from `FIGURE_LABELS` rather than listed, so a part added to the figure is
 * measured by the description slot without a second edit here.
 */
const PARTS: string[] = [...new Set(["whole", ...FIGURE_LABELS.map((l) => l.blockId)])];

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
  /**
   * The chapter link, promoted out of the list into the panel's call to action.
   *
   * `stackFigure.ts` already gives most parts a chapter as their first link, so this
   * needs no new content — only a way to tell which one it is, which the route is.
   * Detecting it beats tagging it: a link added there is picked up here without a
   * second edit, and a part with no chapter simply prints no button rather than a
   * dead one.
   */
  const chapter = panel.links.find((link) => link.href.startsWith("#/chapter/"));
  const rest = panel.links.filter((link) => link !== chapter);
  const total = parameters(SIZES[variant]).total;
  /* The whole object is the denominator, so printing its share would say "100.0% of"
     itself. It gets the total alone; everything else gets its slice of it. */
  const shareOf = (p: { params: number | null }, id: string) =>
    p.params == null
      ? "No parameters of its own"
      : id === "whole"
        ? `${formatCount(p.params)} in total`
        : `${formatCount(p.params)} · ${((p.params / total) * 100).toFixed(1)}% of ${formatCount(total)}`;

  /**
   * Every part's panel, resolved once for the current variant.
   *
   * The three variable fields below are each set as a stack of all fourteen values with
   * one visible, so that each reserves the height of its own longest entry — which is
   * what stops "Its shape", "Its share of the budget" and "What holds it" moving as the
   * reader clicks from one block to the next.
   */
  const parts = PARTS.map((id) => ({
    id,
    p: id === selected ? panel : figurePanel(id, variant, attentionHref),
  }));

  useEffect(() => {
    setAnnouncement(`${panel.name} selected`);
  }, [panel.name]);

  return (
    <section className="stack-figure" aria-labelledby="stack-figure-title">
      <div className="rule-hair" />
      {/* Two words and a rule. The plate numeral and the "Figure · the model this
          repository builds" kicker were furniture from when this sat among four numbered
          results plates on a much longer page; with the plates gone from the front page
          there is no series left for a V to be the fifth of, and the line under it
          described what the reader is already looking at. */}
      <header className="stack-fig-head">
        <h2 id="stack-figure-title" className="stack-fig-title">
          The architecture
        </h2>
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

        {/* Two parts, and the split is load-bearing where the panel is given a fixed
            height: the prose scrolls and the way out does not. The parts run 374px to
            547px of copy, so with one scrolling box the reader's next step — the chapter
            button — sat below the fold on half the stack, which is the one thing in here
            that must never need looking for. */}
        <aside className="stack-fig-panel">
          {/* Grouped into fields so the panel keeps its shape from part to part.
              Flat, every row sat wherever the row above it ended: the description runs from
              two lines to five and only some parts print a variant note, so "Its shape" and
              the two labels under it moved through 119px depending on what had been clicked.
              A reader comparing two blocks was re-finding the same four labels each time.
              The wrappers are what let the panel reserve a slot for each. */}
          {/* A scroll region, so it is focusable and can be scrolled by keyboard — the
              same treatment `DataTable` gives a wide table. On a short window this box
              holds more than it can show, and without a tab stop the only way to reach
              the rest was a pointer. */}
          <div
            className="stack-fig-panel-body"
            role="region"
            aria-label={`${panel.name} — detail`}
            tabIndex={0}
          >
            <p className="eyebrow">{selected === "whole" ? "The object" : "Selected block"}</p>
            <h3 className="stack-fig-panel-title">{panel.name}</h3>

            {/**
             * Every part's description, stacked in one grid cell, with all but the
             * selected one hidden.
             *
             * The slot has to be as tall as the longest description or the fields below it
             * move as the reader clicks from block to block. Reserving a fixed eight lines
             * for it worked at the width it was measured at and nowhere else: in a 328px
             * panel the same prose wraps to thirteen, and seven of the fourteen parts
             * overflowed their slot by up to 150px — an inner scrollbar on the one piece
             * of the panel that should never need one.
             *
             * Stacked, the cell is the height of the tallest description at whatever width
             * the panel currently has, and that is true at every width without a number
             * being written down. `visibility: hidden` rather than `display: none` because
             * the hidden ones still have to take up their space to do the reserving; it
             * also takes them out of the accessibility tree, and `aria-hidden` says so
             * explicitly.
             */}
            <div className="stack-fig-prose">
              {parts.map(({ id, p }) => (
                <div key={id} data-on={id === selected ? "1" : "0"} aria-hidden={id !== selected}>
                  <p className="stack-fig-what">{p.what}</p>
                  {p.differs && (
                    <p className="stack-fig-differs">
                      {VARIANTS.find((v) => v.id === variant)!.label}: {p.differs}
                    </p>
                  )}
                </div>
              ))}
            </div>

            <div className="stack-fig-field">
              <p className="eyebrow">Its shape</p>
              {/* Stacked for the same reason the description is: "12 layers · 768 wide ·
                  12 heads · 1,024 context" takes two lines in a narrow panel where
                  "batch × time × 50,304" takes one, and that difference moved the two
                  fields under it. */}
              <div className="stack-fig-stack">
                {parts.map(({ id, p }) => (
                  <p key={id} className="stack-fig-shape mono"
                     data-on={id === selected ? "1" : "0"} aria-hidden={id !== selected}>
                    {p.shape}
                  </p>
                ))}
              </div>
            </div>

            <div className="stack-fig-field">
              <p className="eyebrow">Its share of the budget</p>
              <div className="stack-fig-stack">
                {parts.map(({ id, p }) => (
                  <p key={id} className="stack-fig-shape mono"
                     data-on={id === selected ? "1" : "0"} aria-hidden={id !== selected}>
                    {shareOf(p, id)}
                  </p>
                ))}
              </div>
            </div>

            <div className="stack-fig-field stack-fig-holds">
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
            </div>
          </div>

          <div className="stack-fig-panel-exit">
            {/* The figure's whole claim to being navigation rather than decoration: a
                reader who has just clicked a block and read what it does is one button
                from the chapter that explains it. */}
            {chapter && (
              <a className="stack-fig-read" href={chapter.href}>
                <span className="stack-fig-read-label">Read</span>
                {chapter.text} →
              </a>
            )}

            {rest.length > 0 && (
              <>
                <p className="eyebrow">Also</p>
                <nav className="stack-fig-links">
                  {rest.map((link) => (
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
              </>
            )}
          </div>
        </aside>

        <div className="stack-fig-foot">
          <div className="seg seg-paper" role="radiogroup" aria-label="Architecture">
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
          {/* What the geometry means, in one line.
              The full legend was cut for height, and with it went the only statement that
              the drawing is derived rather than drawn: a slab whose thickness encodes a
              share of the parameter budget is decoration until a reader is told so. This
              is the shortest form of that claim, and it sits beside the variant control
              where the reader is already looking. */}
          <p className="fig-encoding">
            Thickness is share of parameters · colour is role
          </p>
          {/* The interaction note is kept for assistive technology and taken off the page.
              On screen the figure announces itself — the cursor changes over a block and
              the labels are buttons — but a reader who cannot see it has nothing else that
              says the plate can be orbited or clicked at all. It costs no height here. */}
          <p className="fig-note stack-fig-sr">{note}</p>
          <div className="rule-hair" />
        </div>
      </div>

      <p className="stack-fig-sr" role="status" aria-live="polite">
        {announcement}
      </p>
    </section>
  );
}
