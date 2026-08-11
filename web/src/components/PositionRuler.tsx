import { useCallback, useRef } from "react";

const W = 1000;
const H = 122;
const PAD = 30;
const AXIS_Y = 88;
const SPAN = W - 2 * PAD;

type Props = {
  maxPos: number;
  m: number;
  n: number;
  gapLocked: boolean;
  onChange: (m: number, n: number) => void;
};

/**
 * The two token positions, draggable along a sequence.
 *
 * Sliders would have been less code, but the thing worth feeling here is that the
 * two markers are *points in a sequence* — and that under a locked gap they move as
 * one rigid object. Handles on a shared axis show that; two unrelated sliders do not.
 */
export default function PositionRuler({ maxPos, m, n, gapLocked, onChange }: Props) {
  const svgRef = useRef<SVGSVGElement>(null);
  const dragging = useRef<"m" | "n" | null>(null);

  const toPos = useCallback(
    (clientX: number) => {
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return 0;
      const vbX = ((clientX - rect.left) / rect.width) * W;
      const frac = (vbX - PAD) / SPAN;
      return Math.max(0, Math.min(maxPos, Math.round(frac * maxPos)));
    },
    [maxPos],
  );

  const move = useCallback(
    (which: "m" | "n", next: number) => {
      if (!gapLocked) {
        onChange(which === "m" ? next : m, which === "n" ? next : n);
        return;
      }
      // Rigid translation: preserve m - n, and refuse to move rather than let the
      // pair deform when it hits an end. A gap that silently changed while the
      // readout claims it is held would undermine the whole demonstration.
      const gap = m - n;
      const nextN = which === "m" ? next - gap : next;
      if (nextN < 0 || nextN + gap < 0 || nextN > maxPos || nextN + gap > maxPos) return;
      onChange(nextN + gap, nextN);
    },
    [gapLocked, m, n, maxPos, onChange],
  );

  const onPointerDown = (which: "m" | "n") => (e: React.PointerEvent) => {
    dragging.current = which;
    (e.target as Element).setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    move(dragging.current, toPos(e.clientX));
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (dragging.current) (e.target as Element).releasePointerCapture?.(e.pointerId);
    dragging.current = null;
  };

  const onKeyDown = (which: "m" | "n") => (e: React.KeyboardEvent) => {
    const step = e.shiftKey ? 10 : 1;
    const current = which === "m" ? m : n;
    if (e.key === "ArrowLeft") move(which, current - step);
    else if (e.key === "ArrowRight") move(which, current + step);
    else if (e.key === "Home") move(which, 0);
    else if (e.key === "End") move(which, maxPos);
    else return;
    e.preventDefault();
  };

  const x = (pos: number) => PAD + (pos / maxPos) * SPAN;
  const tickEvery = maxPos <= 64 ? 8 : maxPos <= 256 ? 32 : 128;
  const ticks = Array.from({ length: Math.floor(maxPos / tickEvery) + 1 }, (_, i) => i * tickEvery);

  return (
    <svg
      ref={svgRef}
      className="ruler"
      viewBox={`0 0 ${W} ${H}`}
      role="group"
      aria-label="Token positions"
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onPointerCancel={onPointerUp}
    >
      <line
        x1={PAD}
        y1={AXIS_Y}
        x2={W - PAD}
        y2={AXIS_Y}
        stroke="var(--color-neutral-400)"
        strokeWidth={2}
      />
      {ticks.map((t) => (
        <g key={t}>
          <line
            x1={x(t)}
            y1={AXIS_Y}
            x2={x(t)}
            y2={AXIS_Y + 7}
            stroke="var(--color-neutral-400)"
            strokeWidth={2}
          />
          <text
            x={x(t)}
            y={AXIS_Y + 22}
            fontSize={11}
            fill="var(--color-neutral-600)"
            textAnchor="middle"
            fontFamily="var(--mono)"
          >
            {t}
          </text>
        </g>
      ))}

      {/* The gap itself, drawn as an object — it is the only quantity the logit sees. */}
      <line
        x1={x(Math.min(m, n))}
        y1={AXIS_Y}
        x2={x(Math.max(m, n))}
        y2={AXIS_Y}
        stroke="var(--color-accent)"
        strokeWidth={5}
        strokeLinecap="round"
        opacity={0.4}
      />

      {(
        [
          // Stacked on two rows rather than one: a small gap is the interesting
          // case, and on one row the two pills would cover each other exactly when
          // the reader most wants to see both.
          { which: "n", pos: n, label: "K", color: "var(--color-accent-2)", y: AXIS_Y - 66 },
          { which: "m", pos: m, label: "Q", color: "var(--color-accent)", y: AXIS_Y - 28 },
        ] as const
      ).map((h) => (
        <g
          key={h.which}
          className="handle"
          tabIndex={0}
          role="slider"
          aria-label={h.which === "m" ? "Query position m" : "Key position n"}
          aria-valuemin={0}
          aria-valuemax={maxPos}
          aria-valuenow={h.pos}
          onPointerDown={onPointerDown(h.which)}
          onKeyDown={onKeyDown(h.which)}
        >
          <line
            x1={x(h.pos)}
            y1={h.y + 22}
            x2={x(h.pos)}
            y2={AXIS_Y}
            stroke={h.color}
            strokeWidth={2}
          />
          {/* rx=2 is --radius-md: nothing in this design is rounder than 2px except
              the gap line's cap. */}
          <rect x={x(h.pos) - 23} y={h.y - 12} width={46} height={26} rx={2} fill={h.color} />
          <text
            x={x(h.pos)}
            y={h.y + 5}
            fontSize={13}
            fill="var(--color-bg)"
            textAnchor="middle"
            fontWeight={600}
            fontFamily="var(--mono)"
            pointerEvents="none"
          >
            {h.label}
            {h.pos}
          </text>
          {/* A generous invisible target: the visible pill is too small to grab on a phone. */}
          <rect
            x={x(h.pos) - 28}
            y={h.y - 18}
            width={56}
            height={44}
            fill="transparent"
            pointerEvents="all"
          />
        </g>
      ))}
    </svg>
  );
}
