import { useEffect, useRef, useState } from "react";

import { MEASURED } from "../content/measured";

const W = 1000;
const H = 320;
const PAD_L = 56;
const PAD_R = 120;
const PAD_T = 18;
const PAD_B = 40;

const POINTS = MEASURED.cache.points;

/**
 * Ease a value from 0 to 1 whenever the target flips.
 *
 * Worth the twelve lines: the claim is that one three-line change moved the cache from
 * losing at every length to winning at the longest, and a curve that *travels* between
 * the two states makes that a thing you watch happen rather than two pictures you
 * compare. Respects `prefers-reduced-motion` by snapping instead.
 */
function useTween(target: number, ms = 420): number {
  const [value, setValue] = useState(target);
  const from = useRef(target);
  const start = useRef(0);

  useEffect(() => {
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) {
      setValue(target);
      return;
    }
    from.current = value;
    start.current = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start.current) / ms);
      // Smoothstep: no visible jerk at either end, which a linear ramp has at both.
      const eased = t * t * (3 - 2 * t);
      setValue(from.current + (target - from.current) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // `value` is deliberately not a dependency: it changes on every frame, and reading
    // it here is how the tween starts from wherever an interrupted one left off.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, ms]);

  return value;
}

/**
 * Cached decoding against recomputing from scratch, with the mask bug on a switch.
 *
 * The most instructive figure on the site, because its subject is a mistake. With the
 * mask restored the cache loses at every length — which is what was measured, explained
 * away as a property of small-model decoding, and written up as a finding. Flip it off
 * and the same code wins at 1,024 tokens. Both curves are measured: `benchmarks-cuda.json`
 * and `benchmarks-cuda-before-mask-fix.json`, taken at the commits either side of the fix.
 */
export default function CacheSweep({ masked, onMasked }: { masked: boolean; onMasked: (v: boolean) => void }) {
  const t = useTween(masked ? 1 : 0);

  const values = POINTS.map((p) => ({
    ...p,
    // At t = 0 this is the fixed measurement, at t = 1 the buggy one, and in between it
    // is neither — an interpolation for the eye, never a number the page prints.
    cachedNow: p.cached + (p.cachedBefore - p.cached) * t,
    naiveNow: p.naive + (p.naiveBefore - p.naive) * t,
  }));

  const yMax = 280;
  const sx = (i: number) => PAD_L + (i / (POINTS.length - 1)) * (W - PAD_L - PAD_R);
  const sy = (v: number) => PAD_T + (1 - v / yMax) * (H - PAD_T - PAD_B);
  const path = (get: (v: (typeof values)[number]) => number) =>
    values.map((v, i) => `${i === 0 ? "M" : "L"} ${sx(i)} ${sy(get(v))}`).join(" ");

  const winning = values.at(-1)!.cachedNow > values.at(-1)!.naiveNow;

  return (
    <>
      <svg
        className="loss-curve"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Decode throughput against sequence length, cached versus recomputed, with the decode-step mask ${masked ? "present" : "removed"}`}
      >
        {[0, 70, 140, 210, 280].map((y) => (
          <g key={y}>
            <line x1={PAD_L} y1={sy(y)} x2={W - PAD_R} y2={sy(y)} stroke="var(--color-neutral-200)" />
            <text x={PAD_L - 8} y={sy(y) + 4} fontSize={11} fill="var(--color-neutral-600)"
              textAnchor="end" fontFamily="var(--mono)">
              {y}
            </text>
          </g>
        ))}
        {values.map((v, i) => (
          <text key={v.totalLen} x={sx(i)} y={H - 14} fontSize={11} fill="var(--color-neutral-600)"
            textAnchor="middle" fontFamily="var(--mono)">
            {v.totalLen}
          </text>
        ))}

        <path d={path((v) => v.naiveNow)} fill="none" stroke="var(--color-neutral-500)"
          strokeWidth={2} strokeDasharray="6 4" />
        <path d={path((v) => v.cachedNow)} fill="none"
          stroke={winning ? "var(--color-accent)" : "var(--color-accent-2)"} strokeWidth={2.8} />

        {values.map((v, i) => (
          <g key={v.totalLen}>
            <circle cx={sx(i)} cy={sy(v.naiveNow)} r={3.5} fill="var(--color-neutral-500)" />
            <circle cx={sx(i)} cy={sy(v.cachedNow)} r={5}
              fill={winning ? "var(--color-accent)" : "var(--color-accent-2)"}
              stroke="var(--color-bg)" strokeWidth={2} />
          </g>
        ))}

        <text x={W - PAD_R + 12} y={sy(values.at(-1)!.naiveNow) + 4} fontSize={13}
          fill="var(--color-neutral-600)" fontFamily="var(--mono)">
          recompute
        </text>
        <text x={W - PAD_R + 12} y={sy(values.at(-1)!.cachedNow) + 4} fontSize={13}
          fill={winning ? "var(--color-accent-700)" : "var(--color-accent-2-700)"}
          fontFamily="var(--mono)" fontWeight={600}>
          KV cache
        </text>
        <text x={PAD_L} y={H - 14} fontSize={11} fill="var(--color-neutral-600)"
          textAnchor="start" fontFamily="var(--mono)" opacity={0}>
          .
        </text>
      </svg>

      <div className="fig-row fig-row-wide" style={{ marginTop: "var(--space-3)" }}>
        <label className="field field-inline">
          <input type="checkbox" checked={masked} onChange={(e) => onMasked(e.target.checked)} />
          build the causal mask on the decode step
        </label>
        <span className="fig-note" style={{ margin: 0 }}>
          {masked
            ? `The bug, as measured at ${MEASURED.cache.commitBefore.slice(0, 8)}: the cache loses at every length.`
            : `Fixed, at ${MEASURED.cache.commitAfter.slice(0, 8)}: the same cache wins by ${values.at(-1)!.advantage.toFixed(2)}× at ${values.at(-1)!.totalLen}.`}
        </span>
      </div>
    </>
  );
}
