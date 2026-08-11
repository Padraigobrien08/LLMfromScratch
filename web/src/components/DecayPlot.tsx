import { useMemo } from "react";

import { dotRelative } from "../lib/rope";

const W = 1000;
const H = 260;
const PAD_L = 52;
const PAD_R = 16;
const PAD_T = 14;
const PAD_B = 34;

type Props = {
  q: Float64Array;
  k: Float64Array;
  theta: number;
  d: number;
  maxD: number;
};

/**
 * The logit as a function of relative offset alone.
 *
 * Worth drawing because it is the shape people assume RoPE has and does not: not a
 * smooth decay but an oscillating one, the sum of many cosines at different rates.
 * The marker is the current `m - n`, and dragging either token slides it along a
 * curve that never itself changes.
 */
export default function DecayPlot({ q, k, theta, d, maxD }: Props) {
  const { yMin, yMax, points } = useMemo(() => {
    const pts: Array<[number, number]> = [];
    // One sample per offset up to a few hundred, then subsample: at this width more
    // points than pixels is wasted work that also aliases into visual noise.
    const stride = Math.max(1, Math.round(maxD / 900));
    for (let x = 0; x <= maxD; x += stride) pts.push([x, dotRelative(q, k, x, theta)]);
    const ys = pts.map((p) => p[1]);
    return { points: pts, yMin: Math.min(...ys), yMax: Math.max(...ys) };
  }, [q, k, theta, maxD]);

  const pad = (yMax - yMin) * 0.12 || 0.1;
  const lo = yMin - pad;
  const hi = yMax + pad;

  const sx = (x: number) => PAD_L + (x / maxD) * (W - PAD_L - PAD_R);
  const sy = (y: number) => PAD_T + (1 - (y - lo) / (hi - lo)) * (H - PAD_T - PAD_B);

  const line = points.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p[0])} ${sy(p[1])}`).join(" ");

  const absD = Math.abs(d);
  const current = dotRelative(q, k, d, theta);
  const gridYs = [lo, (lo + hi) / 2, hi];
  const gridXs = [0, maxD * 0.25, maxD * 0.5, maxD * 0.75, maxD];

  return (
    <svg className="decay-plot" viewBox={`0 0 ${W} ${H}`} role="img"
      aria-label="Attention logit as a function of relative offset">
      {gridYs.map((y) => (
        <g key={y}>
          <line x1={PAD_L} y1={sy(y)} x2={W - PAD_R} y2={sy(y)} stroke="var(--color-neutral-200)" />
          <text x={PAD_L - 8} y={sy(y) + 4} fontSize={11} fill="var(--color-neutral-600)"
            textAnchor="end" fontFamily="var(--mono)">
            {y.toFixed(2)}
          </text>
        </g>
      ))}
      {gridXs.map((x) => (
        <text key={x} x={sx(x)} y={H - 12} fontSize={11} fill="var(--color-neutral-600)"
          textAnchor="middle" fontFamily="var(--mono)">
          {Math.round(x)}
        </text>
      ))}

      <path d={line} fill="none" stroke="var(--color-accent)" strokeWidth={2}
        strokeLinejoin="round" />

      {absD <= maxD && (
        <g>
          <line x1={sx(absD)} y1={PAD_T} x2={sx(absD)} y2={H - PAD_B}
            stroke="var(--color-accent-2)" strokeWidth={1.5} strokeDasharray="4 3" />
          <circle cx={sx(absD)} cy={sy(current)} r={5} fill="var(--color-accent-2)"
            stroke="var(--color-neutral-100)" strokeWidth={2} />
        </g>
      )}
    </svg>
  );
}
