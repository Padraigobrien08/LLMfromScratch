import FigureScroll from "./FigureScroll";

const W = 1000;
const H = 300;
const PAD_L = 56;
const PAD_R = 18;
const PAD_T = 16;
// Two lines below the plot: the ticks, then the axis title on its own — the arrangement
// `AccumCurve` already uses. Sharing one line put "step" underneath the last tick, which
// was already a collision at 11 units and an unreadable one once a phone needs 17.
const PAD_B = 52;

export type Series = {
  label: string;
  color: string;
  dashed?: boolean;
  points: Array<{ step: number; loss: number }>;
};

/** Validation loss against step, for however many arms are on screen. */
export default function CurvePlot({ series }: { series: Series[] }) {
  const all = series.flatMap((s) => s.points);
  if (!all.length) {
    return (
      <p className="fig-note" style={{ margin: 0 }}>
        No curve data for this selection.
      </p>
    );
  }

  const xs = all.map((p) => p.step);
  const ys = all.map((p) => p.loss);
  const xMax = Math.max(...xs);
  const xMin = Math.min(...xs);
  let lo = Math.min(...ys);
  let hi = Math.max(...ys);
  // The whole question is a gap of a few hundredths between curves. Padding the
  // range to a "nice" round number would flatten exactly the difference on trial.
  const margin = (hi - lo) * 0.1 || 0.05;
  lo -= margin;
  hi += margin;

  const sx = (x: number) => PAD_L + ((x - xMin) / Math.max(xMax - xMin, 1)) * (W - PAD_L - PAD_R);
  const sy = (y: number) => PAD_T + (1 - (y - lo) / (hi - lo)) * (H - PAD_T - PAD_B);

  const yTicks = [lo, lo + (hi - lo) / 2, hi];
  const xTicks = [xMin, Math.round((xMin + xMax) / 2), xMax];

  // Display only — the domain above stays exactly what the data spans. `6.228` on an
  // axis reads as a serial number; `6.23` reads as a quantity, and the third digit was
  // never a number anyone could act on. Three decimals return if two would print the
  // same label twice, which a tight enough range would.
  const decimals = new Set(yTicks.map((y) => y.toFixed(2))).size === yTicks.length ? 2 : 3;

  return (
    <FigureScroll label="Validation loss by step">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Validation loss by step">
        {yTicks.map((y) => (
          <g key={y}>
            <line x1={PAD_L} y1={sy(y)} x2={W - PAD_R} y2={sy(y)} stroke="var(--color-neutral-200)" />
            <text x={PAD_L - 8} y={sy(y) + 4} fontSize={11} fill="var(--color-neutral-700)" textAnchor="end"
              fontFamily="var(--mono)">
              {y.toFixed(decimals)}
            </text>
          </g>
        ))}
        {xTicks.map((x, i) => (
          // The end ticks sit on the plot's edges, so centring them hangs half of each
          // off the viewBox — the same correction `LossCurve` makes.
          <text key={x} x={sx(x)} y={H - 28} fontSize={11} fill="var(--color-neutral-700)"
            textAnchor={i === 0 ? "start" : i === xTicks.length - 1 ? "end" : "middle"}
            fontFamily="var(--mono)">
            {x}
          </text>
        ))}
        <text x={(PAD_L + W - PAD_R) / 2} y={H - 6} fontSize={11} fill="var(--color-neutral-700)"
          textAnchor="middle">
          step
        </text>

        {series.map((s) => (
          <path
            key={s.label}
            d={s.points.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.step)} ${sy(p.loss)}`).join(" ")}
            fill="none"
            stroke={s.color}
            strokeWidth={2}
            strokeDasharray={s.dashed ? "5 4" : undefined}
            strokeLinejoin="round"
          />
        ))}

        {series.map((s, i) => (
          <g key={s.label} transform={`translate(${PAD_L + 10}, ${PAD_T + 14 + i * 18})`}>
            <line x1={0} y1={-4} x2={18} y2={-4} stroke={s.color} strokeWidth={2}
              strokeDasharray={s.dashed ? "5 4" : undefined} />
            <text x={24} y={0} fontSize={12} fill="var(--color-text)" fontFamily="var(--mono)">
              {s.label}
            </text>
          </g>
        ))}
      </svg>
    </FigureScroll>
  );
}
