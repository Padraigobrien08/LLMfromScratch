import { MEASURED } from "../content/measured";
import { curvePoints, efficiencyAt, residual } from "../lib/amortisation";

const W = 1000;
const H = 330;
const PAD_L = 60;
const PAD_R = 24;
const PAD_T = 18;
const PAD_B = 42;

const { a, b } = MEASURED.accumulation.fit;
const POINTS = MEASURED.accumulation.points;

const MIN_ACCUM = 1;
const MAX_ACCUM = 8;
const Y_LO = 84;
const Y_HI = 100;

// Log2, because the sweep doubled: 1, 2, 4, 8 land evenly and the halvings between them
// get the same width as the halvings anywhere else.
const lg = (accum: number) => Math.log2(accum);
const sx = (accum: number) =>
  PAD_L + (lg(accum) / lg(MAX_ACCUM)) * (W - PAD_L - PAD_R);
const sy = (pct: number) => PAD_T + (1 - (pct - Y_LO) / (Y_HI - Y_LO)) * (H - PAD_T - PAD_B);

type Props = {
  /** Where the slider sits, in accumulation steps. Continuous — the model has no grid. */
  accum: number;
  revealed: boolean;
};

/**
 * Scaling efficiency against gradient accumulation, and a curve that predicted half of
 * the points on it.
 *
 * The best figure the project has, because the mechanism is *predictive* rather than
 * descriptive. `a + b/accum` was fitted to the accum 8 and 4 measurements — drawn filled
 * — and then used to state where accum 2 and 1 would land before either had been run.
 * Those two are hidden until the reader asks for them, because a curve drawn through
 * four points and a curve that anticipated two of them look identical afterwards, and
 * only one of them is evidence.
 */
export default function AccumCurve({ accum, revealed }: Props) {
  const curve = curvePoints(a, b, MIN_ACCUM, MAX_ACCUM);
  const path = curve
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.accum)} ${sy(p.efficiency * 100)}`)
    .join(" ");

  const predictedHere = efficiencyAt(a, b, accum) * 100;

  return (
    <svg
      className="loss-curve"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label="Scaling efficiency at eight GPUs against gradient accumulation, with the fitted amortisation curve"
    >
      {[86, 90, 94, 98].map((y) => (
        <g key={y}>
          <line x1={PAD_L} y1={sy(y)} x2={W - PAD_R} y2={sy(y)} stroke="var(--color-neutral-200)" />
          <text x={PAD_L - 8} y={sy(y) + 4} fontSize={11} fill="var(--color-neutral-600)"
            textAnchor="end" fontFamily="var(--mono)">
            {y}%
          </text>
        </g>
      ))}
      {[1, 2, 4, 8].map((x) => (
        <text key={x} x={sx(x)} y={H - 16} fontSize={11} fill="var(--color-neutral-600)"
          textAnchor="middle" fontFamily="var(--mono)">
          {x}
        </text>
      ))}
      <text x={(PAD_L + W - PAD_R) / 2} y={H - 2} fontSize={11}
        fill="var(--color-neutral-600)" textAnchor="middle" fontFamily="var(--font-body)">
        gradient accumulation steps
      </text>

      <path d={path} fill="none" stroke="var(--color-accent)" strokeWidth={2.4} />

      <line x1={sx(accum)} y1={PAD_T} x2={sx(accum)} y2={H - PAD_B}
        stroke="var(--color-accent-2)" strokeWidth={1.4} strokeDasharray="4 3" />
      <circle cx={sx(accum)} cy={sy(predictedHere)} r={5.5} fill="var(--color-accent-2)"
        stroke="var(--color-bg)" strokeWidth={2} />

      {POINTS.map((point) => {
        if (point.predicted && !revealed) return null;
        const y = sy(point.efficiency * 100);
        return (
          <g key={point.accum}>
            <circle
              cx={sx(point.accum)}
              cy={y}
              r={7}
              fill={point.predicted ? "var(--color-bg)" : "var(--color-text)"}
              stroke="var(--color-text)"
              strokeWidth={point.predicted ? 2.5 : 1}
            />
            <text
              x={sx(point.accum)}
              y={y - 14}
              fontSize={12}
              fill="var(--color-text)"
              textAnchor="middle"
              fontFamily="var(--mono)"
              fontWeight={600}
            >
              {(point.efficiency * 100).toFixed(1)}%
            </text>
            {point.predicted && (
              <text x={sx(point.accum)} y={y + 24} fontSize={11}
                fill="var(--color-neutral-600)" textAnchor="middle" fontFamily="var(--mono)">
                {residual(a, b, point.accum, point.efficiency) >= 0 ? "+" : ""}
                {residual(a, b, point.accum, point.efficiency).toFixed(2)} pts
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}
