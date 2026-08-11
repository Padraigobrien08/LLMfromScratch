import type { Comparison } from "../lib/ablations";

const W = 1000;
const H = 96;
const PAD_L = 16;
const PAD_R = 16;
const MID = 46;

/**
 * The significance rule, drawn rather than asserted.
 *
 * An arm counts as a result only when the range of its per-seed deltas does not straddle
 * zero — every seed agreed on the direction. That is the whole verdict, and it is a
 * geometric fact about three ticks and a line, so it should be a picture: a bar spanning
 * the seeds, a rule at zero, and the single question of whether the two touch.
 *
 * Written this way on purpose instead of as a p-value. With three seeds nothing stronger
 * would be honest, and a reader can check this rule by looking, which is not true of a
 * test statistic.
 */
export default function SeedDeltas({ row, scale }: { row: Comparison; scale: number }) {
  const sx = (delta: number) =>
    (PAD_L + W - PAD_R) / 2 + (delta / scale) * ((W - PAD_L - PAD_R) / 2);

  const lo = Math.min(...row.deltas);
  const hi = Math.max(...row.deltas);
  const straddles = lo <= 0 && hi >= 0;
  const better = (row.delta ?? 0) < 0;
  const colour = straddles
    ? "var(--color-neutral-500)"
    : better
      ? "var(--color-accent)"
      : "var(--color-accent-2)";

  const ticks = [-scale, -scale / 2, 0, scale / 2, scale];

  return (
    <svg
      className="seed-deltas"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Per-seed deltas for ${row.name}: ${row.deltas.map((d) => d.toFixed(4)).join(", ")}. ${
        straddles ? "The range straddles zero, so this is not a result." : "Every seed agreed on the direction."
      }`}
    >
      {ticks.map((t) => (
        <g key={t}>
          <line x1={sx(t)} y1={MID - 26} x2={sx(t)} y2={MID + 26}
            stroke={t === 0 ? "var(--color-text)" : "var(--color-neutral-200)"}
            strokeWidth={t === 0 ? 1.5 : 1} />
          <text x={sx(t)} y={H - 6} fontSize={11} fill="var(--color-neutral-600)"
            textAnchor="middle" fontFamily="var(--mono)">
            {t === 0 ? "0" : (t > 0 ? "+" : "") + t.toFixed(2)}
          </text>
        </g>
      ))}

      <line x1={sx(lo)} y1={MID} x2={sx(hi)} y2={MID} stroke={colour} strokeWidth={9}
        strokeLinecap="round" opacity={0.28} />

      {row.deltas.map((delta, i) => (
        <line key={i} x1={sx(delta)} y1={MID - 15} x2={sx(delta)} y2={MID + 15}
          stroke={colour} strokeWidth={2.5} strokeLinecap="round" />
      ))}

      {row.delta != null && (
        <circle cx={sx(row.delta)} cy={MID} r={6} fill={colour}
          stroke="var(--color-bg)" strokeWidth={2} />
      )}

      <text x={PAD_L} y={16} fontSize={11} fill="var(--color-neutral-600)"
        fontFamily="var(--mono)">
        better ←
      </text>
      <text x={W - PAD_R} y={16} fontSize={11} fill="var(--color-neutral-600)"
        textAnchor="end" fontFamily="var(--mono)">
        → worse
      </text>
    </svg>
  );
}
