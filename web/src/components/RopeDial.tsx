const SIZE = 92;
const R = 33;
const C = SIZE / 2;
/** The wedge is drawn well inside the arms, so both stay legible where they cross. */
const WEDGE_R = 20.5;

/** Wrap into (-π, π] so the drawn wedge is always the short way round. */
function wrap(a: number): number {
  const t = ((a + Math.PI) % (2 * Math.PI)) - Math.PI;
  return t <= -Math.PI ? t + 2 * Math.PI : t;
}

type Props = {
  index: number;
  angleQ: number;
  angleK: number;
  freq: number;
  contribution: number;
};

/**
 * One dimension pair, drawn as the 2-D rotation it actually is.
 *
 * The two arrows are where this pair of q's and k's components point after being
 * rotated by their own positions. The shaded wedge is the angle between them — and
 * that wedge is the entire mechanism: it depends on `m - n` and nothing else, so
 * sliding both tokens leaves it untouched while both arrows spin.
 */
export default function RopeDial({ index, angleQ, angleK, freq, contribution }: Props) {
  const pt = (a: number, r: number) => [C + r * Math.cos(a), C - r * Math.sin(a)] as const;
  const [qx, qy] = pt(angleQ, R);
  const [kx, ky] = pt(angleK, R);

  const delta = wrap(angleQ - angleK);
  const [wx, wy] = pt(angleK, WEDGE_R);
  const [wx2, wy2] = pt(angleK + delta, WEDGE_R);
  // y is inverted on screen, so a positive (counter-clockwise) delta is sweep=0.
  const wedge = `M ${C} ${C} L ${wx} ${wy} A ${WEDGE_R} ${WEDGE_R} 0 0 ${
    delta > 0 ? 0 : 1
  } ${wx2} ${wy2} Z`;

  const turnsPerToken = freq / (2 * Math.PI);
  const period = turnsPerToken > 0 ? 1 / turnsPerToken : Infinity;

  return (
    <div>
      <svg viewBox={`0 0 ${SIZE} ${SIZE}`} width="100%" role="img"
        aria-label={`Dimension pair ${index}, angle between q and k ${delta.toFixed(2)} radians`}>
        <circle cx={C} cy={C} r={R} fill="var(--color-bg)" stroke="var(--color-neutral-300)" />
        <path d={wedge} fill="var(--color-accent)" opacity={0.2} />
        <line x1={C} y1={C} x2={kx} y2={ky} stroke="var(--color-accent-2)" strokeWidth={2.2}
          strokeLinecap="round" />
        <line x1={C} y1={C} x2={qx} y2={qy} stroke="var(--color-accent)" strokeWidth={2.2}
          strokeLinecap="round" />
        <circle cx={C} cy={C} r={2} fill="var(--color-neutral-700)" />
      </svg>
      <div className="dial-label">
        pair {index}
        <br />
        {period === Infinity
          ? "static"
          : period < 1000
            ? `${period.toFixed(period < 10 ? 1 : 0)} tok/turn`
            : `${(period / 1000).toFixed(0)}k tok/turn`}
        <br />
        {/* Categorical, not semantic: a positive contribution to a logit is not
            "good", so this is the page's cyan/magenta pairing, not a red/green one. */}
        <span className={contribution >= 0 ? "dial-positive" : "dial-negative"}>
          {contribution >= 0 ? "+" : ""}
          {contribution.toFixed(3)}
        </span>
      </div>
    </div>
  );
}
