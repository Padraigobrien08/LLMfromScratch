import { useId, useMemo } from "react";

import { type Curve, thin, valAt } from "../lib/reproductionCurve";

const W = 1000;
const H = 340;
const PAD_L = 56;
const PAD_R = 18;
const PAD_T = 16;
const PAD_B = 38;

// The same window `results/reproduction_curve.png` uses. The run starts at a loss of
// 10.95 and spends its first two hundred steps falling through territory no one needs
// to see at this resolution; drawing all of it would compress everything the reader is
// actually here for — the approach to 3.29 — into the bottom eighth of the frame.
const Y_LO = 2.95;
const Y_HI = 6.0;

type Props = {
  curve: Curve;
  step: number;
  onStep: (step: number) => void;
};

/**
 * The run, with the target drawn across it and a scrubber you drag along it.
 *
 * The interaction is the argument. A target quoted in prose is a claim; a target drawn
 * as a line the curve visibly crosses a third of the way in, with the run continuing to
 * improve for the remaining two thirds, is the same claim in a form that cannot be
 * arranged after the fact. Everything to the right of the scrubber is dimmed, so
 * dragging reads as advancing the run rather than as inspecting a finished picture.
 */
export default function LossCurve({ curve, step, onStep }: Props) {
  const clipId = useId();

  const { trainPath, valPath } = useMemo(() => {
    const sx = (s: number) => PAD_L + (s / curve.finalStep) * (W - PAD_L - PAD_R);
    const sy = (v: number) =>
      PAD_T + (1 - (v - Y_LO) / (Y_HI - Y_LO)) * (H - PAD_T - PAD_B);
    const path = (points: Array<{ step: number; loss: number }>) =>
      points.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.step)} ${sy(p.loss)}`).join(" ");
    return {
      trainPath: path(thin(curve.train, 900)),
      valPath: path(curve.val),
    };
  }, [curve]);

  const sx = (s: number) => PAD_L + (s / curve.finalStep) * (W - PAD_L - PAD_R);
  const sy = (v: number) => PAD_T + (1 - (v - Y_LO) / (Y_HI - Y_LO)) * (H - PAD_T - PAD_B);

  const here = valAt(curve.val, step);
  const gridYs = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0];
  const gridXs = [0, 5000, 10_000, 15_000, curve.finalStep];

  return (
    <svg
      className="loss-curve"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Validation loss against training step, with the pre-registered target of ${curve.targetLoss} drawn across it`}
    >
      <defs>
        <clipPath id={clipId}>
          <rect x={PAD_L} y={PAD_T} width={W - PAD_L - PAD_R} height={H - PAD_T - PAD_B} />
        </clipPath>
      </defs>

      {gridYs.map((y) => (
        <g key={y}>
          <line x1={PAD_L} y1={sy(y)} x2={W - PAD_R} y2={sy(y)} stroke="var(--color-neutral-200)" />
          <text
            x={PAD_L - 8}
            y={sy(y) + 4}
            fontSize={11}
            fill="var(--color-neutral-600)"
            textAnchor="end"
            fontFamily="var(--mono)"
          >
            {y.toFixed(1)}
          </text>
        </g>
      ))}
      {gridXs.map((x) => (
        <text
          key={x}
          x={sx(x)}
          y={H - 14}
          fontSize={11}
          fill="var(--color-neutral-600)"
          textAnchor="middle"
          fontFamily="var(--mono)"
        >
          {x.toLocaleString()}
        </text>
      ))}

      <g clipPath={`url(#${clipId})`}>
        <path d={trainPath} fill="none" stroke="var(--color-accent-300)" strokeWidth={1} />
        <path
          d={valPath}
          fill="none"
          stroke="var(--color-text)"
          strokeWidth={2.6}
          strokeLinejoin="round"
        />

        {/* The pre-registered target. Drawn under nothing and over everything else,
            because it is the only line on the chart that was fixed before the run. */}
        <line
          x1={PAD_L}
          y1={sy(curve.targetLoss)}
          x2={W - PAD_R}
          y2={sy(curve.targetLoss)}
          stroke="var(--color-accent-2)"
          strokeWidth={1.8}
          strokeDasharray="7 4"
        />
        <text
          x={W - PAD_R - 6}
          y={sy(curve.targetLoss) - 7}
          fontSize={12}
          fill="var(--color-accent-2-700)"
          textAnchor="end"
          fontFamily="var(--mono)"
        >
          target {curve.targetLoss}
        </text>

        {curve.crossing && (
          <line
            x1={sx(curve.crossing.step)}
            y1={PAD_T}
            x2={sx(curve.crossing.step)}
            y2={H - PAD_B}
            stroke="var(--color-accent-2-300)"
            strokeWidth={1}
          />
        )}

        {/* Everything not yet reached. Dimming it is what turns a static chart into a
            run in progress: the reader is moving through it, not looking back at it. */}
        <rect
          x={sx(step)}
          y={PAD_T}
          width={Math.max(0, W - PAD_R - sx(step))}
          height={H - PAD_T - PAD_B}
          fill="var(--color-bg)"
          opacity={0.72}
        />
      </g>

      <line
        x1={sx(step)}
        y1={PAD_T}
        x2={sx(step)}
        y2={H - PAD_B}
        stroke="var(--color-accent)"
        strokeWidth={1.5}
      />
      {here && (
        <circle
          cx={sx(here.step)}
          cy={sy(here.loss)}
          r={5}
          fill="var(--color-accent)"
          stroke="var(--color-bg)"
          strokeWidth={2}
        />
      )}

      <rect
        x={PAD_L}
        y={PAD_T}
        width={W - PAD_L - PAD_R}
        height={H - PAD_T - PAD_B}
        fill="transparent"
        style={{ cursor: "ew-resize" }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          scrub(e, curve.finalStep, onStep);
        }}
        onPointerMove={(e) => {
          if (e.buttons === 1) scrub(e, curve.finalStep, onStep);
        }}
      />
    </svg>
  );
}

/** Map a pointer position on the plot area back to a training step. */
function scrub(
  event: React.PointerEvent<SVGRectElement>,
  finalStep: number,
  onStep: (step: number) => void,
) {
  const box = event.currentTarget.getBoundingClientRect();
  const fraction = (event.clientX - box.left) / box.width;
  onStep(Math.round(Math.min(1, Math.max(0, fraction)) * finalStep));
}
