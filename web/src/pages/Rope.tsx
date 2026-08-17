import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import DecayPlot from "../components/DecayPlot";
import PositionRuler from "../components/PositionRuler";
import RopeDial from "../components/RopeDial";
import { dotRotated, invFreqs, pairAngles, pairContributions, seededVector } from "../lib/rope";

const MAX_POS = 512;
const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";

type Tracker = { min: number; max: number; count: number };

export default function Rope() {
  const [m, setM] = useState(24);
  const [n, setN] = useState(8);
  const [headDim, setHeadDim] = useState(16);
  const [theta, setTheta] = useState(10_000);
  const [seed, setSeed] = useState(7);
  const [gapLocked, setGapLocked] = useState(true);
  const [playing, setPlaying] = useState(false);
  // 128 by default: pair 0 turns once every ~6 tokens, so a 512-wide axis packs
  // eighty oscillations into a thousand pixels and reads as noise rather than as
  // the structure it is.
  const [plotRange, setPlotRange] = useState(128);

  const q = useMemo(() => seededVector(headDim, seed), [headDim, seed]);
  const k = useMemo(() => seededVector(headDim, seed + 1000), [headDim, seed]);

  const d = m - n;
  const logit = dotRotated(q, k, m, n, theta);

  /**
   * The assertion, run live.
   *
   * `tests/test_rope.py` claims the logit depends only on `m - n`. Rather than
   * restate that claim in prose, this accumulates the range the logit actually
   * occupies while the pair slides, and prints the spread. If the claim were false
   * the number would climb visibly; it stays at float64 noise.
   */
  const [tracker, setTracker] = useState<Tracker>({ min: logit, max: logit, count: 1 });
  const resetKey = `${d}|${headDim}|${theta}|${seed}`;
  const lastResetKey = useRef(resetKey);

  useEffect(() => {
    if (lastResetKey.current !== resetKey) {
      lastResetKey.current = resetKey;
      setTracker({ min: logit, max: logit, count: 1 });
    } else {
      setTracker((t) => ({ min: Math.min(t.min, logit), max: Math.max(t.max, logit), count: t.count + 1 }));
    }
  }, [resetKey, logit]);

  const step = useCallback(() => {
    setN((prevN) => {
      const gap = d;
      const nextN = prevN + 1;
      // Wrap the pair as a unit once the leading token runs off the end.
      const wrapped = nextN + Math.max(gap, 0) > MAX_POS ? Math.max(-Math.min(gap, 0), 0) : nextN;
      setM(wrapped + gap);
      return wrapped;
    });
  }, [d]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(step, 45);
    return () => window.clearInterval(id);
  }, [playing, step]);

  // Playing only makes sense with the gap held: the whole point is that one quantity
  // is pinned while both absolute positions change.
  useEffect(() => {
    if (playing && !gapLocked) setPlaying(false);
  }, [playing, gapLocked]);

  const angQ = pairAngles(m, headDim, theta);
  const angK = pairAngles(n, headDim, theta);
  const contribs = pairContributions(q, k, d, theta);
  const freqs = invFreqs(headDim, theta);

  const spread = tracker.max - tracker.min;

  return (
    <div className="shell page">
      <p className="kicker">The deep end · explorable explanation</p>
      <h1 className="page-headline">What rotary embeddings actually do</h1>
      <p className="page-standfirst">
        RoPE encodes a token's position by <i>rotating</i> its query and key vectors by an angle
        proportional to that position. The payoff is that the attention logit between them ends up
        depending only on how far apart they are — never on where either one sits. That claim is
        easy to write and hard to believe. So move them and watch.
      </p>

      <div className="figure-panel" style={{ marginBottom: "var(--space-6)" }}>
        {/* h2: this figure is a sibling of the two sections below it, not a subsection of
            anything — as an h3 it left a level skipped straight after the headline, which
            is a hole in the outline a screen-reader user navigates by. */}
        <h2 className="figure-title">Two tokens, one sequence</h2>
        <p className="fig-note" style={{ margin: "0 0 var(--space-2)" }}>
          Drag <b>Q</b> or <b>K</b>, or focus one and use the arrow keys (hold shift for ten). With
          the gap held, they move as a rigid pair.
        </p>

        <div className="ruler-scroll">
          <PositionRuler
            maxPos={MAX_POS}
            m={m}
            n={n}
            gapLocked={gapLocked}
            onChange={(nm, nn) => {
              setM(nm);
              setN(nn);
            }}
          />
        </div>

        <div className="fig-row fig-row-wide" style={{ marginTop: "var(--space-3)", marginBottom: 0 }}>
          <button className="btn btn-primary" onClick={() => setPlaying((p) => !p)} disabled={!gapLocked}>
            {playing ? "Pause" : "Slide the pair"}
          </button>
          <label className="field field-inline">
            <input type="checkbox" checked={gapLocked} onChange={(e) => setGapLocked(e.target.checked)} />
            hold the gap at {d}
          </label>
          <label className="field field-inline">
            head dim
            <select className="input input-sm" value={headDim}
              onChange={(e) => setHeadDim(Number(e.target.value))}>
              <option value={16}>16</option>
              <option value={32}>32</option>
              <option value={64}>64</option>
            </select>
          </label>
          <label className="field field-inline">
            θ
            <select className="input input-sm" value={theta}
              onChange={(e) => setTheta(Number(e.target.value))}>
              <option value={10_000}>10,000 (GPT-NeoX, Llama 2)</option>
              <option value={500_000}>500,000 (Llama 3, long context)</option>
            </select>
          </label>
          <button className="btn btn-secondary" onClick={() => setSeed((s) => s + 1)}>
            New q, k
          </button>
        </div>
      </div>

      <div className="readouts">
        <div>
          <p className="eyebrow">Attention logit ⟨R(q,m), R(k,n)⟩</p>
          <div className="readout-xl">{logit.toFixed(6)}</div>
          <div className="readout-pair">
            <div>
              <p className="eyebrow">Offset m − n</p>
              <div className="readout-md">{d}</div>
            </div>
            <div>
              <p className="eyebrow">Positions</p>
              <div className="readout-md">
                {m}, {n}
              </div>
            </div>
          </div>
        </div>

        <div>
          <p className="eyebrow" style={{ color: "var(--color-accent-2-700)" }}>
            The assertion, running live
          </p>
          <div className={`readout-xl ${spread < 1e-9 ? "spread-holding" : "spread-moving"}`}>
            {spread.toExponential(1)}
          </div>
          <p className="tracker-sentence">
            Total range of the logit across <b>{tracker.count}</b>{" "}
            {tracker.count === 1 ? "sample" : "samples"} at offset <b>{d}</b>, spanning absolute
            positions up to {MAX_POS}. Press <i>Slide the pair</i> and watch it stay at
            floating-point noise while both positions change by hundreds. Change the offset and the
            counter resets.
          </p>
        </div>
      </div>

      <div className="rule-hair" style={{ marginBottom: "var(--space-6)" }} />
      <h2 className="section-h2">The mechanism, one dial per dimension pair</h2>
      <p className="note-wide">
        The head dimension is split in half and dimension <code>i</code> is paired with{" "}
        <code>i + headDim/2</code> — the split-half convention from GPT-NeoX, which is what this
        repository implements. Each pair is rotated by its own angle. Cyan is <b>q</b>, magenta is{" "}
        <b>k</b>; the shaded wedge is the angle between them. Slide the pair and every arrow spins
        while every wedge holds.
      </p>

      <div className="dials">
        {Array.from({ length: headDim / 2 }, (_, i) => (
          <RopeDial
            key={i}
            index={i}
            angleQ={angQ[i]!}
            angleK={angK[i]!}
            freq={freqs[i]!}
            contribution={contribs[i]!}
          />
        ))}
      </div>
      <p className="caveat-wide" style={{ marginBottom: "var(--space-8)" }}>
        Pair 0 turns a full revolution every ~6 tokens; the last pair takes longer than any context
        you will ever run. Fast pairs resolve neighbours, slow pairs carry long-range order, and the
        sum of their contributions is the logit above.
      </p>

      <div className="rule-hair" style={{ marginBottom: "var(--space-6)" }} />
      <h2 className="section-h2" style={{ marginBottom: "var(--space-4)" }}>
        The logit is a function of distance alone
      </h2>
      <div className="fig-row fig-row-wide" style={{ marginBottom: "var(--space-2)" }}>
        <label className="field field-inline">
          offsets shown
          <select className="input input-sm" value={plotRange}
            onChange={(e) => setPlotRange(Number(e.target.value))}>
            <option value={64}>0 – 64</option>
            <option value={128}>0 – 128</option>
            <option value={512}>0 – 512</option>
          </select>
        </label>
        <span style={{ font: "400 14px/1.5 var(--font-body)", color: "var(--color-neutral-700)" }}>
          horizontal axis: relative offset |m − n|; vertical: the logit
        </span>
      </div>
      <DecayPlot q={q} k={k} theta={theta} d={d} maxD={plotRange} />
      <p className="caveat-wide" style={{ margin: "var(--space-3) 0 var(--space-8)" }}>
        This curve is computed from the offset directly, without ever forming an absolute position —
        the closed form in <code>dotRelative</code>. The marker is your current offset. Note that it
        oscillates rather than decaying smoothly: it is a sum of cosines at {headDim / 2} different
        rates, which is also why raising θ to 500,000 stretches the whole picture out and buys
        long-context resolution.
      </p>

      <div className="rule-heavy" style={{ marginBottom: "var(--space-4)" }} />
      <div className="closing-cols">
        <p style={{ font: "400 17px/1.6 var(--font-body)" }}>
          <b>This page runs the real thing.</b> The rotation above is a TypeScript port of{" "}
          <a href={`${REPO}/src/llmfs/model/rope.py`}>
            <code>src/llmfs/model/rope.py</code>
          </a>
          , and a fixture generated from the Python implementation pins the two together in both
          directions — the browser tests assert the port reproduces it, and the Python suite asserts
          the fixture still reproduces the model. Either side changing alone fails CI.
        </p>
        <p style={{ font: "400 16px/1.6 var(--font-body)", color: "var(--color-neutral-700)" }}>
          One measured caveat, since the page invites you to read small numbers: the Python builds
          its cos/sin tables in float32 while this port uses float64, so the two agree to about 1e-8
          near the start of a sequence and 3e-7 by position 200 — far below the ~1e-2 resolution of
          the bf16 activations that consume them, and the same choice HF Llama makes.
        </p>
      </div>
    </div>
  );
}
