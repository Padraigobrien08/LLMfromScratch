import { useState } from "react";

import DataTable from "../components/DataTable";
import CacheSweep from "../components/CacheSweep";
import PlateNumeral from "../components/PlateNumeral";
import PlateFoot from "../components/PlateFoot";
import Caveat from "../components/Caveat";
import { MEASURED } from "../content/measured";
import { plateHeadline, plateKicker } from "../content/path";

const REPO = "https://github.com/Padraigobrien08/LLMfromScratch/blob/main";
const CACHE = MEASURED.cache;
const QUANT = MEASURED.quantization;
const SPEC = MEASURED.speculative;

export default function Efficiency() {
  const [masked, setMasked] = useState(true);
  const longest = CACHE.points.at(-1)!;

  return (
    <div className="shell page">
      <p className="kicker">{plateKicker("efficiency")}</p>
      <h1 className="page-headline">{plateHeadline("efficiency")}</h1>
      <p className="page-standfirst">
        Three optimisations, all hand-implemented, all measured. The theme running through them is
        that the headline number and the useful number are rarely the same one: 4-bit quantization
        is a memory win that costs speed, speculative decoding can hit its algorithmic ceiling and
        still lose on the clock, and the KV cache — the one optimisation nobody thinks to question —
        was for a while making decoding <i>slower</i>.
      </p>

      {/* The plate's opening trio, in the grammar Reproduction and Scaling already use.
          Every figure is read from `measured.ts` — the gain the fix bought at the longest
          sequence, the ceiling compression runs into, and the best speculative run. */}
      <div className="figure-strip">
        <figure>
          <PlateNumeral value={`${longest.gainFromFix.toFixed(2)}×`} />
          <figcaption>
            more decode throughput after the mask fix, at {longest.totalLen} tokens
          </figcaption>
        </figure>
        <figure>
          <PlateNumeral
            value={`${Math.max(...QUANT.schemes.map((s) => s.compression)).toFixed(2)}×`}
          />
          <figcaption>
            the compression ceiling, not 8× — the tied embedding stays in fp32
          </figcaption>
        </figure>
        <figure>
          <PlateNumeral value={`${SPEC.best.speedup.toFixed(2)}×`} />
          <figcaption>
            the best speculative run, {SPEC.best.drafter} on a {SPEC.best.prompt} prompt
          </figcaption>
        </figure>
      </div>

      <div className="rule-heavy" />
      <h2 className="section-h2">The KV cache, and the story that hid a bug</h2>
      <div className="two-col">
        <p className="prose">
          An earlier version of this project reported that the KV cache gave no speedup, and
          explained it as a property rather than a defect: decoding a 124M model is bound by
          streaming weights from memory, not by attention over the prefix. That is a real effect,
          and it is why the cache curve below is nearly flat. It was also the wrong explanation for
          the number in front of me, and its plausibility is exactly what stopped me looking
          further.
        </p>
        <p className="prose">
          The sweep that was supposed to <i>confirm</i> the story broke it. A cache that does
          strictly less arithmetic cannot lose on work; it can only lose on overhead. Throughput
          flat at ~170 tok/s regardless of length is the signature of a fixed per-step cost, not of
          attention. The cause was three lines: a decode step has one query, so it took the branch
          that builds an explicit causal mask — and passing a mask to SDPA disqualifies it from the
          fused flash kernels. For a single query every cached key precedes it, so that mask was
          all-true. Pure cost, no information.
        </p>
      </div>

      <div className="figure-panel" style={{ marginTop: "var(--space-4)" }}>
        <h3 className="figure-title">Decode throughput against sequence length, {CACHE.gpu}</h3>
        <p className="fig-note" style={{ margin: "0 0 var(--space-3)" }}>
          Both curves are measured. Untick the box to remove the mask and watch the cache travel
          from losing at every length to winning at the longest — the same code, one branch apart.
        </p>
        <CacheSweep masked={masked} onMasked={setMasked} />

        <DataTable label="Cache versus recompute by sequence length" style={{ marginTop: "var(--space-4)" }}>
          <thead>
            <tr>
              <th>Total length</th>
              <th className="num">Recompute</th>
              <th className="num">KV cache</th>
              <th className="num">Cache advantage</th>
              <th className="num">Gain from the fix</th>
            </tr>
          </thead>
          <tbody>
            {CACHE.points.map((p) => {
              const cached = masked ? p.cachedBefore : p.cached;
              const naive = masked ? p.naiveBefore : p.naive;
              const advantage = masked ? p.advantageBefore : p.advantage;
              return (
                <tr key={p.totalLen}>
                  <td className="mono">{p.totalLen}</td>
                  <td className="num mono">{naive.toFixed(0)}</td>
                  <td className="num mono">{cached.toFixed(0)}</td>
                  <td className={`num mono ${advantage >= 1 ? "verdict-better" : "verdict-noise"}`}>
                    {advantage.toFixed(2)}×
                  </td>
                  <td className="num mono">{p.gainFromFix.toFixed(2)}×</td>
                </tr>
              );
            })}
          </tbody>
        </DataTable>
        <p className="fig-note">
          The recompute column moves {Math.min(...CACHE.points.map((p) => p.naive / p.naiveBefore)).toFixed(2)}–
          {Math.max(...CACHE.points.map((p) => p.naive / p.naiveBefore)).toFixed(2)}× between the two
          runs — the untouched control that makes this a measurement rather than a coincidence.
          Measured at <code className="mono">{CACHE.commitBefore.slice(0, 8)}</code> and{" "}
          <code className="mono">{CACHE.commitAfter.slice(0, 8)}</code>.
        </p>
      </div>

      <Caveat columns>
        The original explanation was directionally right and quantitatively wrong: decode at this
        scale <i>is</i> overhead-bound, which is why the cache curve is flat — but the crossover is
        real and lands at {longest.totalLen} tokens, exactly this model's context length. What broke
        the story open was the shape of the data rather than the headline, and{" "}
        <b>a plausible story for a disappointing measurement is the most expensive kind of mistake</b>,
        because it converts a bug into a finding and closes the investigation. The gap was in the
        tests too: every test asserted the cache produced the right <i>answer</i>, none that it took
        the fast <i>path</i>, so a 30% regression passed a green suite. Two now do, both
        mutation-checked.
      </Caveat>

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">Quantization, and where the compression stops</h2>
      <DataTable label="Quantization schemes">
        <thead>
          <tr>
            <th>Scheme</th>
            <th className="num">Memory</th>
            <th className="num">Compression</th>
            <th className="num">Perplexity</th>
            <th className="num">Δ ppl</th>
            <th className="num">Decode</th>
          </tr>
        </thead>
        <tbody>
          {QUANT.schemes.map((s) => (
            <tr key={s.name}>
              <td className="mono">{s.name}</td>
              <td className="num mono">{s.memoryMib.toFixed(0)} MiB</td>
              <td className="num mono">{s.compression.toFixed(2)}×</td>
              <td className="num mono">{s.perplexity.toFixed(3)}</td>
              <td className="num mono">
                {s.deltaPerplexity === 0 ? "—" : (s.deltaPerplexity >= 0 ? "+" : "") + s.deltaPerplexity.toFixed(3)}
              </td>
              <td className="num mono">{s.decodeTokS.toFixed(1)}</td>
            </tr>
          ))}
        </tbody>
      </DataTable>
      <Caveat columns>
        Grouping is worth two perplexity points at four bits: one scale per tensor is set by its
        largest outlier, and per-128-feature groups confine the damage. Every scheme is{" "}
        <i>slower</i> than fp32, because dequantize-then-matmul materialises a full-size weight, so
        bytes moved go up — the saving is in what is stored, not in what is moved. A fused kernel is
        what would turn one into the other, and it is the first item in{" "}
        <a href={`${REPO}/docs/roadmap.md`}>the roadmap</a>. Compression also caps at{" "}
        {Math.max(...QUANT.schemes.map((s) => s.compression)).toFixed(2)}×, not 8×, because the tied
        token embedding is a third of this model and is left in fp32.
      </Caveat>

      <div className="rule-hair" style={{ margin: "var(--space-6) 0" }} />
      <h2 className="section-h2">Speculative decoding, and why acceptance is not speedup</h2>
      <DataTable label="Speculative decoding results">
        <thead>
          <tr>
            <th>Drafter</th>
            <th>Prompt</th>
            <th className="num">k</th>
            <th className="num">Speedup</th>
            <th className="num">Acceptance</th>
            <th className="num">Tokens / target fwd</th>
          </tr>
        </thead>
        <tbody>
          {[...SPEC.rows]
            .sort((x, y) => y.speedup - x.speedup)
            .slice(0, 6)
            .map((r) => (
              <tr key={`${r.drafter}-${r.prompt}-${r.k}`}>
                <td className="mono">{r.drafter}</td>
                <td className="status-state">{r.prompt}</td>
                <td className="num mono">{r.k}</td>
                <td className="num mono">{r.speedup.toFixed(2)}×</td>
                <td className="num mono">
                  {r.acceptanceRate == null ? "—" : `${(r.acceptanceRate * 100).toFixed(1)}%`}
                </td>
                <td className="num mono">
                  {r.tokensPerTargetForward == null ? "—" : r.tokensPerTargetForward.toFixed(2)}
                </td>
              </tr>
            ))}
        </tbody>
      </DataTable>
      <Caveat columns>
        All {SPEC.losslessRuns} benchmark runs reproduced greedy decoding token for token, and{" "}
        {SPEC.divergedRuns === 0 ? "none diverged" : `${SPEC.divergedRuns} diverged`} — without
        that, a speedup would just be a different model running faster. The row worth reading twice
        is the model drafter: essentially the algorithmic ideal, nearly every proposal accepted, and
        barely any speedup at all, because the drafter is the same size as the target.{" "}
        <b>A drafter must be cheap first and accurate second</b>, and acceptance rate measures only
        the second.
      </Caveat>

      <div className="rule-heavy" style={{ margin: "var(--space-6) 0 var(--space-4)" }} />
      <div className="closing-cols">
        <p style={{ font: "400 17px/1.6 var(--font-body)" }}>
          <b>Throughput is a property of a device-algorithm pair, not of an algorithm.</b> Moving
          from MPS to CUDA flipped the sign of the prose speculative result, so the earlier claim
          that prompt-lookup "loses on prose" turned out to be a statement about MPS. Memory and
          quality are device-independent; the numbers above that are not carry their hardware with
          them.
        </p>
        <p style={{ font: "400 16px/1.6 var(--font-body)", color: "var(--color-neutral-700)" }}>
          Full write-up, including the training-throughput sweep on both an H100 and a 4090 and what
          that contrast says about gradient checkpointing, in{" "}
          <a href={`${REPO}/docs/efficiency.md`}>
            <code>docs/efficiency.md</code>
          </a>
          .
        </p>
      </div>

      <PlateFoot current="efficiency" />
    </div>
  );
}
