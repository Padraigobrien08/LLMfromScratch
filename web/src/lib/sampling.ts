/**
 * Temperature, top-k and top-p, in the order `Transformer.generate` applies them.
 *
 * The order is not cosmetic. Temperature is applied to the logits *before* the
 * cutoffs, so raising it widens the distribution the cutoffs then act on; and top-p
 * is evaluated on the already-truncated top-k set. Any other ordering gives
 * different text from the same seed, so the page would be teaching a sampler the
 * repository does not have.
 *
 * The top-p rule mirrors the subtlety in `transformer.py:230`: a token is kept when
 * the cumulative mass *before* it is under the threshold, so the token that crosses
 * the threshold is kept rather than dropped. Excluding it makes `top_p = 0` sample
 * nothing at all.
 */

export type Candidate = { id: number; text: string; count: number };

export type Scored = Candidate & {
  /** Probability before any cutoff, at the current temperature. */
  raw: number;
  /** Probability after cutoffs and renormalisation; zero when dropped. */
  prob: number;
  kept: boolean;
};

export type SamplingOptions = {
  temperature: number;
  topK: number | null;
  topP: number | null;
};

function softmax(logits: number[], temperature: number): number[] {
  // Greedy decoding is the limit of temperature → 0, and the real implementation
  // special-cases it rather than dividing by zero.
  if (temperature <= 0) {
    const best = logits.indexOf(Math.max(...logits));
    return logits.map((_, i) => (i === best ? 1 : 0));
  }
  const scaled = logits.map((l) => l / temperature);
  const max = Math.max(...scaled);
  const exp = scaled.map((l) => Math.exp(l - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map((e) => e / sum);
}

/**
 * Score candidates whose evidence is a count.
 *
 * Counts become logits as `log(count)`: softmax of the log of a count *is* the
 * empirical frequency, so at temperature 1 the distribution shown is exactly the
 * corpus statistic, with no fitting or smoothing in between.
 */
export function scoreCandidates(candidates: Candidate[], options: SamplingOptions): Scored[] {
  const logits = candidates.map((c) => Math.log(c.count));
  const raw = softmax(logits, options.temperature);

  const order = raw
    .map((p, i) => ({ p, i }))
    .sort((a, b) => b.p - a.p);

  const kept = new Array<boolean>(candidates.length).fill(true);

  if (options.topK !== null) {
    const k = Math.max(0, Math.min(options.topK, candidates.length));
    order.forEach((entry, rank) => {
      if (rank >= k) kept[entry.i] = false;
    });
  }

  if (options.topP !== null) {
    // Renormalise across whatever top-k left, then walk down until the mass before
    // the current token has already reached the threshold.
    const survivingMass = order.reduce((sum, e) => sum + (kept[e.i] ? e.p : 0), 0) || 1;
    let cumulative = 0;
    for (const entry of order) {
      if (!kept[entry.i]) continue;
      const p = entry.p / survivingMass;
      if (cumulative > options.topP) kept[entry.i] = false;
      cumulative += p;
    }
  }

  const total = raw.reduce((sum, p, i) => sum + (kept[i] ? p : 0), 0);
  return candidates.map((c, i) => ({
    ...c,
    raw: raw[i]!,
    prob: kept[i] && total > 0 ? raw[i]! / total : 0,
    kept: kept[i]!,
  }));
}

/** Draw one candidate, using a caller-supplied uniform so demos stay reproducible. */
export function sample(scored: Scored[], uniform: number): Scored | null {
  let cumulative = 0;
  for (const s of scored) {
    cumulative += s.prob;
    if (uniform < cumulative) return s;
  }
  return scored.filter((s) => s.kept).at(-1) ?? null;
}

/**
 * Perplexity: the loss expressed as "how many equally-likely options is the model
 * effectively choosing between?".
 *
 * This is the translation that makes a validation loss mean something. A loss of
 * 3.29 is not obviously good or bad; "as uncertain as picking from 27 equally
 * likely tokens, out of 50,257" is.
 */
export const perplexity = (loss: number) => Math.exp(loss);

/** The loss a model would get by guessing uniformly among `vocabSize` tokens. */
export const uniformLoss = (vocabSize: number) => Math.log(vocabSize);
