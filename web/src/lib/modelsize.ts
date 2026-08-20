/**
 * Where a transformer's parameters, memory and arithmetic actually go.
 *
 * Mirrors the shapes built by `src/llmfs/model/` — `transformer.py` for the
 * embeddings and head, `block.py` for the two norms per layer, `attention.py` for
 * the fused QKV projection, and `mlp.py` for the GELU/SwiGLU split. A fixture
 * generated from the real `Transformer` asserts the totals agree exactly, because
 * a calculator that is merely close is a calculator that teaches the wrong thing
 * about where the budget goes.
 */

import { ARCHITECTURES } from "../content/architecture";

export type SizeConfig = {
  vocabSize: number;
  nLayer: number;
  nHead: number;
  nKvHead: number;
  nEmbd: number;
  blockSize: number;
  norm: "layernorm" | "rmsnorm";
  posEmb: "learned" | "rope" | "none";
  mlp: "gelu" | "swiglu";
  tieEmbeddings: boolean;
  bias: boolean;
  mlpRatio: number;
  mlpHiddenMultipleOf: number;
};

/**
 * The two shipped configs, taken from the generated `architecture.ts` rather than
 * typed. They used to be literals here — thirteen fields each, two lines away from
 * the YAML-resolved copies the Architecture page also reads — and nothing asserted
 * the twins agreed: a changed `n_embd` in the YAML would have printed the new shape
 * beside the old parameter count with a green suite. Field-picking from the resolved
 * config makes that disagreement unrepresentable, and `tsc` holds the field list.
 */
const fromResolved = (c: (typeof ARCHITECTURES)[keyof typeof ARCHITECTURES]["config"]): SizeConfig => ({
  vocabSize: c.vocabSize,
  nLayer: c.nLayer,
  nHead: c.nHead,
  nKvHead: c.nKvHead,
  nEmbd: c.nEmbd,
  blockSize: c.blockSize,
  norm: c.norm,
  posEmb: c.posEmb,
  mlp: c.mlp,
  tieEmbeddings: c.tieEmbeddings,
  bias: c.bias,
  mlpRatio: c.mlpRatio,
  mlpHiddenMultipleOf: c.mlpHiddenMultipleOf,
});

export const GPT2_124M: SizeConfig = fromResolved(ARCHITECTURES.gpt2.config);

export const LLAMA_124M: SizeConfig = fromResolved(ARCHITECTURES.llama.config);

export const headDim = (c: SizeConfig) => Math.floor(c.nEmbd / c.nHead);

/**
 * Feed-forward hidden width, including SwiGLU's 2/3 correction.
 *
 * SwiGLU has three projections rather than two, so at the naive 4x width it would
 * carry 1.5x the parameters and any comparison against GELU would be measuring the
 * parameter budget instead of the activation.
 */
export function mlpHidden(c: SizeConfig): number {
  let hidden = Math.trunc(c.mlpRatio * c.nEmbd);
  if (c.mlp === "swiglu") {
    hidden = Math.trunc((2 * hidden) / 3);
    const m = c.mlpHiddenMultipleOf;
    hidden = m * Math.ceil(hidden / m);
  }
  return hidden;
}

export type Breakdown = {
  tokenEmbedding: number;
  positionEmbedding: number;
  attention: number;
  feedForward: number;
  norms: number;
  lmHead: number;
  total: number;
};

const normParams = (c: SizeConfig) => (c.norm === "rmsnorm" ? c.nEmbd : c.nEmbd * (c.bias ? 2 : 1));

export function parameters(c: SizeConfig): Breakdown {
  const hd = headDim(c);
  const qSize = c.nHead * hd;
  const kvSize = c.nKvHead * hd;
  const hidden = mlpHidden(c);

  const tokenEmbedding = c.vocabSize * c.nEmbd;
  const positionEmbedding = c.posEmb === "learned" ? c.blockSize * c.nEmbd : 0;

  // One fused QKV GEMM, then the output projection.
  const qkv = c.nEmbd * (qSize + 2 * kvSize) + (c.bias ? qSize + 2 * kvSize : 0);
  const out = c.nEmbd * c.nEmbd + (c.bias ? c.nEmbd : 0);
  const attention = c.nLayer * (qkv + out);

  const ff =
    c.mlp === "swiglu"
      ? c.nEmbd * (2 * hidden) + (c.bias ? 2 * hidden : 0) + hidden * c.nEmbd + (c.bias ? c.nEmbd : 0)
      : c.nEmbd * hidden + (c.bias ? hidden : 0) + hidden * c.nEmbd + (c.bias ? c.nEmbd : 0);
  const feedForward = c.nLayer * ff;

  // Two per block (pre-attention and pre-MLP), plus the final one before the head.
  const norms = (2 * c.nLayer + 1) * normParams(c);

  // Tied weights are the same tensor as the token embedding, so they are not new
  // parameters. Untying adds a whole vocabulary-sized matrix: 38.6M at GPT-2 124M, which
  // is 5.45 blocks' worth, not the "two layers" this comment used to claim.
  const lmHead = c.tieEmbeddings ? 0 : c.nEmbd * c.vocabSize;

  return {
    tokenEmbedding,
    positionEmbedding,
    attention,
    feedForward,
    norms,
    lmHead,
    total: tokenEmbedding + positionEmbedding + attention + feedForward + norms + lmHead,
  };
}

/**
 * Bytes of key/value cache held while decoding, at `seqLen` tokens.
 *
 * The number that decides how many users fit on a GPU, and the entire argument for
 * grouped-query attention: it scales with `nKvHead`, not `nHead`.
 */
export function kvCacheBytes(c: SizeConfig, seqLen: number, batch = 1, bytesPerValue = 2): number {
  return 2 * batch * c.nLayer * c.nKvHead * headDim(c) * seqLen * bytesPerValue;
}

/**
 * Forward-pass FLOPs for one token, counting matmuls only.
 *
 * The convention every scaling paper uses: 2 FLOPs per multiply-accumulate, and
 * training costs roughly 3x the forward pass. Attention's quadratic term is listed
 * separately because at 1k context it is a rounding error and at 128k it dominates,
 * which is the whole reason long context is hard.
 */
export function flopsPerToken(c: SizeConfig, seqLen: number): { dense: number; attention: number } {
  const p = parameters(c);
  // The embedding lookup is a gather, not a matmul; the head is a real matmul even
  // when its weights are tied.
  const dense = 2 * (p.attention + p.feedForward + c.nEmbd * c.vocabSize);
  const attention = 2 * 2 * c.nLayer * seqLen * c.nEmbd;
  return { dense, attention };
}

export function formatCount(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`;
  return `${n}`;
}

export function formatBytes(n: number): string {
  if (n >= 2 ** 30) return `${(n / 2 ** 30).toFixed(2)} GiB`;
  if (n >= 2 ** 20) return `${(n / 2 ** 20).toFixed(1)} MiB`;
  if (n >= 2 ** 10) return `${(n / 2 ** 10).toFixed(1)} KiB`;
  return `${n} B`;
}
