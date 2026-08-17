import { describe, expect, it } from "vitest";

import sizes from "../data/model-sizes.json";
import {
  GPT2_124M,
  LLAMA_124M,
  type SizeConfig,
  headDim,
  kvCacheBytes,
  mlpHidden,
  parameters,
} from "./modelsize";

/** The fixture stores the Python field names; the calculator uses camelCase. */
function toConfig(c: (typeof sizes.cases)[number]["config"]): SizeConfig {
  return {
    vocabSize: c.vocab_size,
    nLayer: c.n_layer,
    nHead: c.n_head,
    nKvHead: c.n_kv_head,
    nEmbd: c.n_embd,
    blockSize: c.block_size,
    norm: c.norm as SizeConfig["norm"],
    posEmb: c.pos_emb as SizeConfig["posEmb"],
    mlp: c.mlp as SizeConfig["mlp"],
    tieEmbeddings: c.tie_embeddings,
    bias: c.bias,
    mlpRatio: c.mlp_ratio,
    mlpHiddenMultipleOf: c.mlp_hidden_multiple_of,
  };
}

describe("parity with the real Transformer", () => {
  it.each(sizes.cases.map((c) => [c.name, c] as const))(
    "counts %s exactly",
    (_name, c) => {
      const cfg = toConfig(c.config);
      expect(headDim(cfg)).toBe(c.head_dim);
      expect(mlpHidden(cfg)).toBe(c.mlp_hidden);
      // Exact, not approximate: being off by one bias vector per layer is precisely
      // the error this fixture exists to catch.
      expect(parameters(cfg).total).toBe(c.total_params);
    },
  );
});

describe("what the breakdown says about the budget", () => {
  it("puts ~31% of GPT-2 124M in the token embedding", () => {
    const p = parameters(GPT2_124M);
    expect(p.tokenEmbedding / p.total).toBeGreaterThan(0.3);
    expect(p.tokenEmbedding / p.total).toBeLessThan(0.32);
  });

  it("charges nothing for a tied head, and the full matrix for an untied one", () => {
    expect(parameters(GPT2_124M).lmHead).toBe(0);
    const untied = parameters({ ...GPT2_124M, tieEmbeddings: false });
    expect(untied.lmHead).toBe(GPT2_124M.nEmbd * GPT2_124M.vocabSize);

    // How many blocks' worth, exactly. The page said "as much as adding two whole
    // blocks"; it is 5.45, and a `toBeGreaterThan(two blocks)` assertion was true of
    // both numbers, so nothing caught it. On the page whose own claim is that nothing
    // there is arithmetic anybody did by hand, the figure has to be the computed one.
    const added = untied.total - parameters(GPT2_124M).total;
    const perBlock =
      (parameters({ ...GPT2_124M, nLayer: GPT2_124M.nLayer + 1 }).total -
        parameters(GPT2_124M).total);
    expect(added / perBlock).toBeCloseTo(5.45, 2);
    expect(added).toBe(38_633_472);
  });

  it("keeps SwiGLU within a percent of GELU at equal width", () => {
    const gelu = parameters(GPT2_124M).feedForward;
    const swiglu = parameters({ ...GPT2_124M, mlp: "swiglu" }).feedForward;
    // The 2/3 correction is what makes the ablation a test of the activation
    // rather than of the parameter budget.
    expect(Math.abs(swiglu - gelu) / gelu).toBeLessThan(0.01);
  });
});

describe("kv cache", () => {
  it("scales with kv heads, not query heads — the case for GQA", () => {
    const mha = kvCacheBytes(GPT2_124M, 1024);
    const gqa = kvCacheBytes(LLAMA_124M, 1024);
    expect(mha / gqa).toBeCloseTo(3, 10);
  });

  it("grows linearly with sequence length and batch", () => {
    expect(kvCacheBytes(GPT2_124M, 2048)).toBe(2 * kvCacheBytes(GPT2_124M, 1024));
    expect(kvCacheBytes(GPT2_124M, 1024, 8)).toBe(8 * kvCacheBytes(GPT2_124M, 1024));
  });
});
