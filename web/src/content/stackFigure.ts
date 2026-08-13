import { BLOCKS, type Variant } from "./blocks";
import { ARCHITECTURES } from "./architecture";
import { SIZES } from "./blocks";
import { formatCount, parameters } from "../lib/modelsize";
import { href } from "../router";
import type { LabelSpec, Tier } from "../components/stack/figureRules";

/**
 * What Figure V's detail panel says, and where each part sends the reader.
 *
 * The figure draws fourteen labels but the stack only has nine blocks with parameters,
 * and `blocks.ts` is the authority on those nine. Every one of them resolves straight
 * through `FROM_BLOCKS` below: the panel prints that module's own `what`, `shape`,
 * `params`, `differs` and `pins`, so the figure and the Architecture page cannot come
 * to disagree about the model, and no sentence about the stack exists twice.
 *
 * The five remaining labels — the whole object, the inputs, the tie, the residual
 * stream and the logits — are not blocks and have no entry there. Their copy lives
 * here, in the same shape, and follows the same rule `blocks.ts` sets for itself:
 * **a pin names a test that was read, and a part with no property test of its own says
 * so rather than borrowing a neighbour's.** Three of the five are genuinely pinned.
 * The residual stream is not, and prints the honest line instead.
 */

export type FigureLink = { href: string; text: string; external?: boolean };

export type FigurePanel = {
  name: string;
  what: string;
  shape: string;
  /** `null` where the part holds no weights of its own — not zero, which reads as measured. */
  params: number | null;
  differs: string | null;
  pins: { test: string; claim: string } | null;
  links: FigureLink[];
};

/** Figure part → the block in `blocks.ts` that owns its prose. */
const FROM_BLOCKS: Record<string, string> = {
  tok_emb: "token-embedding",
  pos_emb: "position",
  norm: "norm",
  attn: "attention",
  ffn: "feed-forward",
  block: "blocks",
  final_norm: "final-norm",
  head: "output-head",
  kv: "kv-cache",
};

/** Where each part sends the reader. `attn` also gets the external explorer, appended
 *  by `figurePanel` because only the page knows that URL. */
const LINKS: Record<string, Array<[string, string]>> = {
  whole: [
    [href({ kind: "chapter", n: 8 }), "So does any of the design actually matter?"],
    [href({ kind: "ablations" }), "Plate II — the ablation table"],
  ],
  inputs: [[href({ kind: "chapter", n: 1 }), "A model never sees your text"]],
  tok_emb: [
    [href({ kind: "chapter", n: 2 }), "Each token becomes a list of numbers"],
    [href({ kind: "efficiency" }), "Plate III, for the quantization cap"],
  ],
  pos_emb: [
    [href({ kind: "chapter", n: 5 }), "Telling the model where each token sits"],
    [href({ kind: "rope" }), "The RoPE explorer"],
    [href({ kind: "ablations" }), "Plate II"],
  ],
  norm: [[href({ kind: "ablations" }), "Plate II — RMSNorm measurably did nothing to loss"]],
  attn: [
    [href({ kind: "chapter", n: 4 }), "Letting every token look back"],
    [href({ kind: "efficiency" }), "Plate III"],
  ],
  kv: [[href({ kind: "efficiency" }), "Plate III — the bug that hid in the numbers"]],
  ffn: [
    [href({ kind: "chapter", n: 3 }), "Words only mean things in context"],
    [href({ kind: "architecture" }), "The stack, block by block"],
  ],
  block: [[href({ kind: "architecture" }), "The stack, block by block"]],
  tie: [
    [href({ kind: "chapter", n: 6 }), "Out comes a probability for every token"],
    [href({ kind: "efficiency" }), "Plate III, for the quantization cap"],
  ],
  residual: [[href({ kind: "architecture" }), "The stack, block by block"]],
  final_norm: [[href({ kind: "architecture" }), "The stack, block by block"]],
  head: [
    [href({ kind: "chapter", n: 6 }), "Out comes a probability for every token"],
    [href({ kind: "architecture" }), "The stack, block by block"],
  ],
  logits: [[href({ kind: "chapter", n: 6 }), "Out comes a probability for every token"]],
};

type Extra = Omit<FigurePanel, "links"> | ((v: Variant) => Omit<FigurePanel, "links">);

const int = (n: number) => n.toLocaleString("en-US");

/**
 * The five parts that are not blocks.
 *
 * Each pin below names a test that exists and was read. `test_gpt2_124m_parameter_count`
 * deliberately is not cited for the whole object: it checks the *published* GPT-2 small
 * figure to within a range and excludes the position table, so it does not pin the total
 * this figure's thicknesses are shares of. `modelsize.test.ts` does, exactly.
 */
const EXTRAS: Record<string, Extra> = {
  whole: (v) => {
    const c = ARCHITECTURES[v].config;
    return {
      name: "The whole object",
      what:
        "A decoder-only transformer: one tower, no encoder, nothing crossing in from the " +
        `side. A sentence enters at the bottom as integers, becomes ${c.nEmbd} numbers per ` +
        `token, passes ${c.nLayer} times through the same two-part block, and leaves the top ` +
        "as a probability for every token in the vocabulary.",
      shape: `${c.nLayer} layers · ${c.nEmbd} wide · ${c.nHead} heads · ${int(c.blockSize)} context`,
      params: parameters(SIZES[v]).total,
      differs: null,
      pins: {
        test: "web/src/lib/modelsize.test.ts",
        claim:
          "that this figure's arithmetic reproduces the real Transformer's parameter count " +
          "exactly — not approximately — across twelve configurations, against a fixture " +
          "generated by instantiating the model and summing p.numel(). Every thickness here " +
          "is a share of that total.",
      },
    };
  },
  inputs: (v) => ({
    name: "Inputs",
    what:
      "A sentence, tokenised into integers. Nothing is learned here — the tokeniser is a " +
      "fixed table, and this is the only place in the model where the data is not a tensor " +
      "of floats.",
    shape: `batch × time · time ≤ ${int(ARCHITECTURES[v].config.blockSize)}`,
    params: null,
    differs: null,
    pins: {
      test: "test_model.py::test_sequence_longer_than_block_size_rejected",
      claim:
        "that a sequence longer than block_size raises rather than being quietly truncated — " +
        "the one failure here that would otherwise look like a modelling problem much later.",
    },
  }),
  tie: (v) => {
    const c = ARCHITECTURES[v].config;
    return {
      name: "The same matrix, twice",
      what:
        "The dashed strand is not a part. It marks that the output head and the token " +
        "embedding are one tensor: the table that turns a token into a vector, transposed, " +
        "turns a vector back into a score per token. Untying them would add " +
        `${formatCount(c.vocabSize * c.nEmbd)} parameters — about as much as two more blocks.`,
      shape: `${int(c.vocabSize)} × ${c.nEmbd}, shared`,
      params: null,
      differs: null,
      pins: {
        test: "test_model.py::test_weight_tying_shares_storage",
        claim:
          "that the head's weight tensor *is* the embedding tensor — identity, not equality — " +
          "and that untying makes the model strictly larger. Tying implemented as a copy would " +
          "pass an equality check and still be wrong.",
      },
    };
  },
  residual: (v) => ({
    name: "The residual stream",
    what:
      "The spine. One vector per position that runs the full height of the tower, and that " +
      "every sublayer adds to rather than replaces. Because the additions never interrupt " +
      "it, the gradient reaches the bottom of the stack as easily as the top — which is what " +
      "the norms sitting beside the stream, rather than across it, are there to preserve.",
    shape: `batch × time × ${ARCHITECTURES[v].config.nEmbd}, unbroken`,
    params: null,
    differs: null,
    // No test asserts the stream identity itself. `test_residual_init_is_downscaled` pins
    // the 1/√(2·n_layer) initialisation and is already cited by the block it belongs to;
    // borrowing it here would be claiming more than the suite checks.
    pins: null,
  }),
  logits: (v) => ({
    name: "Logits → sampling",
    what:
      "One number per token in the vocabulary, for every position. Temperature and top-k " +
      "turn that into a choice, the choice is appended to the sentence, and the whole tower " +
      "runs again.",
    shape: `batch × time × ${int(ARCHITECTURES[v].config.vocabSize)}`,
    params: null,
    differs: null,
    pins: {
      test: "test_model.py::test_top_k_restricts_support",
      claim:
        "that sampling with top-k = 1 produces exactly the sequence greedy decoding does — " +
        "the boundary where a sampler and an argmax must agree, and the cheapest place for an " +
        "off-by-one in the top-k mask to show itself.",
    },
  }),
};

/** The panel's content for one part, in one architecture. */
export function figurePanel(blockId: string, variant: Variant, attentionHref: string): FigurePanel {
  const links: FigureLink[] = (LINKS[blockId] ?? []).map(([h, text]) => ({ href: h, text }));
  if (blockId === "attn") {
    links.splice(1, 0, { href: attentionHref, text: "The attention explorer", external: true });
  }

  const fromBlocks = FROM_BLOCKS[blockId];
  if (fromBlocks) {
    const b = BLOCKS.find((x) => x.id === fromBlocks)!;
    return {
      name: b.title,
      what: b.what,
      shape: b.shape(variant),
      params: b.params(variant),
      differs: b.differs?.(variant) ?? null,
      pins: b.pins,
      links,
    };
  }

  const extra = EXTRAS[blockId] ?? EXTRAS.whole!;
  const resolved = typeof extra === "function" ? extra(variant) : extra;
  return { ...resolved, links };
}

/**
 * The labels, bottom of the stack first, each ruled out to a margin.
 *
 * Every entry carries a tier (rule 5) and a margin (rule 4), and its `blockId` must
 * exist in `ROLE_INK` (rule 2) — all three are checked at mount by `validate`, which
 * warns rather than failing silently the day a fourteenth label arrives.
 */
export type FigureLabel = LabelSpec & { anchor: string; text: string; tier: Tier };

export const FIGURE_LABELS: FigureLabel[] = [
  { key: "inputs", anchor: "inputs", blockId: "inputs", text: "Inputs", tier: "flow", side: "left" },
  { key: "tok_emb", anchor: "tok_emb", blockId: "tok_emb", text: "Token embedding", tier: "region", side: "left" },
  { key: "pos_emb", anchor: "pos_emb", blockId: "pos_emb", text: "Position", tier: "part", side: "left" },
  { key: "residual", anchor: "residual", blockId: "residual", text: "Residual stream", tier: "flow", side: "left" },
  { key: "norm_a", anchor: "norm_a", blockId: "norm", text: "Norm → attention", tier: "part" },
  { key: "attn", anchor: "attn", blockId: "attn", text: "Attention", tier: "part" },
  { key: "kv", anchor: "kv", blockId: "kv", text: "KV cache", tier: "part" },
  { key: "norm_b", anchor: "norm_b", blockId: "norm", text: "Norm → feed-forward", tier: "part" },
  { key: "ffn", anchor: "ffn", blockId: "ffn", text: "Feed-forward", tier: "part" },
  { key: "block", anchor: "block", blockId: "block", text: "The block, repeated", tier: "region" },
  { key: "tie", anchor: "tie", blockId: "tie", text: "the same matrix, twice", tier: "flow" },
  { key: "final_norm", anchor: "final_norm", blockId: "final_norm", text: "Final norm", tier: "part" },
  { key: "head", anchor: "head", blockId: "head", text: "Output head", tier: "region" },
  { key: "logits", anchor: "logits", blockId: "logits", text: "Logits → sampling", tier: "flow" },
];

/** The label's text can depend on the architecture; nothing else about it can. */
export function labelText(label: FigureLabel, variant: Variant): string {
  const c = ARCHITECTURES[variant].config;
  if (label.key === "block") return `The block, ×${c.nLayer}`;
  if (label.key === "pos_emb" && c.posEmb === "rope") return "Position (rotary, in attention)";
  return label.text;
}
