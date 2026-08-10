import { describe, expect, it } from "vitest";

import vocab from "../data/gpt2-vocab.json";
import fixture from "../data/tokenizer-fixture.json";
import { Tokenizer } from "./tokenizer";

const tok = new Tokenizer(vocab as string[]);

describe("parity with the Python tokenizer", () => {
  it("has the same vocabulary size", () => {
    expect(tok.size).toBe(fixture.n_vocab);
  });

  /**
   * Loose parity is not good enough here. A tokenizer that is right for ordinary
   * prose and wrong on a leading space produces a page that teaches a subtly false
   * thing, in the one place the reader is least able to notice.
   */
  it.each(fixture.cases.map((c) => [JSON.stringify(c.text), c] as const))(
    "encodes %s exactly as Python does",
    (_label, c) => {
      expect(tok.encode(c.text)).toEqual(c.ids);
    },
  );
});

describe("round trips", () => {
  it("decodes back to the original text", () => {
    for (const c of fixture.cases) {
      expect(tok.decode(c.ids)).toBe(c.text);
    }
  });

  it("keeps the leading space inside the token, where GPT-2 puts it", () => {
    // The classic off-by-one: " the" and "the" are different tokens, and a port
    // that strips the space silently produces a different vocabulary.
    const [withSpace] = tok.tokenize(" the");
    const [without] = tok.tokenize("the");
    expect(withSpace!.text).toBe(" the");
    expect(without!.text).toBe("the");
    expect(withSpace!.id).not.toBe(without!.id);
  });
});

describe("tokenize spans", () => {
  it("reports offsets that index back into the original string", () => {
    const text = "The cat sat";
    for (const token of tok.tokenize(text)) {
      expect(text.slice(token.start, token.end)).toContain(token.text.trim());
    }
  });

  it("covers the whole input", () => {
    const text = "Dorothy lived in the midst of the great Kansas prairies.";
    expect(tok.tokenize(text).map((t) => t.text).join("")).toBe(text);
  });
});
