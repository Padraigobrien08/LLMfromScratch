/**
 * GPT-2 byte-pair encoding, in the browser.
 *
 * The same reasoning as the RoPE port: the explainer's opening claim is that the
 * model sees tokens rather than text, and that claim is only worth making if the
 * reader can type their own sentence and watch the *real* vocabulary split it.
 * A hand-waved approximation would teach the wrong lesson in the exact place the
 * reader has no way to check.
 *
 * The algorithm is tiktoken's, which needs only a map from token bytes to rank:
 * repeatedly merge whichever adjacent pair has the lowest-ranked concatenation,
 * and stop when no adjacent pair is in the vocabulary. `tokenizer.test.ts` pins the
 * output against ids produced by the Python tokenizer.
 *
 * Bytes are handled in GPT-2's byte-to-unicode representation, where every byte
 * maps to exactly one character. That makes a token a plain string, so merging is
 * string concatenation and a rank lookup is a single Map hit.
 */

/** GPT-2's reversible byte→character map, rebuilt rather than shipped. */
function bytesToUnicode(): string[] {
  const bs: number[] = [];
  for (let i = "!".charCodeAt(0); i <= "~".charCodeAt(0); i++) bs.push(i);
  for (let i = "¡".charCodeAt(0); i <= "¬".charCodeAt(0); i++) bs.push(i);
  for (let i = "®".charCodeAt(0); i <= "ÿ".charCodeAt(0); i++) bs.push(i);

  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const table: string[] = new Array(256);
  for (let i = 0; i < bs.length; i++) table[bs[i]!] = String.fromCharCode(cs[i]!);
  return table;
}

/**
 * GPT-2's pre-tokenization pattern.
 *
 * It is what puts the leading space *inside* a token — the detail that makes
 * " the" and "the" different entries, and the one most ports get wrong.
 */
const PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

export type Token = {
  id: number;
  /** The token's text, decoded back to something readable. */
  text: string;
  /** Character offsets into the original input, for highlighting. */
  start: number;
  end: number;
};

export class Tokenizer {
  private readonly ranks: Map<string, number>;
  private readonly byteEncoder: string[];
  private readonly byteDecoder: Map<string, number>;
  readonly vocab: string[];

  constructor(vocab: string[]) {
    this.vocab = vocab;
    this.ranks = new Map();
    for (let i = 0; i < vocab.length; i++) this.ranks.set(vocab[i]!, i);
    this.byteEncoder = bytesToUnicode();
    this.byteDecoder = new Map();
    for (let b = 0; b < 256; b++) this.byteDecoder.set(this.byteEncoder[b]!, b);
  }

  get size(): number {
    return this.vocab.length;
  }

  private toByteChars(text: string): string[] {
    const bytes = new TextEncoder().encode(text);
    const out: string[] = new Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) out[i] = this.byteEncoder[bytes[i]!]!;
    return out;
  }

  /** Merge the lowest-ranked adjacent pair until none of them is in the vocabulary. */
  private mergePiece(chars: string[]): string[] {
    if (chars.length <= 1) return chars;
    const parts = chars.slice();
    for (;;) {
      let bestRank = Infinity;
      let bestIndex = -1;
      for (let i = 0; i < parts.length - 1; i++) {
        const rank = this.ranks.get(parts[i]! + parts[i + 1]!);
        if (rank !== undefined && rank < bestRank) {
          bestRank = rank;
          bestIndex = i;
        }
      }
      if (bestIndex === -1) return parts;
      parts.splice(bestIndex, 2, parts[bestIndex]! + parts[bestIndex + 1]!);
    }
  }

  encode(text: string): number[] {
    return this.tokenize(text).map((t) => t.id);
  }

  /** Encode, keeping each token's span in the original text so the UI can align them. */
  tokenize(text: string): Token[] {
    const out: Token[] = [];
    PATTERN.lastIndex = 0;
    for (const match of text.matchAll(PATTERN)) {
      const piece = match[0];
      const start = match.index;
      for (const part of this.mergePiece(this.toByteChars(piece))) {
        const id = this.ranks.get(part);
        if (id === undefined) continue;
        out.push({ id, text: this.decodeToken(part), start, end: start + piece.length });
      }
    }
    return out;
  }

  private decodeToken(part: string): string {
    const bytes = new Uint8Array(part.length);
    for (let i = 0; i < part.length; i++) bytes[i] = this.byteDecoder.get(part[i]!) ?? 0;
    return new TextDecoder().decode(bytes);
  }

  decode(ids: number[]): string {
    const chars = ids.map((id) => this.vocab[id] ?? "").join("");
    return this.decodeToken(chars);
  }
}

let pending: Promise<Tokenizer> | null = null;

/**
 * Load the vocabulary on demand.
 *
 * A dynamic import so Vite splits the 0.5 MB of vocabulary into its own chunk:
 * a reader who never opens the explainer never downloads it.
 */
export function loadTokenizer(): Promise<Tokenizer> {
  pending ??= import("../data/gpt2-vocab.json").then(
    (module) => new Tokenizer(module.default as string[]),
  );
  return pending;
}
