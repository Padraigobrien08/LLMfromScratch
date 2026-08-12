/**
 * Small counts, spelled out.
 *
 * The front page used to say "Thirteen stops" above a list of sixteen, because the
 * word was typed and the list was not. Nothing here is a measured figure — these are
 * counts of links and table rows — but the same rule applies: a sentence that states
 * a number the page also renders should derive it, or it will drift.
 *
 * Above twenty the digits are better prose anyway, so the fallback is not a gap.
 */
const WORDS = [
  "zero",
  "one",
  "two",
  "three",
  "four",
  "five",
  "six",
  "seven",
  "eight",
  "nine",
  "ten",
  "eleven",
  "twelve",
  "thirteen",
  "fourteen",
  "fifteen",
  "sixteen",
  "seventeen",
  "eighteen",
  "nineteen",
  "twenty",
];

export function numberWord(n: number): string {
  return WORDS[n] ?? String(n);
}

/** The same word, where it opens a sentence. A digit fallback is unchanged by this. */
export function numberWordCapitalized(n: number): string {
  const word = numberWord(n);
  return word.charAt(0).toUpperCase() + word.slice(1);
}
