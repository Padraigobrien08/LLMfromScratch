/**
 * A display figure, set as instrument output.
 *
 * This used to print through Broadsheet's `.cmyk-num`: four copies of the glyphs — a
 * paper span plus three aria-hidden cyan/magenta/yellow repeats nudged out of register
 * and multiplied — so a figure read as a colour plate that had drifted on the press.
 * It was the loudest thing on the front page and it fringed every measured number in
 * pink and cyan, which is a poor way to print a result someone is meant to trust.
 *
 * There is no quieter version of that effect: the plates multiply, so tightening the
 * offsets does not give near-black with faint edges, it gives the C×M×Y product — a
 * muddy maroon. It goes or it stays. It goes.
 *
 * What replaces it is the mono the site already uses for code, artifact paths and
 * table figures, in ink. Every number this component prints is a readout — a parameter
 * count, a validation loss, a logit spread — and setting them in the same face as the
 * data tables says so. `tabular-nums` keeps columns of them aligned; the size is still
 * the caller's, set by the context this sits in.
 */
export default function PlateNumeral({ value }: { value: string }) {
  return <span className="plate-num">{value}</span>;
}
