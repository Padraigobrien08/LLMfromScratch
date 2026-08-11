/**
 * A display figure printed as its process plates.
 *
 * Broadsheet's `.cmyk-num` builds the treatment out of four copies of the glyphs: a
 * `.paper` span carrying the real text in the white of the sheet, and three
 * aria-hidden `.plate` repeats in cyan, magenta and yellow, nudged out of register
 * and multiplied together — the dark core is where all three overlap, the fringes
 * are the drift. The markup is the system's, documented in its own stylesheet; the
 * size is the caller's, set by the context this sits in.
 */
export default function PlateNumeral({ value }: { value: string }) {
  return (
    <span className="cmyk-num plate-num">
      <span className="paper">{value}</span>
      <span className="plate plate-c" aria-hidden="true">
        {value}
      </span>
      <span className="plate plate-m" aria-hidden="true">
        {value}
      </span>
      <span className="plate plate-y" aria-hidden="true">
        {value}
      </span>
    </span>
  );
}
