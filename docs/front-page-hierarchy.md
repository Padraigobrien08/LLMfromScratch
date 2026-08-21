# Front page hierarchy: what is still wrong

A backlog for the front page (`web/src/pages/Front.tsx`, `web/src/components/StackFigure.tsx`,
`web/src/styles.css`). Written to be acted on without the conversation that produced it.

Measured at **1352x722**, which is the review window, with cross-checks at 1280x720,
1340x715, 1439x715, 1440x777 and 1920x1080. Every number below was read off the page, not
estimated.

The short version: the type scale is fixed, and what is left is information architecture.
Four of the six remaining items are about the same thing, which is that the page offers the
reader more entry points than it has ranks to put them in.

---

## Already fixed, do not redo

| What | Where |
|---|---|
| Type scale: 19 independently tuned sizes replaced by 5 tokens, 6 levels | `--type-label/small/body/title/lead` in `styles.css` `:root` |
| Figure title inverted under the panel title below ~1500px | figure title on `--type-lead`, panel title on `--type-title` |
| Metric values byte-identical to route titles (18px/600 both) | metric values on `--type-body` |
| Signpost kicker at 15px against the panel's 10.5px, wrapping to two lines | `.front-routes .section-label` on `--type-label` |
| `01 02 03` ordinals painted in the link colour while containing no link | `.front-routes .destination-num` now `--color-neutral-700` |
| Deck 173 characters, needing more measure than any window under 1440 had | trimmed to 149 in `Front.tsx` |
| 40px of dead margin under the variant strip | `.front-figure .stack-figure` margin-bottom in the short-window block |
| The page not fitting 1352x722 at all (72px over, 189 on one selection) | the 1340-1439 x 715-800 rebalance block |

**A rule that was added and then deliberately removed:** a `scrollbar-width: none` on the
front page, to reclaim the 15px gutter. It was worth 21px of height when the deck was 173
characters. Once the deck was trimmed it bought nothing (measured: the page needs 775px at
1440 with or without it, because the plate's floor sets the columns row), and its only
remaining effect was hiding a scroll on windows that genuinely overflow. The reasoning is
recorded in the short-window block in `styles.css`. Please do not re-add it without new
evidence.

---

## Still wrong

### 1. The loudest element on the page is a footnote (P1)

The `READ` box is the only inverted block on the screen: **335x44px, 14,752px² of solid
near-black at 14.86:1**. A primary route button is **128x27px, 3,403px²**, outlined, on the
page background. So the visual weight ratio is about **4.3 to 1 in favour of the secondary
read**, and the box's own eyebrow says `READ` while it sits directly above one that says
`ALSO`.

The three route buttons are what the page's own copy calls the answer to "where do you want
to go?". They lose to a cross-reference.

**Fix direction:** either the routes take the filled treatment and the `READ` box becomes a
ruled link, or the box keeps its fill and the routes get a weight that beats an outline.
Do not solve it by making the box bigger for the routes as well; two filled black blocks in
one viewport is the same problem twice.

**Watch:** the routes column is not the binding column for page height (467px against the
figure's 573px at 1440x777), so growing the buttons is close to free vertically. Making the
`READ` box quieter is also free. Making both louder is not.

### 2. The same numbers are printed two and three times (P2)

Within about 90px of each other:

| | dateline strip | metrics row |
|---|---|---|
| tests | `412 tests green` | `412 tests` |
| loss | `GPT-2 124M · validation loss 3.05` | `3.05 validation loss` |
| runs | `Ablation sweep complete · 39 runs` | `39 ablation runs` |
| params | (not present) | `124.5M parameters` |

`124.5M` then appears a third time in the panel as `124.5M in total`, and again on most
selections as the denominator in `38.6M · 31.0% of 124.5M`.

Neither of the first two rows is styled as the authoritative one, so a reader compares them
instead of reading them. Three of the four figures are pure repetition.

**Fix direction:** pick which row owns the numbers. The dateline is a newspaper's furniture
and reads as provenance; the metrics row is evidence attached to the claim and each figure
links to the page that proves it. That argues for the metrics row keeping the numbers and
the dateline carrying only what is not repeated (byline, licence, and something that is
genuinely status rather than measurement). Copy change, so it is Padraig's call which words
survive.

### 3. Five overlapping ways to reach three destinations (P3)

**57 clickable elements** on one screen. The overlap is exact, not approximate:

```
nav  Learn    -> #/chapter/1      route  Start the path       -> #/chapter/1
nav  Results  -> #/reproduction   route  See the results      -> #/reproduction
nav  Explore  -> #/rope           route  Open the explorers   -> #/rope
```

Byte-identical hrefs. Plus 14 clickable figure labels, the variant control, the `READ` box,
a plate reference, and two colophon links.

This is not automatically wrong; a signpost that restates a nav item with a sentence of
explanation is a real editorial device. But it is worth deciding on purpose rather than by
accumulation, because the duplication is what makes the routes column and the nav compete
for the same job.

**Fix direction:** decide what the nav is for. If the routes are the front page's real
entry points, the nav can be quieter here than on a reading page. If the nav is the primary
system, the routes should stop being buttons and become what they nearly are already: three
short paragraphs with links.

### 4. The ordinals imply a sequence the copy denies (P3)

The kicker asks "Where do you want to go?", which offers alternatives. The items beneath it
are numbered `01`, `02`, `03`, which reads as steps to complete in order. A reader is told
to choose and shown a procedure.

The colour bug is fixed (they were the link colour) but the sequence implication is not.

**Fix direction:** either drop the ordinals, or commit to the sequence and change the
kicker so the numbers are true. Both are defensible; the current pair is not.

### 5. The masthead is off the scale (P3)

Deliberately left alone, because it is shared by every page and this pass was scoped to the
front page. It is now the only part of the page not on the new tokens:

| Element | Size | Nearest token step |
|---|---|---|
| nameplate | 25px flat | between `title` (24 at 1920) and `lead` (30) |
| nav | 13px flat | `label` caps at 13, `small` floors at 12.5 |
| dateline | 11.5px flat | between `label` (10.5) and `small` (12.5) |

All three are fixed pixels, so they do not scale with the window while everything around
them now does. That is the same two-systems fault the type scale was written to remove,
surviving in the one component the pass did not touch.

**Fix direction:** put the nameplate on `--type-lead`, the nav on `--type-small`, the
dateline on `--type-label`. It affects every page, so it needs a look at the chapter,
results and explorer pages rather than the front page alone.

### 6. Two eyebrow systems (P3)

Front page eyebrows are now fluid, `--type-label`, 10.5px to 13px. Every other page's
`.eyebrow` is still flat 10.5px. Consistent within the front page, divergent across the
site. Low priority, but it will look like a bug to the next person who compares two pages
side by side.

---

## Constraints any fix must respect

**The page fits by single digits.** At 1352x722 it fits exactly, on all fourteen block
selections. There is no slack. Anything that adds vertical height needs re-verifying, and
"it looks fine" is not verification, because the panel's height changes with which block
the reader has clicked and four of the fourteen are much taller than the rest (the two
norms, the position embedding, the output head).

**The rebalance band is narrow: 1340-1439 wide by 715-800 tall.** Below 1340 the panel
stops being window-bound and becomes reserve-bound, dropping to 379px at 1280, where eight
of the fourteen selections overflow and labels print 34px into the variant strip. Below
715px tall the plate hits its floor and labels start reaching the strip. Outside that gate
the page reverts to a scrolling page, which is intended.

**`LIST_BELOW_PLATE = 470` in `stack/engine.ts` is paired with the 520px reserve in
`styles.css`.** One decision written in two files. Below that plate width the figure stops
ruling labels into margins and prints them as a list, which is a worse page. If you widen
the panel further, lower the constant first and re-run the containment sweep.

**No em dashes in anything a reader sees.** Enforced by convention in this repo (see the
copy commits) and by the site's own tests in places. Commas, colons, semicolons,
parentheses, full stops.

**Two sessions share this checkout.** Commit with explicit pathspecs, never a bare
`git commit -a`, and verify pushes with `git ls-remote` rather than trusting the local
origin cache.

---

## How to verify a change

Preview is a static snapshot, not a dev server:

```bash
npm run snapshot --prefix web && python3 -m http.server 4173 --directory local-site
```

`python -m http.server` lets the browser cache `index.html`, so reload with a cache-busting
query (`/?cb=1`) or a hard reload, and confirm the served CSS hash actually changed before
believing a result.

Then run the fourteen-selection sweep at 1352x722 and 1440x777 as a minimum, plus the gate
corners 1340x715 and 1439x715 if the change touches layout. Pass criteria, all of which
the page currently meets:

- `documentElement.scrollHeight - clientHeight === 0` on every selection
- no label rectangle intersecting `.stack-fig-foot` (the variant strip)
- no label rectangle intersecting `.stack-fig-panel`
- no label rectangle extending below `.stack-fig-area`
- zero pairwise label intersections
- `.stack-fig-area[data-fig-layout]` stays `"margins"`, never `"list"`

**Harness note, this cost an hour to find.** The Browser pane reports
`document.visibilityState === "hidden"`, so `setTimeout` is throttled to about 2s and then
stalls entirely, and `requestAnimationFrame` does not fire, which means the figure's morph
never settles and label positions read stale. Drive it manually instead: patch
`requestAnimationFrame` to dispatch through a `MessageChannel` port, and busy-wait on
`performance.now()` rather than sleeping. With that in place a fourteen-selection sweep runs
in a few seconds.

---

## One thing I could not settle

The plate's height floor is 385px in the rebalance band, and I could not establish where
label containment actually breaks. Two measurement methods disagree:

- Pinning `grid-template-rows` to a fixed height and sweeping 375px to 425px: **clean at
  every height**, no label reaching the strip.
- Leaving `minmax(floor, 1fr)` and shrinking the window until the floor binds: **21px of
  intrusion at a 385px floor, 1px at 405px** (measured at 1340x660).

Same resolved plate height, different results, so one of the two is an artifact of the
harness rather than a real measurement. My guess is stale label ranging: the engine may not
re-range when the plate's height changes without a selection change, which would be a real
bug for anyone resizing their window, and would also explain why the pinned sweep (where I
always clicked a label afterwards) came out clean.

**This is why the band carries a `min-height: 715px` bound rather than a measured floor.**
Resolving it would let the band cover shorter windows honestly. It is the highest-value
thing on this list for anyone who wants the underlying mechanism to be trustworthy, as
opposed to the page to look right.
