# Known issues

Items 1–17 were found by rendering every file in `examples/` — first the default kind, then
**all 55 plot kinds** — and inspecting the output, 2026-09-01. Items 18–21 came from running
the Hall analyzers against a real measurement, and item 22 from a GUI investigation, both
2026-09-02. Item 23 was found 2026-09-04 by looking at the
figures produced while verifying the item-1 fix. They are recorded here rather than quietly
left because the project's own rule is that a result you cannot trust is more useful reported
than hidden.

Items 24 onwards were found later, most of them while making the Hall analyzers decline to
publish quantities they cannot support; each names what reproduces it. **Items 39–41 are
open** — every other item is fixed.

**Which items change a number.** Of items 1–23: only 19 and 20 change a fitted number or a
value in an exported CSV. Items 1–8 and 11–16 are display; items 9 and 18 are reporting-honesty
gaps; item 10 is CLI ergonomics; item 17 hides a GUI control and item 22 makes one act on the
wrong target; item 21 is an unfinished feature. Of the later items, **26, 27, 29, 30 and 33
change a fitted number or a CSV value**; the rest are display, reporting or documentation.

**Several later items were fixed before any release carried them** — they were introduced and
corrected inside the same unreleased work. Those are marked *never released*, and if you are
running a published version they cannot affect you.

Items 1–17 each name the shipped example that reproduces them, and item 22 reproduces on any
file with more than one plot — so those have a target to verify a fix against. Items 18–21
reproduce on **no** shipped file, and that gap is described where they are listed. Everything
here that is scheduled rather than merely recorded is in [ROADMAP.md](ROADMAP.md).

## Display

**1. An inset can be drawn on top of the main curve.** *FIXED 2026-09-04 (89d39ab): the low-T
inset is now placed by the same measured-occupancy machinery as the legend — corners in the
shipped preference order, then an anchor grid, then a bounded least-bad corner that may graze
a midsection but never hide a curve's terminal point. On this reproducer the inset moves to
the clear band between the curve and the Tc annotation and every point is visible; on
`hall_temperature_dependence.dat` it stays in its shipped lower-right corner. When no position
qualifies, the inset is dropped and the figure says so ("low-T inset omitted"), because a
supplement must never hide the primary data — silently or otherwise.*
`examples/resistivity_superconductor.dat`, default `resistivity_rho_t`: the low-T inset covers
155–295 K of the main axes, hiding **55 of 157 points on each of the two bridges** (110 total,
35 % of each curve). The curve appears to stop at 155 K and reappear as a stub at the right
edge. The inset is placed at a fixed position rather than in whichever region of the axes is
empty — on `hall_temperature_dependence.dat` the same inset lands in genuinely empty space and
looks correct, because that curve rises; this one is flat, so the fixed position sits on data.

**2. Two different estimators are drawn in one R_H(T) panel with no visual separation.**
*FIXED 2026-09-05 (d7ce963, note fitted in f52dcc9): the 0-field+1 fallback series is drawn
visually secondary — hollow markers + dashed connector, keyed off its existing
`role="two_point"` tag — and when both estimator families share the panel a title-slot note
(width-fitted to the canvas) says the step between them is a change of method, not physics.*
`examples/hall_temperature_dependence.dat`: `R_H (antisym)` covers 18–40 K at −3.0e−7 m³/C and
the `R_H (0-field+1)` fallback covers 41–55 K at −2.5e−7, producing an apparent **20 % step at
~40 K that is a change of method, not physics**. The legend distinguishes them; nothing warns
the reader not to read the discontinuity as a result.

**3. Axis offset notation is unreadable at Hall magnitudes.** *FIXED 2026-09-05
(d7ce963, extended in 9a83135): EVERY kind drawing an R_H axis — `hall_rh_t`,
`hall_tdep_RH_T`, `hall_tdep_summary` and `hall_tdep_rh_n_twin`, overlay paths included —
disables the ScalarFormatter offset and uses a mathtext scale, so the header reads ×10⁻⁷
over absolute-valued ticks (−2.50000 …) and the headline R_H reads straight off the axis.
The first fix reached only the two kinds the item named, leaving the summary and the twin
rendering a plain `1e-7` for the same quantity.*
`examples/hall_field_sweeps.dat`, `hall_tdep_r_h_t`: the y-axis header renders as
`1e-11-2.5e-7` — matplotlib's scale and offset concatenated. The headline value the docs
advertise (R_H = −2.5e−7 m³/C) cannot be recovered from the plot.

**4. A legend can be placed over the data.** *FIXED 2026-09-04 (43e5e48): default legend
placement now scores all nine inside positions against the measured data/text/inset occupancy
and relocates outside when nothing inside is clear — on this reproducer the legend leaves the
data at 9 pt and at GUI font size alike. The nine matplotlib positions are also selectable
directly (GUI "Legend loc"), so a user who can see the right spot can pin it.*
`examples/magnetization_vsm_multifield.dat`, `inverse_chi`: the 7-entry legend sat
centre-right with five curves running behind it; the old rule relocated only above a size
threshold and never tested whether the chosen anchor was occupied.

**5. Small legends are relocated far outside-right.** *FIXED 2026-09-04 (43e5e48): the
unconditional outside-right rule is gone — both reproducers now keep full canvas width with
the legend inside (TTO upper-right, ACMS in the clear band between the plateaus). A file
whose data genuinely fills the panel still relocates, measured rather than assumed.*
`thermal_transport.dat` and `ac_susceptibility.dat` spent 20–25 % of canvas width on a
**two-entry** legend, squeezing the panels.

**6. No top headroom.** *FIXED 2026-09-05 (4fb8654): a measured top-headroom pass raises
the frame so the topmost data point keeps ≥8 % of the span clear (the old 5 % margin was
eaten by the 7 pt marker glyph, and the χ″ peak tip was actually clipped at −0.5 %);
bounded so a robust-view exclusion of a far outlier is never re-opened.* The κ peak
(`thermal_transport.dat`, panel a) and the χ′ high-T plateau (`ac_susceptibility.dat`, top
panel) touch the axes frame.

**7. Large fields are labelled in Oe.** *FIXED 2026-09-05 (ce1509e + af64ded), by owner
decision: the unit system stays Oe, and |H| ≥ 10 kOe formats with a k prefix — `90 kOe`,
`45.5 kOe` — trailing zeros trimmed. The 10 kOe threshold leaves the low-field regime
(Curie-Weiss labels, the MPMS 1000 Oe oracle) byte-identical, and the rule lives only in
`fmt_field`, which the first commit made the single source of truth for Oe labels (five
call sites used to bypass it). The Oe↔T toggle is unchanged.* Legends read `90000 Oe` /
`100000 Oe` rather than 9 T /
10 T. There is a global Oe↔T display toggle and Oe is the deliberate default, so this is a
default-choice question, not a bug — but at these magnitudes it costs readability.

**8. A zero-field curve omits its field in TTO legends.** *FIXED 2026-09-05 (4fb8654): on
a multi-field file every curve names its field — the |H| < 50 Oe zero-field convention
collapses the instrument's 0.077 Oe reading to a nominal `0 Oe, cooling`. Single-field
files keep direction-only labels.* `thermal_transport.dat` read
`cooling` for the zero-field curve beside `90000 Oe, cooling` for the other.

## Reporting

**9. A flagged-unphysical value is not flagged *on the figure*.** *FIXED 2026-09-04 (c6fd994):
the fit now carries a machine-readable `gamma_negative` in its `quality_flags` and the
annotation reads it — the γ line on this reproducer says
`γ = -8.3e-03 J/mol·K² (unphysical)`. The value stays visible (γ < 0 is what was measured;
this is not a case for declining), the verdict now travels with the figure. The same audit
that guards text placement renders this file's default kind, so the longer line is checked
against the data too.*
`examples/heat_capacity_multifield.dat`, low-T Cp/T vs T² fit: the annotation box prints
**γ = −8.3e−03 J/mol·K²** as a plain number. A negative Sommerfeld coefficient is unphysical —
it says the T² window or the model is wrong.

The analyzer itself is *not* silent about this: it emits both
`gamma(H) goes negative at one or more fields (unphysical electronic term)` and
`γ<0: unphysical electronic (Sommerfeld) coefficient`, and the GUI status bar shows them. The
gap is narrower than it first appears — a reader who has only the **exported figure**, which is
what ends up in a talk or a paper draft, sees the number without the warning that travels with
it everywhere else.

## CLI

**10. `cryosweep plots <file>` ignores the file.** *FIXED 2026-09-05 (bf24f96): with a
file, the dump's `plots` filters to the detected probe and each entry carries `available`
— whether its series builder yields anything against this analyzed result, the same
predicate `reconcile_layout` uses for "backed" — plus top-level `probe`/`file`. The
no-file dump is byte-identical to before.* `probes`, `fits`, `plots` and `observables`
all emit the *same* global registry dump — verified byte-identical — so `plots` on a resistivity
file lists ACMS kinds. There is no way to ask which kinds a given file can actually draw, which
is exactly what you want after a render returns `data.plot: null` with
`plot kind 'X' unavailable: no series selected`.

## Display (found in the all-kinds pass)

**11. Legend entries can overprint each other.** *FIXED 2026-09-04 (43e5e48): text artists
are obstacles to the occupancy chooser, and on this figure no inside position clears the
Dulong-Petit label, the data plateau, and the inset at once — so the legend relocates
outside-right and every label reads cleanly at both 9 pt and GUI font size.*
`heat_capacity.dat`, `hc_full_cp_t`: the "Cp" and "Dulong-Petit" labels were drawn on top of
one another — on the strongest figure the project produces.

**12. The legend lists artists that were never drawn.** *FIXED 2026-09-04 (43e5e48): the
single-axis entropy path — taken exactly when the analyzer ruled magnetic entropy unresolved —
no longer draws the flat-zero "S magnetic" curve, so neither the invisible line nor its legend
entry exists. Display-only: `entropy_magnetic` still reaches the CSV and JSON, and the
per-field magnetic overlays (explicit opt-ins) are untouched.*
`heat_capacity.dat`, `hc_entropy_vs_t`: "S magnetic" appeared with a dashed swatch although no
visible dashed curve existed, sending the reader hunting for a curve that was not there.

**13. The `vsm_mh` low-field panel does not rescale its y-axis.** *FIXED 2026-09-05
(32ca24a): the zoom panel's y-view is fitted to the data inside its ±10 % field window,
padded like the robust view; the main panel is untouched.*
`magnetization_vsm_multifield.dat`: the right-hand "low field" panel inherits the full-range
y-limits (0–0.55 µ_B) while its data spans 0–0.06, so the zoom panel is ~80 % empty and shows
a short line in one corner — the opposite of what a zoom panel is for.

**14. Field setpoint labels print raw floats.** *FIXED 2026-09-05 (7f708b5):
`fmt_field_setpoint` display-rounds held-field labels — |H| < 50 Oe collapses to the
nominal 0, the rest to 4 significant figures — so the legend reads 0 / 50000 / 100000 /
130000 Oe. Display only; group keys and exported values keep the measured median. Tesla
display remains the item-7 default question.* `heat_capacity_multifield.dat`,
`hc_lowt_multifield`: the legend reads `0.524968 Oe`, `50000.5 Oe`, `100001 Oe`, `130000 Oe`
for what `examples/README.md` correctly calls 0 / 5 / 10 / 13 T. Nominal zero field is printed
to six significant figures. Setpoint labels should be rounded and, at these magnitudes, shown
in tesla.

**15. Multi-field low-T fits are unreadable.** *FIXED 2026-09-05 (7f708b5): each fit
wears its field's colour (matching its data series) and its model's linestyle, with grey
linestyle proxies naming the four models in the legend; the y-view is framed by the data
alone, so a diverging fit clips at the panel edge instead of stretching the axes around
its own overshoot.* Same figure: four fields × four low-T models
draws sixteen curves whose colours do not match their data series, several of which overshoot
the axes entirely. Which fit belongs to which field cannot be read off the plot.

**16. `tto_lorenz_t` cannot show the thing it exists to show.** *FIXED 2026-09-05
(4fb8654): the kind is log-y by default and the view is extended to bracket L/L₀ = 1, so
the divergence, the full curve and the labelled Wiedemann–Franz reference are all visible;
`yscale: linear` still restores the old view.* `thermal_transport.dat`: L/L₀
diverges at low T, so the linear y-axis runs to ~200 (`×10²`), the curve is clipped at the top,
and the **Wiedemann-Franz reference line at L/L₀ = 1 — the entire point of the panel — is
flattened onto the bottom axis**, where its annotation also collides with the data. This panel
wants a logarithmic y-axis.

## GUI

**17. The "Colour…" button is clipped out of the left panel at the default width.**
*FIXED 2026-09-05 (b83cbac): root cause was the preset bar's four buttons carrying the
platform style's 80 px minimum against their own 64 px maximum, forcing the left panel's
minimum to ~392 px. An explicit 48 px minimum restores the intended geometry, the
splitter's initial left pane is measured from the content instead of a fixed 300 px
(screenshot-verified at the 1100×650 default), and the scroll area's horizontal policy is
AsNeeded so a future overflow shows a scrollbar instead of silently clipping controls.*
`cryosweep_gui/file_manager.py:23` gives each of the three file-row buttons
`setMaximumWidth(120)` — up to 360 px plus spacing — inside a panel whose minimum width is
280 px (`probe_tab.py:49`), and the enclosing scroll area sets
`setHorizontalScrollBarPolicy(ScrollBarAlwaysOff)` (`probe_tab.py:86`). So at any default
window size the third button is cut off at the panel edge, with no scrollbar to reach it. It is
recoverable only by dragging the splitter wider, which nothing advertises. Reproduced at both
1700 px and 2000 px window width, so it is not a small-screen artifact.

Unlike everything above, this one hides a **control**, not a value.

## Hall (found against a real measurement, 2026-09-02)

These four differ from everything above. **Items 19 and 20 do change an exported value**, and
**none of the four reproduces on a shipped `examples/` file** — so, unlike items 1–17, they
name no example. They were found by running `hall` and `hall-tdep` against a real Hall-wired
PPMS resistivity file, which does not ship because it carries a real sample identity. The
synthetic examples are symmetric in field and sit exactly on their setpoints, so they exercise
none of these paths: `hall examples/hall_field_sweeps.dat --hall-channel 1 --thickness 0.5
--long-channel 2` and the matching `hall-tdep examples/hall_temperature_dependence.dat` both
exit 0 at confidence ≥ 0.9999 with no `R_H` missing. Reproducing 18–20 needs a file with
**unpaired field points and temperature setpoints that drift by more than 0.05 K**, and no such
file was in the repository. That was itself the finding: this whole class of defect was
invisible to the shipped fixtures. *Closed 2026-09-05 (7993d90):
`examples/hall_mixed_sweeps.dat` is an anonymized, structure-aware-decimated subset of the
very measurement these three were found on — its 200 K loop still straddles the old bin edge
and most of its temperatures carry a single ± pair — and
`tests/core/test_example_hall_real.py` pins all three fixed paths against it.*

**18. `hall-tdep` reports `confidence 0.0` on a result that is correct.** *FIXED
2026-09-02, before the public history begins: a single ± pair now fits as an antisym point anchored at
R_asym(0) = 0; `tdep_min_antisym_points` defaults to 1; measured confidence 0.0 → 1.0 with
every R_H bit-identical to ≤ 2 ulp, shipped example byte-identical.* Measured on the real
file: `status = low_confidence`, `confidence = 0.0`, exit 11, 138 R_H(T) points of which 122
are tagged `r_h_method = "2point"` and **all 138** flagged `low_confidence` — while the same
data plots as a clean R_H(T) curve, and the numbers themselves are right.

The cause is a chain of three defensible rules that compose into a wrong answer.
`hall_tempdep.py:391` runs the antisymmetrized fit only at `antisym_points >= 2`. On this file
only ±40 kOe and ±90 kOe are paired, so at 121 of 138 temperatures exactly **one** pair exists,
which falls through to the 2-point fallback at `:425`. **All 121 of those are bit-identical to
the single-pair antisymmetrized slope** — with a symmetric ±B pair ΣB = 0, so the shared R(0)
cancels — meaning only 1 of the 122 is genuinely un-antisymmetrized. Then `:658-660` computes
the confidence fraction over the 16 true two-pair points alone, and `tdep_min_antisym_points`
(`config.py:22`) defaults to 3, so every one of those 16 is itself low-confidence and the
fraction is 0.0.

A single ± pair *is* an antisymmetrization. The fix is to fit at `antisym_points >= 1` anchored
at R_asym(0) = 0, label it `antisym`, and rebase the fraction on the points actually fitted.
**This is the item most likely to read as "the analysis does not work"** — the result is sound
and the report disowns it.

*Addendum, 2026-09-12: the "measured confidence 0.0 → 1.0" figure above was correct when
written and was deliberately superseded by the later Hall integrity work, not regressed by it.* Confidence is
now `min(fit_quality, resolved_fraction)` (item 33 below): `fit_quality` was the 1.0 this
note quotes, but that fit-quality reading was itself found tautological — every non-`None`
r² hall-tdep ever reported came from a 2-point antisym fit, which fits any two points
exactly regardless of noise — and `resolved_fraction` (whether R_H is distinguishable from
its own σ) did not exist as a ceiling at all in 2026-09-02. Re-measured at the current HEAD
on the same real file: `status = low_confidence`, `confidence = 0.4782608695652174`, exit
**11**. This is not item 18 regressing — the R_H(T) values this item fixed are untouched;
most of them simply do not resolve against their own instrument σ, which is the new rule
correctly reporting a real property of noisy points, not a defect in this one.

**19. A drifting temperature setpoint is split in two, and the split fabricates a carrier
density.** *FIXED 2026-09-02, before the public history begins: `cluster_field_setpoints` adopted for held
temperatures (abs_floor 0.25 K) and for the two hall_tdep held-field groupers; the 199.9 K
phantom row is gone, the 200 K group fits the full 182-point loop, all other points and
both shipped examples byte-identical.* `hall.py:203` bins by `round(float(Tset), 1)`. On the real file the 200 K loop
arrives as three segments at 199.8521 / 199.9904 / 199.9945 K, which straddle the bin edge and
split into `199.9` (46 points) and `200.0` (136). Verified in the output: T = 199.9 carries
`R_H = None` but `carrier_n = 1.0613e+30`, sitting beside the real 6.6228e+29 at T = 200.0 — a
~60 % spurious spike in n(T) at a temperature that was never a setpoint. `hall_n_t` and
`hall_mobility_t` therefore plot 10 points while `hall_rh_t` and `hall_r2_t` plot 9.

This is the same defect class as the VSM field-setpoint fragmentation fixed on 2026-08-31, and
the remedy already exists in-tree: `cluster_field_setpoints` (`cryosweep_core/grouping.py:26`)
clusters the values actually present instead of binning to a grid, so no value can straddle an
edge. It is generic over its input and applies unchanged here. The same `round()` binning sits
on fields at `hall_tempdep.py:163` and `:219`; it does not bite on this file, but it is the
same latent bug.

**20. Derived quantities are published without the R_H they derive from.** *FIXED
2026-09-02, before the public history begins: Stage C now derives only from Stage B; when Stage B declines,
carrier_n/carrier_type/mobility are withheld and `derived_flags = ["antisym_r_h_missing"]`
is carried into JSON and a new CSV column; R_H_raw stays visible.*
`hall.py:239` falls back to `R_H_raw` (the Stage A raw fit) for `carrier_n` and `mobility`
while `pt.R_H` — the trusted Stage B antisymmetrized value — stays `None`. So a row can export
a carrier density and a mobility with an empty Hall coefficient, which is what makes item 19
visible in the CSV rather than merely in a plot. Under the project's own decline discipline, a
derived quantity whose parent was not measured should be withheld, not published from the
untrusted stage.

**21. `current_density_J` is declared and consumed but never assigned.** *FIXED 2026-09-05
(00aa7e7): every temp-dep Hall point now reports the instrument's excitation current I
directly (`Bridge N Excitation (uA)` canonicalized at last), and J = I/(w·t) fills
`current_density_J` as a capability that activates only when sample width (`--width-mm`, or
the new Width field on the Hall panel) and thickness are both supplied — an ungated J on
unset geometry would be scale-arbitrary. The capability reason carries the caveat that I is
the instrument-reported current, not necessarily the requested drive. `hall_tdep_J_T` and
the summary's third axis now render (a constant-drive file draws a flat line — the correct
result), and hall_tdep gained a real per-point CSV.*
Not a defect in a
result — a feature that is wired up at both ends with nothing in the middle.
`hall_tempdep.py:52` declares the field; `catalog.py:901-932` builds two plot series from it;
`render.py:2295` already documents it as "always None on ..." — and a search of
`cryosweep_core`, `cryosweep_cli` and `cryosweep_gui` finds **no assignment anywhere**.
Consequences: `hall_tdep_J_T` renders zero series, and `hall_tdep_summary` silently degrades
from three axes to two. Implementing it needs `Bridge N Excitation (uA)` canonicalized (it
currently has zero hits in `cryosweep_core`) and the honest quantity is J = I/A.

**22. "Save plot" saves the first plot, not the one on screen.** *FIXED 2026-09-05
(b83cbac): `last_figure` is a derived property — the focused card's figure in Focus mode,
else the first card that has one — never a stored snapshot, so Save plot writes exactly
what is on screen.* The figure that button writes
is captured once while the layout renders — `cryosweep_gui/output_panel.py:670` assigns
`last_figure` from the first card that has one, guarded by `and self.last_figure is None`, and
nothing updates it afterwards. Focus mode steps a separate index (`output_panel.py:476`), so
navigating to a plot and pressing **Save plot** silently writes a different one. Reproduces on
any file with more than one plot in the layout.

Choosing plots explicitly *does* work: the neighbouring **"Export plots…"** button
(`cryosweep_gui/probe_tab.py:74`) opens a dialog with a checkbox per plot, PNG/PDF/SVG, DPI,
tight crop and exact-mm sizing. That remains the way to save a chosen *set* of plots;
the single-plot button now follows the screen (see [ROADMAP.md](ROADMAP.md), done).

## Display (found while verifying the item-1 fix, 2026-09-04)

**23. Reference-line labels are drawn on top of the data.** *FIXED 2026-09-04 (c7945d2):
every reference-line label now slides along its own line to the first stretch that is clear
of data, text and the legend — the current position is always tried first, so a label that
was already clear (and every golden image) does not move. A companion test audits every
shipped example at its default kind and fails if any text artist covers more than a handful
of the plotted points, so this defect class cannot return silently.*
Every reference-line label was pinned at a fixed fraction along its line, the same
fixed-position class as items 1 and 4: `Dulong–Petit` covered 93 of 858 points on
`heat_capacity_multifield.dat` (`hc_full_cp_t`), the `T_c` marker label 39 of 391 on
`resistivity_superconductor.dat` (`resistivity_rho_t`), and `Wiedemann–Franz (L = L₀)` 4 of
135 on `thermal_transport.dat` (`tto_lorenz_t`). Distinct from item 11 (legend entries
overprinting each other): this is line labels over the measured curve itself.

**24. The Hall summary's third axis clips its label at the default canvas size.**
*FIXED 2026-09-05 (d7ce963): constrained layout cannot see an offset spine, so the
renderer now measures the J axis' realized right-side extent and reserves exactly that
band via the layout rect, convergently; the label reads cleanly at the bare default.*
`examples/hall_mixed_sweeps.dat`, `hall_tdep_summary` with a width supplied (so the J axis
exists at all — see item 21): the offset right-hand spine carries its `J (A/m²)` label past
the figure's right edge, where it is cut off. Reproduced through `cryosweep plot`, which
renders at the bare `GlobalStyle()` default, so this is the out-of-the-box result rather
than a large-font edge case; at an explicit export size the layout has room and the label
reads cleanly.

The J axis had never been drawn before item 21 was implemented, because the field it plots
was never assigned — so this is a latent layout defect that item 21 exposed rather than
introduced. A related one was fixed there: the spine's position was hardcoded at axes
fraction 1.18 and collided with the µ label, and is now measured against that label's
realized extent.

**Display only.** Every value on the axis is correct and reaches the JSON and the CSV; it is
the axis *label* that is clipped, not the data.

## Display (found while comparing a composite figure across two workflows, 2026-09-06)

**25. An outside legend detaches a composite's third axis and can push the legend off the
canvas.** *FIXED 2026-09-06: offset spines are re-expressed as an outward offset in points at
the position they already hold, and on an overlaid twin/offset composite the legend anchor is
re-derived in the layout each growth produces, paired with an inversely scaled layout rect that
pins the axes region in pixels. Display only; no fitted value changes.*

Reproduces on a shipped example, unlike items 18–21:

```
cryosweep --hall-channel 1 --long-channel 2 --thickness 0.5 --thickness-unit mm \
  --width-mm 1 --probe hall_tdep --plot-kind hall_tdep_summary \
  --style-file style.json --out out.png plot examples/hall_mixed_sweeps.dat
```

with `{"legend_loc": "outside"}` — or any `legend_size` large enough that the legend covers every
in-axes candidate position, which is what dense data does on its own. Measured: the figure renders
at **1697 × 827** against the normal 1063 × 827, the `J` axis sits roughly 40 % of the figure
width out in whitespace with its ticks against the edge, and the legend is drawn at
x ≈ [1963, 2125] on an 1819 px canvas — that is, not suppressed but placed entirely outside it.

The cause is not the J axis and not its tick-label width: forcing six-digit J values on the same
example renders correctly. When the legend takes the outside-right branch,
`_grow_canvas_for_legend` calls `set_size_inches` *after* every axes-fraction position has been
fixed. Constrained layout then re-flows and the host axes widen — measured 458 → 842 px — so every
axes-fraction quantity is multiplied along with them: the offset spine, converged in the pre-growth
layout and never revisited, travels 946 → 1596 px, and the legend's `bbox_to_anchor`, also in
host-axes fraction, follows it off the canvas. Growing alone cannot converge, because constrained
layout hands most of each added inch back to the axes: instrumented, the overflow went
155.6 → 151.8 → 148.2 → 144.6 px over four passes.

Both quantities are really *text* widths — fixed in points, not a proportion of the axes — which
is why expressing them as a fraction of a resizable axes is what breaks.

**Distinct from item 24**, which was the same figure's J label clipping at the default canvas size
with the legend *inside*; that fix is intact and this defect survives it. Panel grids never had
this defect — verified by probing them against the unfixed code — so they keep the previous path,
which is what preserves layout space for each panel's own legend.

Seventeen rendered figures changed as a result, all of them overlaid twin/offset composites that
take the outside-legend branch, across the Hall, temperature-dependent Hall and magnetization
kinds. Each became narrower by closing the dead band; content is unchanged.

## Hall integrity (real-measurement audit, 2026-09-07 to 2026-09-12)

Items 26–37 came from a focused audit of the Hall analyzers' uncertainty handling, prompted by
the same class of gap items 18–20 found: a fit can be reported without ever asking whether its
own uncertainty makes the reported value meaningless. Where an item names a shipped example, run
it as shown; items marked "real file only" reproduce on the corpus' one real Hall-wired
measurement, reached the same way items 18–21 reach it — by logical key, never a filename.

**26. Carrier density, carrier type and mobility were published from an R_H whose own sign was
undetermined.** *FIXED 2026-09-10 (b14a237, 816bb5d): withheld whenever R_H's own sigma
(instrument-preferred, residual fallback) is not strictly smaller than \|R_H\|, under a
machine-readable `r_h_unresolved` flag and a `withheld` field that keeps the values inspectable
without publishing them as measurements; the `carrier_concentration` capability itself now reads
`applicable: false` once every point in a result declines it.* At σ ≥ \|R_H\| the ±1σ interval on
R_H contains zero, so n = 1/(e·\|R_H\|) has no finite upper bound and carrier sign = sign(R_H) —
the thing a Hall measurement exists to determine — is undetermined within that interval;
reporting either as a number asserts a precision the fit does not have. Reproduces on shipped
examples: `cryosweep hall-tdep examples/hall_temperature_dependence.dat --hall-channel 1
--thickness 0.5 --long-channel 2` withheld 15 of 38 points when this fix landed and withholds
all 38 now; `hall_mixed_sweeps.dat` withholds 57 of 130. The other 23 on the first file had
resolved on a residual σ that is zero to float precision — it is noiseless synthetic data — and
a later change stopped counting such a σ as evidence (`sigma_degenerate`); the file carries no
instrument σ to judge them by instead. On the real file, 72 of 138 points withhold. R_H itself
is never withheld — only what is derived from it.

(Item 20's fix note lists `antisym_r_h_missing` as the withholding reason; this entry adds a second, `r_h_unresolved`. That note is incomplete rather than wrong — the two say different things: no R_H was produced at all, versus an R_H that exists but is not resolved against its own σ.)

**27. Mobility's ρ_xx was averaged over the whole field loop, and could be borrowed from an
unrelated temperature.** *FIXED 2026-09-07 (a9afe1b, 3a17eae): ρ_xx for mobility is now the
median of the longitudinal channel's own rows at \|H\| < 50 Oe (`ZERO_FIELD_OE`), queried per Hall
setpoint and declined — never interpolated or clamped across setpoints — when the nearest
zero-field temperature node sits farther than `temp_interval` away.* Two compounding defects,
both on real data: (a) `_long_rho_xx` had no field mask at all, so a magnetoresistive channel's
whole ±H loop folded into the mobility denominator instead of its zero-field value — re-measured
on the real file, mobility rises 49%/3%/27% at 2/5/10 K once corrected, and ρ_xx(T) is now
monotone across 2/5/10 K (previously the 2 K value sat above 5 K). (b) the per-setpoint
interpolator still let `np.interp` clamp or blend across temperatures with no nearby zero-field
row: a 10 K loop with no \|H\| < 50 Oe row of its own was measured receiving 50 K's zero-field ρ_xx
outright — 3× wrong — with `rho_xx_field_oe` stamped as if a real zero-field measurement existed
at 10 K. Real file only.

**28. The mobility decline reason misdiagnosed why mobility was missing.** *FIXED 2026-09-07, never released
(f53bdb0, 0bb194b): reworked as an evidence ladder — report a file-level fact outright,
generalise a per-point cause only when every declining point carries it, describe a mix as a
mix, and assert no per-point cause when there is no per-point evidence to draw one from.* The
reason text named a missing zero-field row for the whole file whenever any point lacked
mobility, even when that point's own ρ_xx had resolved fine and its R_H fit was the actual
cause, or when the file-level ρ_xx signal was clean but no zero-field row fell within
`temp_interval` of any particular setpoint. A wrong diagnosis sends the wrong remedy. Real file
only — no shipped example currently reaches the mixed-cause case.

**29. A NaN instrument sigma at an otherwise-good row could silently move the resistance fit.**
*FIXED 2026-09-10, never released (c45f894): sigma now gets its own independent mask and interpolation source,
so a row's own resistance and field validity determine which rows feed `R_asym`/`R_H`/`r2`,
regardless of what its sigma column contains.* `_antisymmetrize` AND-ed the sigma mask into the
R,H mask, so a row with perfectly finite resistance but a NaN std-dev was dropped from the fit
too. Reproduced on a targeted synthetic case: with H = [-300,-200,-100,100,200,300] and
R = [-3.5,-2.2,-0.7,1.1,2.3,3.6], a NaN sigma at H = -100 alone moved `R_asym(H=100)` from 0.9 to
1.1 by changing what the interpolator returned — resistance and its own std-dev validity do not
track each other on real files. The defect is general, not tied to a specific shipped file.

**30. An unphysical leading data row could enter the antisymmetrized fit undetected.** *FIXED
2026-09-10 (9419dec): `HallCfg.skip_rows` (CLI `--skip-rows auto|N`, GUI field; default `auto`)
drops leading rows before either Hall analyzer runs. `auto` drops a row only where it is provably
corrupt — more than 10⁶× the file's own median on \|R\| or on the reported σ, on any channel; an
explicit N drops exactly N and turns detection off. The count is always reported
(`data.skipped_rows`), and an explicit count still warns when a row it dropped looks physical.
When this fix first landed the default was a fixed 1; no release carried that default.* Some
PPMS runs write a first
data row taken before the measurement bridge has settled — not noisy, not a reading at all.
Measured on a real **resistivity-option** file's corrupted channel — not the Hall-wired
measurement the rest of this section's "real file only" items refer to: row 0 carried a
resistance 10–11 orders of magnitude off the file median, and left unmasked it moved the
published R_H(300 K) from a sound -2.7424e-10 m³/C (r² = 0.669, on the trend set by the
200 K neighbour) to -1.4346e-04 m³/C (r² = 0.002) — one row in 5786 moving R_H by a factor
of 5×10⁵. (R_H is m³/C; Ω·m is resistivity. A Hall coefficient never carries that unit.) On
the Hall-wired file's ordinary channel, dropping row 0 with `--skip-rows 1` still moves results
(eight of nine field-sweep points bit-identical, the ninth 3.28%, inside its own σ band) — a
physical row, which `auto` keeps.
Real files only; full derivation in `docs/physics-reference.md`, which states the same
measurement with the same file attribution.

**31. Missing thickness degraded to a low-confidence result instead of gating.** *FIXED
2026-09-07 (8d7ce2c): both Hall analyzers now return `status: "gated"` with a `gate[]` entry
naming `--thickness` as the remedy; slope-only points still ship in `data`. Exit code for this
case moves 11 → 10.* R_H = slope × thickness, so without a thickness the analyzers measure a
slope, not a Hall coefficient — the old `low_confidence` status carried an empty `gate[]` and no
warning, so the only trace was a capability reason string an agent keying on `gate[]` would
never see. Reproduces on any shipped Hall example run without `--thickness`, e.g. `cryosweep
hall examples/hall_field_sweeps.dat --hall-channel 1 --long-channel 2`.

**32. `--confidence-min` was silently ignored by the temperature-dependent Hall analyzer.**
*FIXED 2026-09-12 (29e76d4): both analyzers now share one `hall_confidence()` helper reading
`confidence_min` from `RunConfig`, the same route every other consumer of that setting already
used.* `hall_tempdep.py` hardcoded its own `"ok if conf >= 0.5"` instead of reading the
configured threshold, so raising `--confidence-min` moved `hall`'s status but silently left
`hall_tempdep`'s alone — harmless at the shared default, a silent fork at any other value.
Re-verified at the current HEAD: `cryosweep hall-tdep examples/hall_temperature_dependence.dat
--hall-channel 1 --thickness 0.5 --long-channel 2 --config <(echo '{"confidence_min": 0.7}')`
now reports `status: "low_confidence"`, exit 11, on a file whose default-config status is `"ok"`
at confidence 0.605.

**33. `hall-tdep`'s confidence was inflated by construction.** *FIXED 2026-09-11 (ea03bb7,
1a8705c): confidence on both Hall analyzers is now `min(fit_quality, resolved_fraction)`; `r2`
is `None` wherever a fit has zero residual degrees of freedom.* Two compounding defects:
`hall_tempdep.py`'s inline antisym fit set `pt.r2 = fit.r2` unconditionally at
`antisym_points == 2`, and a line through exactly two points fits them exactly regardless of how
noisy the data really is — measured on the real file, this was the ONLY source of a non-`None`
r² `hall-tdep` ever reported (16 points), and every one was exactly 1.0, a tautology rather than
a fit-quality measurement. Confidence itself was the fraction of points meeting
`tdep_min_antisym_points` (defaulted to 1 by item 18's own fix), so it read 1.0 by construction
on any file, including one flagged elsewhere in the same report as instrument noise throughout.
This is the direct continuation of item 18: that fix corrected a confidence reading 0.0 on a
correct result; this one corrects the opposite failure, a confidence reading 1.0 on a result
several of whose points cannot be told apart from zero. See item 18's addendum for the real
file's current, correctly lower number.

**34. A naive CSV read of a Hall export could silently return a plausible-looking but wrong
DataFrame.** *FIXED 2026-09-11, never released (d9e0886, 0e0fbd5): the leading `#` warning block now opens with
a short comma-free marker line before any warning prose.* Hall CSVs lead with a `#` block
whenever the run carries warnings — a documented parse-contract change, since Python's `csv`
module has no comment support and `pandas` needs `comment="#"`:

```python
pandas.read_csv(path, comment="#")   # the one-line remedy any reader of these CSVs needs
```

Before this fix the block's first line was warning prose, and warning prose contains commas:
measured on a real export, `pandas.read_csv(path)` (no `comment=`) returned a **(10, 3) DataFrame
of nonsense** and raised nothing, because the comma-bearing warning line split into three
plausible-looking columns. Silent and wrong is the worst combination, and pandas is the most
common reader in this field. With the comma-free marker line, the same naive call now either
raises `ParserError` or returns a single column named by the remedy sentence itself — never a
plausible multi-column frame — verified on this HEAD against both a real export and a shipped
one: `cryosweep export examples/hall_field_sweeps.dat --hall-channel 1 --thickness 0.5
--long-channel 2 --probe hall`, then `pandas.read_csv()` (no `comment=`) on the resulting
`.points.csv` raises `ParserError: Expected 2 fields in line 3, saw 26`. Readers that already
honour `#` — `numpy.loadtxt`, Origin, gnuplot — skip the block like any other comment and need no
change; that is the justification for the convention, not only its cost.

**If you are running a released version, this cannot affect you:** the comment block is introduced by the same unreleased work that fixed its wording, so an export from 0.6.0 carries no `#` block at all and reads cleanly with a plain `pd.read_csv`.

**35. The Hall confidence banner did not name which ceiling bound a `low_confidence` result,
and once it did, the two ceilings' text could read as though one qualified the other.** *FIXED, never released
2026-09-12 (090e8ee, 3650e3d): each ceiling now carries its own label, separated by a
semicolon, and a `None` fit value reads as words rather than a bare dash.* A poor R_xy-vs-B fit
and an R_H not resolved against its own σ are opposite problems with opposite remedies; before
this pair of fixes the banner showed only the number. Verified on the real file at this HEAD —
`status = low_confidence`, `confidence_parts = {"fit": null, "resolved": 0.478}` — the banner
note now reads exactly:

> binding ceiling: resolved fraction 0.478 (R_H distinguishable from zero); other ceiling: fit
> quality — no r² survived the zero-degrees-of-freedom rule

The intermediate wording this replaced (`"binding ceiling: {resolved} ({fit})"`) stacked the
non-binding ceiling in a bare trailing parenthetical that read as though it qualified the
binding one's number rather than naming an independent, non-binding fact. Real file only — no
shipped Hall example currently reaches `low_confidence`.

**36. The field-sweep Hall row displayed only the residual sigma, hiding the instrument sigma
the temp-dep row already showed.** *FIXED 2026-09-12, never released (868e9c1): the field-sweep R_H@T row now
appends the instrument sigma with the same wording used everywhere else in the panel, so the two
families stay labeled apart wherever both appear.* The two σ families answer different questions
(fit scatter vs instrument noise) and are never interchangeable; showing only one on the
field-sweep probe while the temp-dep probe showed both left a reader of the field-sweep GUI
unable to see whether a result was noise-dominated at all. The shipped field-sweep example's own
instrument sigma is small (this file was never the point of the item), but the row now shows it
regardless — verified in `tests/gui/test_uncertainty_rows.py`.

**37. The Hall carrier-density panel's method-boundary caption described declined points as the
output of a fallback estimator.** *FIXED 2026-09-12 (94f46ea): the note now fires only where a
genuine 0-field+1 fallback series is plotted beside a trusted one, not merely whenever a
hollow-marker (`role="two_point"`) series shares the panel with a non-hollow one.* Points whose
carrier density and mobility were declined (item 26) are drawn hollow through the same
`role="two_point"` convention the panel already used for its 0-field+1 fallback estimator,
because that is the only hollow-marker convention a catalog series can reach.
`_estimator_method_note` read that shared visual role as a shared physical meaning and titled the
figure "open = 0-field+1 fallback estimator; steps between estimators are method, not physics"
whenever it fired — which, with the (default-off) declined-points inspection series switched on,
was **every** point genuinely resolved by antisymmetrization: 71 of 72 open markers on the real
file, 55 of 57 on `hall_mixed_sweeps.dat`. It was accidentally true on the synthetic
`hall_temperature_dependence.dat` example, where all 15 open points really are 0-field+1 points
— this item is about the other two files, not that one. Scope is exactly the
`render_hall_tdep_n_t` panel; `render_hall_tdep_mobility_t` never carried this note. At the default plot
selection the declined-points series is off, so **on that panel** no note fires and nothing was
wrong — this item only bites a reader who turns the new inspection series on. The note does fire
at the default selection on the neighbouring `hall_tdep_RH_T` panel, on the real file and on both
shipped temperature-dependent examples, and it is CORRECT there: that panel ships a genuine
`R_H (0-field+1)` series with `default_on=True`, which is exactly the estimator handover the note
was written to warn about.

**38. The topmost data marker was drawn into the axes frame, and whether it was depended on
the canvas size.** *FIXED 2026-09-12: the glyph allowance is now computed from the marker's
actual pixel radius against the measured axes height and applied as a y-margin, on every plot
kind.* Matplotlib's default 5% y-margin is measured to the data COORDINATE; the marker drawn at
that coordinate has a physical size in points, so whether it fits inside the frame is a function
of how large the figure is. `_ensure_top_headroom` had corrected this (item 6) on the two panels
it was reported against — the thermal-transport κ panel and the AC-susceptibility χ panels — and
nowhere else, and it expressed the allowance as a fixed 8% of the data span, which is the wrong
unit for a physical glyph.

Measured on the temperature-dependent Hall R_H(T) panel, as clear space between the topmost
marker and the frame (negative = the marker crosses it):

| | 3 pt | 7 pt | 9 pt | 12 pt | 16 pt | 20 pt |
|---|---|---|---|---|---|---|
| at the shipped 90 × 70 mm | +18.5 px | +10.2 px | +6.0 px | **−0.2 px** | **−8.6 px** | **−16.9 px** |

| | 140 × 110 mm | 90 × 70 mm | 60 × 45 mm | 40 × 30 mm |
|---|---|---|---|---|
| at 7 pt markers | +71.0 px | +10.2 px | **−2.4 px** | **−10.5 px** |

Both are one spin-box away in the GUI: the marker size and the figure width/height are adjacent
controls in the styling panel. Below about 50 × 38 mm the axes box is only a few marker diameters
tall and matplotlib itself reports `constrained_layout not applied because axes sizes collapsed
to zero`; no allowance can place a glyph inside a box that small, and the fix does not pretend
otherwise.

The allowance is applied as a **margin**, not as a limit. `set_ylim` latches the axis
(`autoscaley_on` → False), and two existing guarantees pull against each other under that:
`robust_view=False` must set no limit at all, while the robust view must be a no-op on clean
data. `set_ymargin` widens the view and leaves the axis unlatched, satisfying both — and it is
the right concept, since the defect is exactly that the existing margin is too small for the
glyph. Margins are symmetric, so the lowest marker gains the same allowance; that clipping is the
same defect and had never been reported separately.

At the shipped defaults nothing changes: every figure in the repository renders byte-identically,
including the byte-pinned VSM oracle images, because at 90 × 70 mm with markers of 9 pt or less
the requirement is already below matplotlib's own 5% margin. **Not covered:** a point excluded
because the robust view deliberately narrowed around a heavy tail still sits outside the frame
with no marker of its own — that is a different question (the view is hiding an outlier on
purpose) and remains open.

## Hall figures (open, found while checking the release figures, 2026-09-18)

Items 39–41 are **open** and ship unfixed. Each names what reproduces it, a workaround, and the
fix it points to. None changes a number: the JSON, the CSV and the confidence are unaffected —
these are about what the figures show.

**39. The temperature-dependent mobility figure can push a published point off the axis, with no
marker saying so.** *OPEN.* The robust y-view (on by default) narrows a linear axis to the union
of per-line median ± k·MAD envelopes whenever a curve has a heavy tail, and a point outside that
view is simply not drawn — item 38's closing note records that such a point has no marker of its
own. On these figures every plotted point has already passed the decline rule (item 26), so what
the view hides is a published result rather than a raw glitch. Reproduces on a shipped example:
`cryosweep hall-tdep examples/hall_mixed_sweeps.dat --hall-channel 1 --thickness 0.5
--long-channel 2`, figure `hall_tdep_mobility_T` — μ(19 K) = 2.997×10⁻³ m²/V·s sits above the
axis top (2.980×10⁻³) and the connecting line runs out of the frame; the summary figure
(`hall_tdep_summary`) hides the same point on its μ panel and the 19 K R_H (2.364×10⁻¹⁰ m³/C) on
its R_H panel. On the real file it is worse: the 5 K mobility, 1.554×10⁻³, sits 2.7× above an
axis top of 5.708×10⁻⁴ and is not visible at all — and the field-sweep analyzer's mobility at 2 K
on the same file is 1.54×10⁻³, so the point is physics, not a glitch. **Workaround:** in that
plot's own "▸ controls", untick "Robust view" (`PlotSpec.robust_view=False`) or set the y scale
to log. **Fix it points to:** default the robust view off for the Hall figures whose points are
all decline-filtered, or — generally — mark any point the view excludes at the axis edge with its
value.

**40. The temperature-dependent carrier density is drawn as a line straight across temperatures
whose density was withheld, and without its uncertainty.** *OPEN.* The carrier-density series is
built from the published points only, and the connected-line kinds join consecutive points in
temperature; nothing tells the renderer that withheld points (item 26) lie between them, so the
line implies densities where none was published. On the real file, 66 of 138 temperatures
publish, the line crosses 25 withheld spans, and the longest runs from 5 K straight to 22 K
across 16 withheld temperatures. On the shipped `hall_mixed_sweeps.dat` (command as in item 39),
73 of 130 publish and the line crosses 23 withheld spans. The same happens on the carrier-density
curve of the R_H + n figure (`hall_tdep_rh_n_twin`) and on μ(T), since mobility is withheld at
the same points. Separately, the published densities are drawn as clean points although they are
barely resolved: on the real file their median relative σ on R_H is 0.905, so each has about a
13% chance that its carrier sign is wrong (`carrier_sign_confidence`, median 0.866), and the
exact ±1σ interval on n reaches a median 10.5× — at most 378× — the published value. That
interval is exported (`carrier_n_ci_low`/`carrier_n_ci_high` in the JSON and CSV) but not drawn.
**Workaround:** untick "connect points" (Styling → Journal frame), and read the interval from the
export. **Fix it points to:** break the line wherever a withheld point lies between two published
ones (the renderer already honours such breaks — the VSM ramp split uses them), and draw the
asymmetric interval as error bars; `hall_tdep_n_T` is already log-scaled, where intervals
spanning decades stay readable.

**41. The temperature-dependent Hall stage figures lose their layout past 56 temperatures: the
legend runs off the canvas and the GUI's status header overprints the panel titles.** *OPEN.*
`hall_tdep_stages` and `hall_tdep_asym_vs_B` carry one legend entry per temperature. A legend
moved outside the axes is capped at 14 rows × 4 columns = 56 entries; past that it is taller than
the figure, matplotlib reports `constrained_layout not applied because axes sizes collapsed to
zero`, and **no** layout is applied at all. The axes stay at fixed default positions, so nothing
makes room for the `⚠ low_confidence` header the GUI adds above a low-confidence figure — it
overprints the "Raw" and "Antisym." titles — and the legend runs off the top and bottom of the
canvas; on `hall_tdep_asym_vs_B` the x-axis label is clipped too. At the default style (no
colormap) the colours also repeat every ten temperatures, so even a legend that fitted could not
identify a point. Reproduces on the shipped `hall_mixed_sweeps.dat` (command as in item 39): 128
legend entries, and both figures lose their layout. Checked in the running app: with two plot
cards side by side (each canvas 2.9 in wide) the header overprints both panel titles; a single
full-width card (6.1 in) clears them, but its legend still overflows the canvas. With the legend
switched off, the layout is applied and the header clears both titles. On the real file, 137
entries. **Workaround:** untick "show legend" (Styling → Journal frame). **Fix it points to:**
past the legend's capacity, colour by temperature on a sequential colormap with a colorbar in
place of the legend; figures with 56 temperatures or fewer would be unchanged.
