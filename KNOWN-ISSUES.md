# Known issues

Items 1–17 were found by rendering every file in `examples/` — first the default kind, then
**all 55 plot kinds** — and inspecting the output, 2026-09-01. Items 18–21 came from running
the Hall analyzers against a real measurement, and item 22 from a GUI investigation, both
2026-09-02. Item 23 was found 2026-09-04 by looking at the
figures produced while verifying the item-1 fix. They are recorded here rather than quietly
left because the project's own rule is that a result you cannot trust is more useful reported
than hidden.

**Items 1–18, 21–22 and 23 change no fitted number and no value in an exported CSV; items 19
and 20 do.** Items 1–8 and 11–16 are display; items 9 and 18 are reporting-honesty gaps; item 10
is CLI ergonomics; item 17 hides a GUI control and item 22 makes one act on the wrong target;
items 19 and 20 are correctness bugs; item 21 is an unfinished feature.

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
`examples/hall_temperature_dependence.dat`: `R_H (antisym)` covers 18–40 K at −3.0e−7 m³/C and
the `R_H (0-field+1)` fallback covers 41–55 K at −2.5e−7, producing an apparent **20 % step at
~40 K that is a change of method, not physics**. The legend distinguishes them; nothing warns
the reader not to read the discontinuity as a result.

**3. Axis offset notation is unreadable at Hall magnitudes.**
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

**6. No top headroom.** The κ peak (`thermal_transport.dat`, panel a) and the χ′ high-T plateau
(`ac_susceptibility.dat`, top panel) touch the axes frame.

**7. Large fields are labelled in Oe.** Legends read `90000 Oe` / `100000 Oe` rather than 9 T /
10 T. There is a global Oe↔T display toggle and Oe is the deliberate default, so this is a
default-choice question, not a bug — but at these magnitudes it costs readability.

**8. A zero-field curve omits its field in TTO legends.** `thermal_transport.dat` reads
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

**10. `cryosweep plots <file>` ignores the file.** `probes`, `fits`, `plots` and `observables`
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

**13. The `vsm_mh` low-field panel does not rescale its y-axis.**
`magnetization_vsm_multifield.dat`: the right-hand "low field" panel inherits the full-range
y-limits (0–0.55 µ_B) while its data spans 0–0.06, so the zoom panel is ~80 % empty and shows
a short line in one corner — the opposite of what a zoom panel is for.

**14. Field setpoint labels print raw floats.** `heat_capacity_multifield.dat`,
`hc_lowt_multifield`: the legend reads `0.524968 Oe`, `50000.5 Oe`, `100001 Oe`, `130000 Oe`
for what `examples/README.md` correctly calls 0 / 5 / 10 / 13 T. Nominal zero field is printed
to six significant figures. Setpoint labels should be rounded and, at these magnitudes, shown
in tesla.

**15. Multi-field low-T fits are unreadable.** Same figure: four fields × four low-T models
draws sixteen curves whose colours do not match their data series, several of which overshoot
the axes entirely. Which fit belongs to which field cannot be read off the plot.

**16. `tto_lorenz_t` cannot show the thing it exists to show.** `thermal_transport.dat`: L/L₀
diverges at low T, so the linear y-axis runs to ~200 (`×10²`), the curve is clipped at the top,
and the **Wiedemann-Franz reference line at L/L₀ = 1 — the entire point of the panel — is
flattened onto the bottom axis**, where its annotation also collides with the data. This panel
wants a logarithmic y-axis.

## GUI

**17. The "Colour…" button is clipped out of the left panel at the default width.**
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
2026-09-02 (cc43b8c): a single ± pair now fits as an antisym point anchored at
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

**19. A drifting temperature setpoint is split in two, and the split fabricates a carrier
density.** *FIXED 2026-09-02 (a72ca20): `cluster_field_setpoints` adopted for held
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
2026-09-02 (41019b5): Stage C now derives only from Stage B; when Stage B declines,
carrier_n/carrier_type/mobility are withheld and `derived_flags = ["antisym_r_h_missing"]`
is carried into JSON and a new CSV column; R_H_raw stays visible.*
`hall.py:239` falls back to `R_H_raw` (the Stage A raw fit) for `carrier_n` and `mobility`
while `pt.R_H` — the trusted Stage B antisymmetrized value — stays `None`. So a row can export
a carrier density and a mobility with an empty Hall coefficient, which is what makes item 19
visible in the CSV rather than merely in a plot. Under the project's own decline discipline, a
derived quantity whose parent was not measured should be withheld, not published from the
untrusted stage.

**21. `current_density_J` is declared and consumed but never assigned.** Not a defect in a
result — a feature that is wired up at both ends with nothing in the middle.
`hall_tempdep.py:52` declares the field; `catalog.py:901-932` builds two plot series from it;
`render.py:2295` already documents it as "always None on ..." — and a search of
`cryosweep_core`, `cryosweep_cli` and `cryosweep_gui` finds **no assignment anywhere**.
Consequences: `hall_tdep_J_T` renders zero series, and `hall_tdep_summary` silently degrades
from three axes to two. Implementing it needs `Bridge N Excitation (uA)` canonicalized (it
currently has zero hits in `cryosweep_core`) and the honest quantity is J = I/A.

**22. "Save plot" saves the first plot, not the one on screen.** The figure that button writes
is captured once while the layout renders — `cryosweep_gui/output_panel.py:670` assigns
`last_figure` from the first card that has one, guarded by `and self.last_figure is None`, and
nothing updates it afterwards. Focus mode steps a separate index (`output_panel.py:476`), so
navigating to a plot and pressing **Save plot** silently writes a different one. Reproduces on
any file with more than one plot in the layout.

Choosing plots explicitly *does* work: the neighbouring **"Export plots…"** button
(`cryosweep_gui/probe_tab.py:74`) opens a dialog with a checkbox per plot, PNG/PDF/SVG, DPI,
tight crop and exact-mm sizing. Use that until this is fixed. Scheduled for 1.0 — see
[ROADMAP.md](ROADMAP.md).

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
