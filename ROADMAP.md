# Roadmap

What is planned, and roughly what it costs. While the version is **0.x** the CLI JSON
envelope, exit codes and CSV columns are not frozen; tagging **1.0.0** is what freezes them,
so the items below are the work that should land before that happens.

Effort figures are estimates unless marked **measured**, in which case a number was obtained by
running the code rather than by reading it. Nothing here is a commitment to a date.

## Targeted for 1.0 — GUI

### 1. "Save plot" should save the plot you are looking at

Today it always saves the **first** plot in the layout, whatever is on screen: the figure it
saves is captured once at render time (`cryosweep_gui/output_panel.py:670`, guarded by
`and self.last_figure is None`), while Focus navigation moves a separate index
(`output_panel.py:476`). Stepping through plots in Focus mode does not change what the button
writes.

Note that **choosing which plots to save is already implemented** — the neighbouring
**"Export plots…"** button (`probe_tab.py:74`, `cryosweep_gui/export_dialog.py`) opens a dialog
with a checkbox per plot, PNG/PDF/SVG, DPI, tight-crop, a filename prefix and exact-mm sizing.
The gap is only in the single-plot button.

*Estimated effort: 2–4 hours. Low risk.*

### 2. Debye-Einstein parameters, both directions

Two independent changes, listed separately because they cost very differently.

**(a) Fitted values into the parameter boxes.** After the 7-parameter Debye-Einstein fit runs,
the edit boxes should show what was fitted instead of the starting guesses. The trap: writing
to the widgets alone is not enough, because `analyze_and_render` restores the panel from the
focused entry's stored state afterwards (`probe_tab.py:314-315`) and would overwrite it. The
stored state has to be updated too.

*Estimated effort: ~0.5 day.*

**(b) Live curve response when a parameter is edited.** A full re-analysis per keystroke is not
viable — **measured**: a heat-capacity `analyze` takes **150–470 ms** (155–245 ms on this
machine, up to 470 ms under load) and runs **synchronously on the GUI thread**. Evaluating the
model and updating the existing line in place, including a full canvas redraw, is **~30 ms**
(29–35 ms measured across runs) — roughly an order of magnitude cheaper, and fast enough to
follow typing.

So the live response should be a **model evaluation, not a refit**. The planned shape: draw the
hand-set curve as a separate dashed "model (manual)" line, leave the fitted curve untouched
beneath it, and keep exports rendering from the analysis result. That last part is the point —
**a hand-tuned curve must never be exportable as a fit.** A figure that says "fit" is a claim
about the data, and the UI has to keep the two visually and structurally distinct.

*Estimated effort: 1.5–2.5 days. The labelling design matters more than the code.*

### 3. Entropy fit off by default

The heat-capacity tab should not compute or show the entropy panel unless it is switched on.
Scope is **GUI only**: the CLI, the JSON envelope and the exported CSVs stay byte-identical, so
nothing downstream of the app changes and no goldens move. The entropy block sits at
`cryosweep_core/analyzers/hc.py:388-448`; gating it behind a config flag that defaults to on
keeps the headless path unchanged while the GUI checkbox defaults to off. The plot kind then
disappears from the cards, the plot checklist and the capability banner through the capability
gating that already exists — no new machinery.

This is a **clarity change, not a performance one**: `compute_entropy` was **measured at
~0 ms**, so nothing is being switched off to make the app faster. It also silences the entropy
warning banner on samples with no magnetic entropy, where the warning is correct but not useful.

It **hides** rather than fixes [KNOWN-ISSUES](KNOWN-ISSUES.md) item 12 — the legend on that
figure lists a curve that was never drawn. Building the legend from the artists actually added
to the axes is a separate ~1 hour change and should land in the same release.

*Estimated effort: 0.5–1 day, plus ~1 hour for the legend fix. About 4 existing GUI tests
touched and ~3 added.*

### 4. Analysis off the GUI thread

`refit_requested` and file changes both call `analyze_and_render` synchronously
(`probe_tab.py:59-60`, `:304`). Every heat-capacity refit therefore freezes the interface for
the 150–470 ms measured above, and items 2 and 3 both make refits more frequent. Moving
analysis to a worker with a busy indicator is the underlying fix, and doing it first would make
item 2(b) simpler rather than harder.

*Estimated effort: not yet costed — this needs a design pass before an estimate is meaningful.*

## Targeted for 1.0 — analysis

### `current_density_J`

[KNOWN-ISSUES](KNOWN-ISSUES.md) item 21: the field is declared, consumed by two plot series,
and never assigned. Resolving it means either implementing J = I/A — which needs the excitation
current column canonicalized and a cross-sectional area the Hall probe currently has no input
for — or removing the wiring. **The decision is deliberately still open**; a costed
recommendation is being prepared, and shipping a half-wired feature past 1.0 is not acceptable
either way.

### Known display issues

The display and ergonomics items in [KNOWN-ISSUES](KNOWN-ISSUES.md) are also 1.0 targets. They
are recorded there with the example file that reproduces each, so they can be picked up
individually.

## Not planned for 1.0

Analyses that are recognized but not implemented are reported as `applicable: false` in each
result's `capabilities[]` list, with a reason. That list is the authoritative statement of what
the app does not yet do — spin-glass relaxation fits, the Callaway phonon model, activated
transport, and others. They are deferred, not forgotten, and none of them blocks 1.0.
