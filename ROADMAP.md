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

### 2. Debye-Einstein parameters, both directions — **done**

Shipped (see `docs/superpowers/plans/2026-09-04-debye-einstein-two-way-params.md` for the
design and the trap inventory).

**(a) Fitted values into the parameter boxes.** After an **accepted** fit, the boxes show the
fitted values (a declined fit never overwrites the guesses). The `set_state` overwrite trap
is handled at its source: `analyze_and_render` folds each entry's own fitted params into that
entry's stored state, so the restore *is* the delivery; the async and tab-change paths absorb
into the widgets and re-commit. Spinboxes went from 4 to 6 decimals — at 4, γ ≈ 0.0098
truncated to two significant digits and the box could not honestly show what was fitted.

**(b) Live curve response when a parameter is edited.** A model evaluation, not a refit —
**re-measured**: HC `analyze` is 148–241 ms on this machine, a model evaluation plus in-place
line update is **2–4 ms** — drawn as a dashed blue "model (manual)" line on `hc_full_cp_t`
and `cp_vs_t`, `gid="manual_model"`, with the fitted curve untouched beneath it. It is
display-state only: it never enters the analysis result, so CSV/JSON/report and "Export
plots…" (which re-render from the result) cannot carry it; "Save plot" saves the on-screen
figure, where the curve travels **with its "model (manual)" label**. A refit or focus change
clears it. **A hand-tuned curve is never presented as a fit** on any surface.

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

Half of this already exists: the **Analyze button** runs off-thread (`AnalyzeWorker`,
`probe_tab.py:140-152`, joined by `stop_worker` on close, covered by
`tests/gui/test_probe_tab_async.py`). What remains synchronous is `analyze_and_render` — the
path `refit_requested` and file-list changes use (`probe_tab.py:59-60`) — so every
heat-capacity **refit** still freezes the interface for the 148–241 ms measured above, and
item 3 makes refits more frequent. (Item 2(b) deliberately does not: the live curve is a
~2–4 ms model evaluation, not a refit.) Moving `analyze_and_render` onto the same worker
pattern, with a busy indicator, is the remaining fix.

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
