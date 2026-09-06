# Roadmap

What is planned, and roughly what it costs. While the version is **0.x** the CLI JSON
envelope, exit codes and CSV columns are not frozen; tagging **1.0.0** is what freezes them,
so the items below are the work that should land before that happens.

Effort figures are estimates unless marked **measured**, in which case a number was obtained by
running the code rather than by reading it. Nothing here is a commitment to a date.

## Targeted for 1.0 — GUI

### 1. "Save plot" should save the plot you are looking at — **done**

Shipped with the known-issues wave (KNOWN-ISSUES #22): `last_figure` is now a derived
property — the focused card's figure in Focus mode, else the first card that has one — so
the button writes exactly what is on screen. It used to be captured once at render time
(first card only, guarded by `and self.last_figure is None`), while Focus navigation moved
a separate index.

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

### 3. Entropy fit off by default — **done**

Shipped as designed: `HeatCapacityCfg.entropy_enabled` defaults **on**, so the headless
path (CLI, JSON data, exported CSVs) is unchanged — hashed before/after on both
heat-capacity examples: every data CSV and report is byte-identical, and the analyze JSON
differs only by the new key appearing in the provenance **config echo**, as any new config
field must. The GUI checkbox ("Compute entropy S(T)", Entropy group) defaults **off** and
gates the other entropy controls; with it off the analyzer skips the entropy block
entirely, so the `hc_entropy_vs_t` card, its plot-checklist entry and the entropy warnings
all disappear through the existing empty-series capability gating — no new machinery.

This was a **clarity change, not a performance one**: `compute_entropy` re-measured at
~1 ms of a ~237 ms analysis. It also silences the entropy warning banner on samples with
no magnetic entropy, where the warning is correct but not useful.

### 4. Analysis off the GUI thread — **done**

The design pass confirmed the worker-reuse case, and it shipped: `refit_requested` and
file-list changes now go through `request_analyze_and_render` — preparation (widget reads)
stays on the GUI thread, the per-file analyses run on `BatchAnalyzeWorker`, and rendering
happens on queued delivery, with the same busy indicator as the Analyze button and with a
request arriving mid-flight **coalesced** into one rerun rather than dropped (a dropped
refit would leave a stale display standing). `analyze_and_render` itself remains the
synchronous seam. Responsiveness is measured, not assumed: `tests/gui/test_refit_async.py`
counts GUI event-loop turns during the refit flight — ~21 turns during a ~240 ms analysis
after the fix, zero on the old wiring (the test fails on it). Item 2(b)'s live curve is
untouched. Still synchronous, out of this item's scope: `MainWindow._reanalyze_active`
(tab switch / unit change / style change), the remaining candidate for the same pattern.

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
the app does not yet do — spin-glass relaxation fits, the Callaway phonon model, Mott
variable-range hopping as a fitted model, and others. They are deferred, not forgotten, and none of them blocks 1.0.
