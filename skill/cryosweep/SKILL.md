---
name: cryosweep
description: Drive the headless cryosweep CLI to analyze Quantum Design PPMS/MPMS .dat files across seven probes (vsm, heatcapacity, resistivity, hall, hall_tdep, acms, tto), fit physics models, export tidy CSVs and publication plots — for unattended/agent use. Covers the JSON envelope, exit codes, gating remedies, and the fit-decline discipline an agent must respect when parsing output.
---

# cryosweep — agent guide

Installed (`pip install .`) the console script is `cryosweep`; from a source checkout run
`python -m cryosweep_cli` — identical CLI. Every command except `report` prints ONE JSON
object on stdout (sorted keys, strict RFC-8259 — never bare NaN/Infinity, so byte-stable
and safe to diff/cache); logs go to stderr; the exit code encodes `status`.

Probes: `vsm`, `heatcapacity`, `resistivity`, `hall`, `hall_tdep`, `acms`, `tto`.
`cryosweep probes` is authoritative — the registry is the contract, this list is documentation.

## Commands

| command | output |
|---|---|
| `cryosweep detect <file>` | `data: {probe, score}` — detected probe + score |
| `cryosweep analyze <file>` | full `Result` envelope (`fit` is an exact alias) |
| `cryosweep hall <file> --hall-channel N` | analyze with probe forced to `hall` (`hall-tdep` → `hall_tdep`) |
| `cryosweep export <file> --out stem` | analyze + write CSVs; `data.exported` maps name → path |
| `cryosweep report <file>` | Markdown on stdout (NOT JSON); exit code still per status |
| `cryosweep plot <file> --out stem` | analyze + render; `data.plot`/`data.plots` list the files |
| `cryosweep probes` / `fits` / `plots` / `observables` | all four print the SAME registry dump `{probes, fits, plots, observables}` |
| `cryosweep schema <name>` | JSON Schema; names: `result`, `fit`, `config`, `analyze:vsm`, `analyze:hc`, `analyze:resistivity`, `analyze:hall`, `analyze:hall_tdep`. Bad/missing name → usage on stderr, exit 3 |
| `cryosweep run pipeline.json` | `{"steps": [{"command": "analyze", "file": "a.dat"}, ...]}` → `{"results": [...], "exit": <worst step>}` — worst by SEVERITY (error > gated > low_confidence > ok), NOT by numeric code. ONLY `detect`/`analyze` are legal step commands (step `options`: `molar_mass`, `mass_mg`, `unit_system`); any other command fails validation and ABORTS the whole pipeline (`results: []`, exit 2). To batch export/plot, loop the shell over `cryosweep export`/`plot` instead |

## Result envelope + exit codes (branch on BOTH)

Envelope keys: `{status, confidence, confidence_parts, data, diagnostics, warnings, gate, errors, provenance}`.

| status | exit | meaning | recovery |
|---|---|---|---|
| ok | 0 | usable result | — |
| gated | 10 | a required input is missing | each `gate[]` entry carries `need`, `reason`, `remedy.flag` + `remedy.example` — re-run with the flag |
| low_confidence | 11 | result emitted, but read `warnings` before trusting it | e.g. CW window reaches below \|θ\| → inspect `cw_ladder` |
| error | 2 | bad/unreadable input | fix the file/args |

Exits 10/11 STILL print a full envelope — parse it, don't treat them as hard failures.
Even `error` prints an envelope (with `errors[]`), never a bare traceback.
`export` on a gated file still exits 10 and writes the CSV files, but they hold headers/no
rows — supply the gate remedies to get data.

**Remedy examples are SYNTAX, not values.** `remedy.example` (e.g. `--molar-mass 200.0`)
shows how to spell the flag; the NUMBER must come from the actual sample (its formula
weight, measured mass, measured thickness). Plugging in the example number produces
quantitatively wrong physics (χ_mol and μ_eff scale with it) with no warning — the one
failure this tool exists to prevent. If you cannot obtain the real value, STOP AND ASK;
never substitute a plausible one.

## Options

- Common: `--out STEM` (default `cryosweep_out`), `--unit-system CGS|SI` (default CGS), `--molar-mass G_PER_MOL`, `--mass-mg MG`
- Resistivity geometry (switches ρ to recompute): `--width-mm`, `--thickness-mm`, `--length-mm`
  (`--width-mm` also feeds the hall-tdep current-density-J capability)
- Probe override: `--probe KEY` (e.g. force `hall` on a resistivity-format file with `export`/`plot`)
- Hall: `--hall-channel N` (required; omitting it gates with a remedy), `--thickness T --thickness-unit mm|um|nm`, `--long-channel M` or `--long-file F` (un-gates mobility), `--geometry-sign 1|-1`, `--temp-interval K` (hall-tdep binning)
- `--config FILE` — a RunConfig JSON (`cryosweep schema config` prints its schema). This is
  the ONLY route to the heat-capacity controls (`heatcapacity.schottky_enabled`,
  `transitions_enabled`, fit windows, `entropy_*`, `full_init`/`full_fixed`) and to
  `quality.exclude_outliers`; explicit flags override the file per key.
  E.g. `{"heatcapacity": {"schottky_enabled": true}}`.
- Plot: `--plot-kind KEY` (default: probe's default kind; keys from `cryosweep plots`), `--all` (every kind → `<prefix>_<kind>.<fmt>`; mutually exclusive with `--plot-kind`), `--format png,pdf,svg` (comma list, default png), `--dpi N`, `--tight`, `--style-file JSON`, `--layout-file JSON`
- An unavailable plot kind is NOT an error: `data.plot` is null and a warning explains — check it.

### --layout-file / --style-file shapes

`--layout-file` is a PlotLayout: per-plot settings live under each entry's `"spec"`
(fields: `xmin/xmax/ymin/ymax`, `xscale/yscale`, `curves`, `error_band`,
`fit_window_shade`, `robust_view`, `width_mm/height_mm`, …). `--style-file` is a flat
GlobalStyle (`grid`, `legend_loc`, `font_pt`, `field_unit: "Oe"|"T"`, …). Both are defined
in `cryosweep_core/plotting/spec.py`.

```json
{"plots": [{"kind": "tto_kappa_t", "spec": {"error_band": true}}]}
```

Unknown keys (a typo, or a spec boolean at the wrong nesting level) are IGNORED with a
warning in `warnings[]` and on stderr — after supplying either file, check `warnings[]`
before claiming the figure has the requested feature.

### Choosing --hall-channel on an unfamiliar file

The Hall bridge is the one whose resistance is odd in B (sign flips with field). The GUI
auto-detects it; the CLI does not — but `cryosweep analyze <file>` (as resistivity) runs
the same detector when the file contains field sweeps: `data.excluded_hall_channel` is the
detected Hall-wired bridge (`excluded_hall_source: "detected"`). On a pure
temperature-sweep file nothing can be detected — the channel must come from the
measurement notes; do not guess.

## DECLINE discipline — read flags before numbers

Some fits refuse to report a number that is not a measurement. Blank CSV cells there are
deliberate, not a parse error; the flags column carries the machine-readable reason.

- **Resistivity power-law** (ρ = ρ₀ + A·Tⁿ): decline flags `n_at_bound` (n pinned at the
  [0.5, 6.0] search bound) and `n_unresolved` (σ_n ≥ |n|) ⇒ CSV cells `power_law_n`,
  `power_law_A`, `power_law_r2`, `residual_rho_ohm_cm` are BLANK and `power_law_flags`
  carries the reason; in JSON `residual_rho` and `power_law_n_spread` are null while the
  `power_law` object keeps `r2`/`quality_flags` for transparency.
  `rho0_unresolved` is NOT a decline: n is still real, only ρ₀ and the fit line are withheld.
- **TTO κ_ph power fit**: declines on `n_at_bound` or `degenerate_window` (or r² ≤ 0);
  `kappa_e_dominant` and `window_sensitive` describe a real fit and do NOT decline.
- Verified demo: `export examples/resistivity_superconductor.dat` → channel 1 (the
  superconductor: its ≤30 K window has no power-law regime) has all four cells blank with
  `power_law_flags = n_unresolved;ladder_incomplete`; channel 2 reports n with
  `window_sensitive`.

## Window-sensitivity ladders — spread ≠ error bar

Fits are re-run across fit-window rungs; `spread = max−min` across rungs is reported next
to the statistical σ and can dwarf it (channel 2 above: `power_law_n_spread` 0.59 vs
σ_n 0.11). The spread is window sensitivity, not an uncertainty — quote both.

- VSM Curie-Weiss: `cw_ladder` + `theta_spread_k` / `mu_eff_spread`
- Resistivity: `power_law_ladder` + `power_law_n_spread` (bound-pinned rungs stay listed with `at_bound: true` but are excluded from the spread)
- TTO κ_ph: `kappa_ph_fit.ladder` + `n_spread`, plus `n_loglog`/`n_method_delta` (second method)
- Fewer than two resolved rungs ⇒ spread is `null` (never 0.0) + `ladder_incomplete` flag.

## capabilities[]

Resistivity, hall, hall_tdep, acms and tto results carry `data.capabilities`: a list of
`{name, applicable, reason}`. `applicable: false` covers both "input missing" (e.g. hall
`mobility` without a longitudinal channel) and recognized-but-not-yet-implemented analyses
(e.g. tto `callaway_fit`). Never assume an analysis ran — read this list.

## Try it

`examples/` ships one runnable file per scenario (details in `examples/README.md`) — in
the REPOSITORY only, not in the wheel. After a plain `pip install`, fetch one first:
`curl -O https://raw.githubusercontent.com/Vova2B/cryosweep/main/examples/heat_capacity.dat`
(same form for every file named below).
`magnetization_vsm.dat` (CW θ = −10 K), `magnetization_mpms.dat` (bare-CSV MPMS — gated
until `--molar-mass 200 --mass-mg 5`; those are this synthetic file's true values, NOT
numbers to reuse on real data), `magnetization_vsm_multifield.dat` (real, exits 11
by design), `heat_capacity.dat` + `heat_capacity_multifield.dat`, `ac_susceptibility.dat`,
`resistivity_superconductor.dat` (Tc detector + decline demo), `resistivity_semiconductor.dat` (Arrhenius E_a = 60 meV; the gap column is named
`e_g_assuming_intrinsic_mev` because E_g = 2·E_a only if intrinsic), `thermal_transport.dat`,
and `hall_field_sweeps.dat` / `hall_temperature_dependence.dat` — run those two as
`cryosweep hall|hall-tdep <file> --hall-channel 1 --thickness 0.5 --long-channel 2`.
