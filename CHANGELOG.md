# Changelog

## 0.1.0 — 2026-09-04

First public release. There is no public history before this version, so this
entry is a summary of what ships, not a diff.

This is a **0.x** release, which under semantic versioning means the interfaces
are not yet frozen. Everything below works and is tested; what 0.x reserves is
the right to change the CLI JSON envelope, exit codes or CSV columns without a
major bump. Tagging **1.0.0** is what freezes them — see "Versioning" in the
README.

### Analysis probes

- **Magnetization (VSM)** — QD VSM and MPMS bare-CSV formats; χ and 1/χ vs T;
  Curie-Weiss fit (θ, C, μ_eff) with a window-sensitivity ladder; multi-field
  M(T) segmentation (one physical field, one curve) and M(H) loops; a 1/χ
  display guard where χ crosses zero.
- **AC susceptibility (ACMS)** — χ′/χ″ per (frequency, amplitude, field) group;
  superconducting screening-step detection with T_c onset/mid; χ″ peak /
  freezing temperature; molar normalization when molar mass and sample mass
  are supplied.
- **Thermal transport (TTO)** — κ, Seebeck, ρ, ZT; Wiedemann-Franz
  decomposition (κ_e, κ_ph) and Lorenz ratio; κ_ph power-law fit with window
  ladder and method cross-check; RRR ± σ; thermal-gradient and Seebeck
  sign-oscillation integrity warnings; opt-in ±1σ error bands.
- **Resistivity** — QD Resistivity and AC Transport formats; ρ(T)/ρ(H) from
  geometry or the instrument column; RRR ± σ; magnetoresistance per field
  loop; resistive superconducting T_c with a narrowness gate; power-law /
  Fermi-liquid low-T fit with cutoff ladder.
- **Heat capacity** — Debye-Einstein full-range Cp(T) fit; low-T Cp/T vs T²
  model family (Debye T³, T³+T⁵, spin-fluctuation); multi-field engine;
  opt-in Schottky anomaly; entropy S(T) with an R ln(2J+1) match verdict;
  transition search.
- **Hall effect** — antisymmetrized R_xy(H), R_H, carrier density, mobility;
  temperature-dependent R_H(T); the two σ families (fit residual vs
  instrument noise) reported separately.

### Interfaces

- `cryosweep` CLI: one JSON object per command, meaningful exit codes
  (0 ok / 10 gated / 11 low-confidence / 2 bad input), byte-stable output,
  JSON Schemas, tidy CSV export, publication plot rendering (PNG/PDF/SVG),
  batch pipelines. An agent guide ships in `skill/cryosweep/SKILL.md`.
- `cryosweep-gui`: PySide6 desktop app on the same Qt-free analysis core.

### Design commitments

- Fits **decline** rather than report a number that is not a measurement
  (search-bound pins, unresolved exponents), and window-sensitive results say
  so instead of quoting a bare σ.
- Required inputs the file does not carry (molar mass, sample mass, thickness)
  **gate** the analysis with a named remedy instead of guessing.
- Runnable example data for every probe in `examples/`; known defects — display,
  ergonomics, and two Hall correctness bugs found on real data — are documented
  in `KNOWN-ISSUES.md` rather than hidden.
