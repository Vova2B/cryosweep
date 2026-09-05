# Changelog

## 0.3.1 — 2026-09-05

Three fixes to the surfaces a *user* or an *agent* meets first. No analysis behaviour
changed; every number this release reports is the number 0.3.0 reported.

### Fixed

- **The agent skill taught a fabricated molar mass.** `SKILL.md` gave
  `--molar-mass 200 --mass-mg 5` as a gate remedy with nothing marking the numbers as
  placeholders, so an unattended agent following it got `status: ok`, a confident
  `mu_eff`, and no warning — from an invented molar mass. Remedy examples are now
  labelled **syntax, not values**, with an instruction to stop and ask when the real
  value cannot be obtained. This is the failure the gating discipline exists to prevent,
  and the documentation was defeating it.
- **A missing `--hall-channel` errored instead of gating.** It returned
  `status: "error"`, exit 2 and an empty `gate[]`, while every other missing user input
  returns `status: "gated"`, exit 10 and a remedy naming the flag. Hall now follows the
  same contract, and so does `hall-tdep`.
- **Option files failed silently.** `--layout-file` and `--style-file` ignored unknown
  keys, so a wrong shape or a single typo (`errorband` for `error_band`) produced exit 0,
  no message, and a figure quietly missing the requested feature. Both now warn, naming
  the offending path and the expected shape. Warning rather than rejecting, so files
  written against other versions keep loading.
- **README shell blocks could not be copy-pasted.** The install commands carried trailing
  `#` comments; zsh does not enable `interactive_comments` for interactive shells, so
  `pip` received `#` as a package name. macOS ships zsh. Comments are out of the
  pasteable blocks entirely — putting them on their own line does not help, since a
  leading `#` is `command not found` under the same option. Two commands calling bare
  `python`, which does not exist on macOS, now use `.venv/bin/python`.
- **The MPMS example was documented as a sample it is not.** The docs said to analyze it
  with 5 mg and advertised `C = 3, mu_eff = 4.90`; the file is built at 10 mg, which
  gives `C = 1.5, mu_eff = 3.46`. Both readings are self-consistent — halving the mass
  doubles the inferred C — but a gating example exists to teach "supply the real inputs"
  and should not ship inputs that are not the file's. Fixed in the generator, so the next
  regeneration cannot revert it.

### Added

- **`--config FILE`** loads a `RunConfig` JSON (`cryosweep schema config`). The CLI could
  previously set only unit system, geometry, Hall and probe override, which left the
  heat-capacity Schottky fit, the transition search and `quality.exclude_outliers`
  permanently unreachable from a headless run — they are opt-in masters that no CLI flag
  could turn on. Precedence: an explicit flag beats the file, the file beats the default.
- The skill documents the `--layout-file`/`--style-file` shapes, how to pick
  `--hall-channel` on an unfamiliar file, `run` pipeline semantics (only `detect` and
  `analyze` are legal steps; one bad step aborts all of them), and that `examples/` ships
  in the repository but not in the wheel.

## 0.3.0 — 2026-09-05

Ten pull requests of analysis and figure work on top of the first published release. The
theme is the same one the project already applies to numbers, now applied to figures: say
what was measured, and place things by measurement rather than by a pinned constant.

### Added

- **Arrhenius activated-transport fit** for insulating ρ(T) (`activated_transport`), for
  semiconductor gap analysis. The intrinsic/extrinsic factor-of-two is the classic way to
  misreport this: intrinsic conduction gives ρ ∝ exp(+E_g/2k_BT), so a fitted E_a is half
  the gap, while an extrinsic donor/acceptor level carries no such factor — and transport
  alone cannot tell you which you have. The assumption therefore rides in the field name,
  `e_g_assuming_intrinsic_mev`, instead of in a footnote. Reported with a window ladder,
  because on variable-range-hopping data the Arrhenius fit reaches r² = 0.968 while E_a
  drifts 24 → 45 meV across windows: r² cannot warn you, only the drift can. New example
  `examples/resistivity_semiconductor.dat`.
- **Fit curves extrapolate to their 0-intercept.** γ on `cp_over_t`, θ via the Curie-Weiss
  line on `inverse_chi`, ρ₀ on the resistivity kinds are numbers the annotation already
  claimed in text while the drawn curve stopped at the first measured point. The
  continuation is dotted, thinner and half-alpha so it never reads as fitted range, and is
  omitted where 0 K is not on the abscissa (`resistivity_arrhenius` plots against 1000/T).
  The measured data is untouched.
- **Excitation current** on temperature-dependent Hall results, with **current density J**
  gated on sample width and thickness rather than assumed.
- **Anonymised real Hall measurement** (`examples/hall_mixed_sweeps.dat`) — public
  regression data for three known issues that previously had none.
- Debye-Einstein parameters flow **both directions** between the GUI fit and its inputs.

### Changed

- **The fit-window shade is opt-in and off by default** (`PlotSpec.fit_window_shade`,
  reachable from the GUI and from a CLI `--layout-file`). It was drawn unconditionally, and
  on `hc_c_over_t_linear` it covered essentially the whole panel.
- **Placement is measured, not pinned.** The legend, the low-T inset, reference-line labels
  and the frameless stats boxes now choose their position from measured occupancy — data
  points, other text, insets, and (for the stats boxes) reference lines. Where nothing is
  clear, the inset drops out with a note rather than covering the curve. Positions that were
  already clear are unchanged: 100 of 101 example × plot-kind renders are byte-identical.

### Fixed

- **An unphysical γ carries its verdict onto the figure**, not only into the data. On a real
  heat-capacity file γ fits to −8.3×10⁻³ J/mol·K²; the annotation says so, and the
  extrapolated curve now visibly crosses zero to show it.
- **Reference lines no longer run through text.** A vertical T_c guide crossed the
  `RRR = 86.7 / T_c = 8.03 K` stats box on the superconductor example.
- KNOWN-ISSUES 1, 4, 5, 9, 11, 12, 21 and 23 — all instances of placement decided without
  measuring.

### Internal

- Four new gates covering the defect classes above, each written to prove it can see:
  text-over-data, text-over-reference-line, legend occupancy and inset occupancy. The
  reference-line gate uses segment-vs-rectangle intersection, because vertex containment is
  blind to it — an `axvline` has two vertices, both outside a mid-panel box, while the
  segment between them crosses it, and a point-in-box scan reports a clean zero with the
  defect on screen.

## 0.2.1 — 2026-09-04

### Fixed

- The version disagreed with itself. `pyproject.toml` and `CITATION.cff` were
  bumped to 0.2.0 while `cryosweep_core/__init__.py` was left at 0.1.0 and the
  CHANGELOG had no 0.2.0 entry, so `cryosweep --version` reported the wrong
  number and `test_version_consistency` failed. **0.2.0 was released with that
  inconsistency**; its archive reports 0.1.0 and does not pass its own test
  suite. This release is the corrected one — prefer it over 0.2.0.

## 0.2.0 — 2026-09-04

First *published* release. 0.1.0 was tagged but never released, and it predates
the fixes below — which is why the DOI is minted here rather than there.

### Fixed

- **Golden fixtures were being LF-normalised**, so byte-exact comparisons passed
  in the working tree that wrote them and failed in every fresh clone
  (`*.golden -text` in `.gitattributes`).
- **A residual made of pure rounding was reported as a lambda anomaly.** On a
  featureless curve the quartic fits to machine precision, so the linear-algebra
  backend decided whether a phase transition existed: it declined on macOS and
  fired on Linux. Now gated on an absolute floor relative to the data span.
- Numeric oracles compare by tolerance rather than serialized text, which had
  pinned them to one machine's BLAS.

### Changed

- **Qt is an optional extra.** `pip install cryosweep` gives the core and CLI
  with no Qt at all; `pip install 'cryosweep[gui]'` adds the desktop app. The
  dependency is `PySide6-Essentials` rather than the `PySide6` meta-package,
  which drags in 175 MB of Addons this codebase never imports.
- `cryosweep-gui` without the extra prints how to install it instead of raising
  a ModuleNotFoundError traceback.
- Every GitHub Action is pinned to a commit SHA, with Dependabot proposing moves
  so the pins cannot rot. CI runs Python 3.11 and 3.12.
- README links are absolute, so they resolve on PyPI instead of 404-ing, and the
  README now shows the application.

### Added

- Release publishing to PyPI via Trusted Publishing (OIDC, no stored token).
- `.zenodo.json`, so the DOI record carries the author and ORCID.

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
