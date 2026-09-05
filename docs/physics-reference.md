# Physics reference — cryosweep

The models, fitted quantities and detection criteria this program implements, probe by probe.
Numbers quoted as measured come from real measurement files that are not distributed with the
source; the physics is reproducible from the formulas here.

### Magnetization (VSM)

- **Magnetic susceptibility**: χ = M/H (moment per unit field)
- **Inverse susceptibility**: 1/χ — linear in T for Curie-Weiss behavior
- **Curie-Weiss law**: 1/χ = (T − θ) / C
  - C = Curie constant
  - θ = Weiss temperature (positive = ferromagnetic, negative = antiferromagnetic)
- **Effective moment**: μ_eff = 2.827√C (CGS) or μ_eff = 797.8√C (SI)
- **Units**: CGS (emu, Oe) ↔ SI (A/m, T). App supports both with conversion.
- **MPMS bare-CSV format:** MPMS M(T) files are a bare CSV (column header on line 0, no QD `[Header]`/`[Data]`/`BYAPP` blocks). The loader recognizes "no `[Header]` and no `[Data]`" as a bare CSV and reads from line 0 (vs the QD `[Data]`-marker path). Detection has no app token, so the VSM detector uses a strong column fingerprint (`Long Moment (emu)` + `Long Scan Std Dev`). Columns `Field (Oe)`/`Long Moment (emu)` canonicalize to field/moment. The file carries no MOLWGHT/MASS, so molar mass + sample mass must be supplied (`--molar-mass`/`--mass-mg`, GUI, or pipeline options → patched onto the header); without them the analyzer returns `status="gated"`.
- **Multi-field M(T) and which ramp is THE Curie-Weiss fit** (2026-08-30): a multi-field M(T) is several temperature ramps at different held fields run back to back. The rolling-activity labeller gives them one label and the span-dominance guard then rejects the merged block (measured: temperature span 1192 vs field span 790, ratio 1.51 < the required 5.0), so the whole sweep was discarded. `find_blocks` now attempts a **recovery pass** on any block the guard rejects: split it on the plateaus of the confounding (runner-up by span) axis and keep the pieces that pass the same guard, which isolates one held field per block. Recovery only ever examines already-rejected blocks — but note it is NOT behaviour-neutral, because a file whose `find_blocks` was previously EMPTY now gets blocks, which suppresses `segment_sweeps`' whole-frame fallback. Multi-field files therefore need a rule for **which** ramp is the reported CW fit, and neither point count nor field alone is defensible: on a real file the 40 kOe ramp beat the 500 Oe ramp by ONE point and silently moved θ, while the lowest-field (100 Oe) ramps on another real file span only 2–50 K and fit to θ = −216 K at r² = 0.87 against −30 K at r² = 0.99 for the full-range ramps. The rule is therefore **two-stage**: keep only ramps covering ≥50 % of the file's temperature range (a CW fit over a truncated window is not a CW fit), then take the **lowest |field|** among them (CW wants low-field linear response; 10 T is not), size breaking ties. Single-field files have one candidate class and are unchanged.
- **Multi-field M(T): one physical field, one curve** (2026-08-31): a held-field label is decided ACROSS blocks, not per block. `setpoint_key` bins to the nearest integer regardless of magnitude, so on a real file two blocks of the SAME 40 kOe ramp (masked medians 40000.8870 and 39999.5860 — 3 parts in 10⁵ apart) were labelled `40001` and `40000` and drawn as two M(T) curves. `cluster_field_setpoints` (`cryosweep_core/grouping.py`) single-link-clusters the medians actually present — no bin edges to straddle — then labels each cluster with `setpoint_key(cluster median)`: cluster to GROUP, round to LABEL, so legends and CSV cells keep round numbers instead of `499.9 Oe`. `setpoint_key` itself is unchanged (temperature setpoints ride on it). The `t_blocks` ramp split also uses `min_len=5` at this call site (module default 3 untouched): a 3-point "warming" run inside a 116-point cooling ramp at 294.8–300.0 K is a noise turnaround, not a measurement. Measured effect: the gate file goes 7 blocks / 4 labels / 1 fragment → **5 blocks / 3 labels / 0 fragments**; a second real file 23 → 11 blocks; the MPMS oracle file is unchanged. Both defects PRE-DATE the multi-field recovery — they were computed and then discarded by the "no temperature sweep found" early return, so they were invisible rather than absent. **Not fixed by this, and not caused by it:** on that file one 40 kOe branch has χ crossing zero 18 times above 200 K (χ_min = −1.08×10⁻⁵), so 1/χ swings between −1.0×10⁶ and +1.4×10⁶ and the 1/χ panel shows vertical stripes. That is a reciprocal-of-near-zero rendering issue, independent of segmentation, and was initially misattributed to the fragments.
- **1/χ display guard** (2026-08-31): 1/χ is the reciprocal of a measured quantity, so wherever χ passes through zero the inverse diverges and a handful of noise points set the whole axis. Measured on the real multi-field file: one 40 kOe branch has χ crossing zero **18 times above 200 K** (χ_min = −1.08×10⁻⁵, ~10⁻⁶ of the sample's typical χ) and 1/χ swings between −1.0×10⁶ and +1.4×10⁶, drawing vertical stripes across the panel and flattening every real curve onto the axis. The `inverse_chi` and `vsm_chi_t` series therefore blank 1/χ wherever |χ| < 10⁻³ × median(|χ|) **over the whole file** — the file, not the block, because the offending block is *mostly* near-zero χ so its own median is already tiny and a per-block rule never fires. **Display only:** `data['inv_chi']` and the CSV still carry every raw value, and the Curie-Weiss fit is unaffected. The threshold is deliberately loose and is NOT finely tuned — the measured drop count is identical for any rel from 10⁻⁴ to 10⁻² (73 of 731 points on the affected file, **0** on both other real VSM files, so their gallery figures stay byte-identical); a test pins that insensitivity, because if it ever becomes sensitive the noise/signal separation has narrowed and the rule needs rethinking rather than retuning.
- **Curie-Weiss window ladder** (uncertainty honesty, 2026-08-10): the CW fit is repeated on T≥{25, 50, 100, 150, 200} K rungs (≥10 pts each; a rung is skipped unless its cutoff sits >20 K below T_max); `cw_ladder` carries per-rung θ/μ_eff/σ/r²/n, and `theta_spread_k`/`mu_eff_spread` = max−min across rungs **plus the full fit** (the full fit's displacement from the rungs is the signal). Spread > max(3σ_θ of the full fit, 2 K floor) ⇒ `window_sensitive` quality flag, which also halves the reported confidence. **The spread is window sensitivity, not an error bar** — it is never written with `±`. The **reported θ does not move**: the headline, the plot annotation and every exported number stay the full-window fit's θ; the ladder is context, not a re-window. GUI: `θ = −50.3 K (full-window fit — REPORTED) — WINDOW-SENSITIVE: θ(full→200 K) = −50.3→−37.5; T≥25 K rung gives −42.2; spread 12.7 K; σ_stat 0.99 K (fit scatter only, not the uncertainty on θ)`. `fit_params.csv` appends `sigma_kind` (`fit_scatter_stat` vs `window_spread`) and `flags` columns, and names the spread rows `theta_window_spread_not_an_error_bar`. Measured on the real MPMS file: θ_full = −50.267 ± 0.989 while every rung sits at −42.2 … −37.5, spread 12.72 K ≈ 13× the statistical σ, with r² ≥ 0.9965 at every rung — **r² cannot warn you**. A low-T-dominated fit whose rungs drift monotonically also warns that the low-T rows are likely outside the paramagnetic regime.

### AC Susceptibility (ACMS)

- **Format:** `BYAPP,ACMS` files carry in-phase M′ and out-of-phase M″ moments per (frequency, AC-amplitude, DC-field) group; `M'`/`M''`/`Frequency (Hz)`/`Amplitude (Oe)` canonicalize to moment_prime/moment_dprime/frequency/amplitude. A net-new relative-tolerance 1-D grouper segments curves by frequency, amplitude, and field; each ramp is split into warming/cooling direction segments.
- **AC susceptibility**: χ′ = M′/H_ac, χ″ = M″/H_ac (emu/Oe), where H_ac = AC drive amplitude (Oe). χ′ = dispersive (screening / in-phase) component; χ″ = dissipative (loss / out-of-phase) component.
- **Molar susceptibility**: χ_mol = χ · M_mol / (m·10⁻³), with M_mol molar mass (g/mol) and m sample mass (mg). Capability-gated (`molar_normalization`) on both being supplied (`--molar-mass`/`--mass-mg`, GUI, or header); absent → `chi_prime_molar = None`, capability `False`.
- **Superconducting screening** (per curve, capability-gated): a diamagnetic χ′ step from the low-T side. Criteria — (a) drop ≥ 10σ of the high-T plateau noise, (b) diamagnetic low-T level χ′_low < 0, (c) tilt guard: low-T level must lie ≥3σ below the extrapolated high-T baseline trend (OLS line fit over the top-20%-of-T window, extrapolated down to the low-T location) — else it is drift, not a step (replaces the old full-curve-σ rule, which false-declined transitions centered in the T window). Reports `tc_onset_k`/`tc_mid_k` from χ′ crossings (mid = 50% of the drop), corroborated by a χ″ peak inside the transition window; `low_confidence` when the plateau is short/noisy → dotted Tc marker. Null result → `sc_transition = None`.
- **χ″ peak** (per curve, capability-gated): generic OLS baseline + MAD residual prominence — a peak is reported when residual prominence ≥ 5σ (scipy-free). Freezing/blocking temperature T_f = peak position; markers drawn per curve. Featureless → `chi_dprime_peaks = []`.
- **Robustness / null result:** a featureless single-frequency file (e.g. a real single-frequency He-3 file) declines both detectors — `sc_transition = None`, `chi_dprime_peaks = []` — and reads flat/clean; only `ac_susceptibility` is applicable.
- **Recognized-but-deferred** (not yet fitted): frequency-dependent freezing T_f(ω) → spin-glass relaxation — Mydosh parameter, Arrhenius/Vogel–Fulcher fits, critical slowing down, and Cole–Cole (χ′ vs χ″ arc) analysis.

### Thermal Transport (TTO)

- **Format:** `BYAPP,THERMAL_TRANSPORT` files carry four instrument-computed quantities per row — Conductivity (W/K-m), Seebeck Coef. (µV/K), Resistivity (Ohm-m) and Figure of Merit ZT, each with a Std.Dev. column. `Conductivity (W/K-m)`/`Seebeck Coef.`/`Resistivity (Ohm-m)`/`Figure of Merit ZT` canonicalize to kappa/seebeck/rho_tto/zt; `Sample Temp. (K)` was added to the temperature variants for this probe. Detection is a `BYAPP` token plus a raw-name fingerprint (the detector compares RAW lower-cased column names, never `_norm` output).
- **No user inputs** (`needs = ()`): sample geometry (lead separation, cross-section, emissivity) rides in the header INFO block, so κ and ρ are already absolute.
- **κ** = thermal conductivity (W·K⁻¹·m⁻¹), instrument-computed from heater power, ΔT and geometry. **S** = Seebeck coefficient (µV/K). **ZT** = S²T/(ρκ) — the instrument column is reported as measured, not recomputed (verified self-consistent on the real file to 4 digits).
- **Wiedemann-Franz:** κ_e = L₀T/ρ with L₀ = 2.443×10⁻⁸ W·Ω·K⁻² (Sommerfeld value); κ_ph = κ − κ_e. **Lorenz ratio** L/L₀ = κρ/(L₀T): ≈ 1 elastic/electron-dominated, > 1 phonon contribution, < 1 inelastic scattering (κ_ph then negative — reported as computed, never clipped).
- **Power factor** PF = S²/ρ (W·K⁻²·m⁻¹; S converted µV→V).
- **κ_ph power law** (capability `kappa_ph_power_fit`): κ_ph = B·Tⁿ fitted with a free exponent on the **primary ≤10 K window** (highest r², and a Tⁿ phonon power law is a low-T asymptotic concept), reported with **three** numbers, never the statistical σ alone: `n ± σ_n` (statistical), `n_spread` = max−min of n over a 10/15/20/30 K **window ladder**, and `n_loglog` from a log-log OLS on the *same* primary window with `n_method_delta = |n − n_loglog|`. On the gate file the window moves n by **0.71** (2.0266 → 1.3145) and the method by **0.019**, against σ_n = 0.0062 — and r² is ≥ 0.99 at every rung, so r² cannot warn you. `window_sensitive` fires when `n_spread > max(3σ_n, 0.05)`; the **absolute 0.05 floor is load-bearing** (without it the rule collapses to convergence noise on exact synthetic data). **On real files this flag is EXPECTED to fire — its absence is the informative case.** `n_spread` is `None`, never `0.0`, when fewer than two rungs fitted (`ladder_incomplete`). The curve fitted is the one with the most finite κ_ph > 0 points below 10 K (zero-field breaks ties), not the widest zero-field ramp. Gate file: n = 2.0266 ± 0.0062, B = 3.5341e-3, r² = 0.99927, 163 pts. **The fit DECLINES rather than reporting a number** when the exponent pins at a search bound (`n_at_bound`, bounds [0.5, 6.0]), when the window holds a single distinct T (`degenerate_window`), or when r² ≤ 0 (the power law fits worse than a constant) — a bound is not a measurement, and a flat κ_ph below 10 K (normal wherever κ_e dominates) otherwise returns n = 0.5 with `n_spread` ≈ 1e-16, i.e. reads as a *perfectly window-stable* exponent. The capability then carries the specific reason and its numbers, and every κ_ph CSV cell is blank. `kappa_e_dominant` and `window_sensitive` do NOT decline — they describe a real fit and are spelled out verbatim in the GUI row's `flags:` clause and the CSV's `kappa_ph_flags`. The deferred `boundary_scattering_fit` stub is **distinct** — it names a specific n = 3 claim this free-n fit deliberately does not make.
- **Integrity signals (always on, independent of any plot toggle):**
  - **Thermal-gradient warning** — |ΔT|/T > 5 % on any kept row (raw `Delta Temp. (K)` column, absolute value: the ΔT sign is a wiring convention). κ there is averaged over a wide T window. Gate file: 20 of 976 rows, max 11.72 % at 2.025 K. An absent column and an all-empty column behave identically (no warning, no error).
  - **Seebeck low-T sign-oscillation warning** — ≥ 5 sign changes in S between consecutive finite points below 20 K, packed into a window narrower than 5 K. The trigger is **density**, not count. Gate file: 11 changes in 10.186–11.910 K. It makes **no claim about noise**: `seebeck_std` is the instrument's repeat-scatter on one measurement, and every bracketing point on the gate file is 11.4–45.5σ from zero, so these crossings are **real structure**.
  - **`rrr_std`** — σ propagated from `rho_std` through the median-of-5 endpoints: `σ_endpoint = 1.2533·median(σ_point)/√k` (the √(π/2) is the median's efficiency penalty), then `σ_RRR = RRR·√((σ_hi/ρ_hi)² + (σ_lo/ρ_lo)²)`. Gate file: RRR = 1.4555 ± 0.01742. When the ±1σ band straddles a 1.02/0.98 classification threshold, `classification` becomes `"unknown"` **and** a `classification_uncertain` warning is emitted — that warning is the disambiguator from `_classify`'s own "invalid endpoints" `"unknown"`.
  - **`zt_peak_std`** — the `zt_std` at the peak row, tracked in `_zt_peak`'s own loop (ties keep the FIRST maximum, so it is not recoverable afterwards). Gate file: 1.59828e-5 on a 3.92322e-4 peak (4.07 %).
- **Error bands (opt-in):** `PlotSpec.error_band` (default **False**, so every existing render stays byte-identical) draws a ±1σ `fill_between` band — `alpha=0.20`, `zorder=1.5`, `gid="errband"` — on `tto_kappa_t`, `tto_seebeck_t`, `tto_zt_t` and all three `tto_summary_t` panels. `tto_wf_t` and `tto_lorenz_t` carry none: their quantities are derived and uncertainty propagation through them is deferred. On the stacked headline the ρ-panel band is drawn **after** `_rho_axis_autoscale`, at `yscale = 100·factor`, because that helper rescales `ax.lines` only and a `PolyCollection` would silently be 10⁶× too small. Exposed in the GUI as an "Error band" checkbox on those four kinds.
- **`tto_lorenz_t` plot kind** — L/L₀ vs T with a hard reference line at the Sommerfeld value 1.0. Gate file span 1.874–7.786.
- **RRR** = ρ(T_high)/ρ(T_low) on the widest zero-field (|H| < 50 Oe) ramp, endpoints = median of the 5 physical points nearest each T-extreme — the same convention as the resistivity probe, modulo one documented divergence: TTO ρ is instrument-computed and sentinel-free, so the local endpoint mask is `isfinite & > 0` only (no sentinel guard, no MAD exclusion). Classification thresholds 1.02/0.98 → metallic/insulating/non_monotonic/unknown.
- **Rows carrying a non-zero `Error (code)`** are KEPT (their values are physically continuous), counted into `n_error_rows` and surfaced as a warning — never silently dropped. Rows with κ ≤ 0 ARE dropped (unphysical sentinel) with a counted warning.
- **Real-file behaviour** (measured on a real thermal-transport file, not distributed with the source): one 976-point cooling curve at 0.077 Oe, 6 error rows kept; RRR = 1.4556 → `metallic` (note the raw-extrema ratio 1.476 is NOT the reported value — the 5-point median endpoints are); **L/L₀ is non-monotonic — 1.874 at the LOWEST T (2.03 K) → peak 7.786 at 27.8 K → 2.108 at 301.4 K**, i.e. its minimum sits at the bottom of the range, it does not fall monotonically from 7.8, and it **never reaches 1** anywhere on this file (so 1.87 is *not* "near-Sommerfeld" — the elastic/WF regime is simply not attained even at 2 K). The 27.8 K peak does **not** track a κ maximum (κ peaks at 284.6 K, κ_ph at 155.9 K): this is a heavily disordered alloy (RRR = 1.48) whose ρ rises only ×1.48 across the whole range (2.519e-6 → 3.720e-6 Ω·m), so L/L₀ = κρ/(L₀T) ≈ **κ/T** up to that slow drift: κ/T peaks at 22.72 K and the residual ρ(T) rise carries the L/L₀ maximum out to 27.8 K. (κ_ph/T peaks at the same 22.72 K, so quoting the phonon term instead of κ/T would wrongly suggest it is doing the work.) Phonon-dominated throughout; κ_ph never negative here. ZT **rises monotonically to 3.923×10⁻⁴ at the highest measured T (301.37 K) — no interior maximum in range**; PF at T_high 4.645×10⁻⁶ W·K⁻²·m⁻¹.
- **ZT "peak" honesty flag:** `summary.zt_peak`/`zt_peak_t_k` are the max valid ZT across all curves and its T. When that maximum sits at either end of the measured T range (the sweep simply stopped, as on the gate file) `summary.zt_peak_at_edge` is `True`, the CSV carries a `zt_peak_at_edge` column and the GUI row reads "ZT peak (at T range edge)" — a boundary maximum is not an observed peak.
- **Recognized-but-deferred** (reported in `capabilities[]`, not yet fitted): Callaway phonon model, the boundary-scattering **n = 3 hypothesis test** (distinct from the free-n fit above), diffusive S/T analysis, κ(H) field sweeps; also magnon thermal transport, PF/ZT optimization plots, radiation-loss corrections (the instrument already applies its own), uncertainty **propagation** through the derived quantities (κ_e, κ_ph, L/L₀, PF) and therefore bands on `tto_wf_t`/`tto_lorenz_t`, a `power_factor` plot kind, and any cross-probe uncertainty refactor. **Method sensitivity of the κ_ph exponent is NOT deferred** — two methods are reported (`curve_fit` and log-log) and takes neither as authority; what stays deferred is σ-weighted / robust (Huber) variants and any attempt to pick a "best" method.

### Resistivity

- **Resistivity from geometry**: ρ = R × (A / L)
  - R = measured resistance (Ω)
  - A = cross-sectional area = width × thickness (cm²)
  - L = length between contacts (cm)
- **Field dependence**: R(H) at fixed temperatures
- **Temperature dependence**: R(T) at fixed fields
- **3 channels**: Bridge 1, 2, 3 — independent resistance measurements

**Analysis quantities (capability-gated by present data):**
- **AC Transport (ACT) option:** the `BYAPP,ACTRANSPORT` format is detected as the same `resistivity` probe (multi-token detector). Its resistivity lives in `Res. chN (ohm-cm)` (channels ch1/ch2), already in Ω·cm, so the instrument-column path uses ×1 — vs the QD Resistivity option's `Bridge N Resistivity (Ohm-m)` which uses ×100. The conversion is unit-aware (`Ohm-m`→×100, `Ohm-cm`→×1).
- **ρ source**: geometry recompute ρ = R·(w·t)/L (Ohm·cm, R from `Bridge N Resistance (Ohms)`) when sample geometry supplied; else the instrument `Bridge N Resistivity (Ohm-m)` column ×100. Empty bridges are skipped.
- **RRR** (residual resistivity ratio): RRR = ρ(T_high)/ρ(T_low) on the widest zero-field (|H|<50 Oe) ρ(T) ramp; endpoints = median of the 5 physical points nearest each T-extreme. Higher = cleaner sample.
- **Magnetoresistance**: MR% = [ρ(H_max) − ρ(0)] / ρ(0) × 100 per field loop; ρ(0) interpolated at H=0 over the loop's physical points. Flagged low-confidence when ρ(0) < noise floor.
- **Metal vs insulator**: sign of robust dρ/dT — metallic if ρ(high-T) > ρ(low-T), else insulating.
- **Power-law / Fermi-liquid fit**: ρ = ρ₀ + A·Tⁿ on the metallic low-T (≤30 K) ramp (n≈2 Fermi liquid, n≈5 phonon Bloch-Grüneisen limit); residual ρ₀ = intercept. The fit normalizes ρ before `curve_fit` so it is scale-invariant (geometry vs instrument units give the same n/r²).
- **Generic linear fit**: slope/intercept/r² over the low-T window.
- **Resistive superconducting Tc** (capability-gated, per ρ(T) ramp): normal-state ρ_N = median ρ over the top-20%-of-T plateau (finite ρ>0, ≥5 pts); reported only when min ρ over the bottom 10% of T < 2%·ρ_N and ρ_N > noise floor. Criteria = first upward crossings from the low-T side: T_c^onset (90%·ρ_N), T_c^mid (50%, THE Tc), T_c^zero (10%). Low confidence when the drop spans <5 points or plateau CV > 20%. **Narrowness gate** (integrity): declines (reports nothing) when the onset→zero span exceeds 0.5×T_c^mid — a real transition is narrow, whereas a clean non-SC metal's steep Bloch-Grüneisen falloff (high RRR, low Θ_D) coasts below the 2%·ρ_N floor without one; only applied when the plateau is trustworthy (CV ≤ 20%), so a noisy-plateau transition stays low-confidence rather than being wrongly declined.
- **Robustness**: all computations filter to finite ρ>0 first (raw files contain negative/sentinel rows).
- **Power-law cutoff ladder** (uncertainty honesty, 2026-08-10): the ρ = ρ₀ + A·Tⁿ fit is repeated at cutoffs T ≤ {10, 15, 20, 30} K (primary = the shipped ≤30 K fit, byte-identical); `power_law_ladder` carries per-rung n/σ/r²/n_pts and `power_law_n_spread` = max−min. Spread > max(3σ_n, 0.05 absolute floor) ⇒ `window_sensitive` flag — **window sensitivity, not an error bar** (measured: dc-ρ n moves 0.269 ≈ 20× the pcov σ 0.013 between the 10 K and 30 K cutoffs). `window_sensitive` does **not** revoke the `power_law_fit` capability (`n_at_bound` and `rho0_unresolved` still do). GUI row uses the TTO idiom: `n ≈` to 1 dp when sensitive, no `±`, a `flags:` clause, `σ_stat` last and qualified. `.derived.csv` carries `power_law_n_sigma`, `power_law_n_spread` and `power_law_flags` in the same row as `power_law_n`, so the exported number never travels bare.
- **RRR σ** (2026-08-10): σ_RRR = RRR·√((σ_hi/ρ_hi)² + (σ_lo/ρ_lo)²), endpoint σ = 1.2533·median(instrument ρ-std of the k=5 endpoint rows)/√k (1.2533 = √(π/2), the median-vs-mean standard-error factor). It is **instrument-derived**, not a fit σ — it is labeled as instrument noise wherever it is shown (`RRR ± σ (σ_inst — instrument noise; excludes ramp/endpoint choice)`) and it excludes the dominant systematic, which is *which ramp and which endpoints*. Uses the instrument std columns (`rho_std_bridge{N}` Ohm-m / `rho_std_ch{N}` Ohm-cm) on the same widest-ramp rows as RRR itself; scale-invariance of the ratio makes it valid for geometry-ρ too. `None` (never NaN) when std columns are absent; exported as `rrr_std`.
- **Geometry-unset warning** (2026-08-10): header `SampleN Cross Section = 1` / `Length = 1` means the user never set geometry in the PPMS software, so when `rho_source = "instrument_column"` the ρ column is resistance × an arbitrary factor. The analyzer keeps reporting but warns: **ratios (RRR, MR%) are unaffected** (verified — RRR is 18.5227 under none / partial / full geometry); **absolute ρ and the power-law residual ρ₀ are scale-arbitrary**. Remedy: enter width/thickness/length, which switches `rho_source` to `"geometry"`. Partial geometry still falls back to the instrument column by design.
- **Activated transport (Arrhenius fit)** (2026-09-05, capability `activated_transport`): on an insulating zero-field ρ(T) ramp, OLS of ln ρ vs 1/T gives ρ = ρ₀·exp(E_a/k_BT) with **E_a = k_B × slope, reported in meV as measured**. **The factor-of-two trap is the whole point of this fit's design**: for *intrinsic* conduction ρ ∝ exp(+E_g/2k_BT), so the band gap is **E_g = 2·E_a** — but for *extrinsic* conduction the activation is a donor/acceptor level and the factor is 1 (or ½ under compensation), and **transport data alone cannot tell the regimes apart**. People have published half-gaps as gaps and gaps as half-gaps because a script silently multiplied (or didn't). The analyzer therefore never converts silently: the only gap field anywhere — JSON, CSV, figure, GUI — is named `e_g_assuming_intrinsic_mev`, so the assumption travels with the number. The figure annotation spells it out: `E_g = 2·E_a (only if intrinsic)`.
  - **Window ladder** (same honesty idiom as the power-law/κ_ph ladders): the fit repeats on T ≥ quantile-{0, 25, 50, 75}% windows (activation is a high-T statement; rungs shrink from the low-T side, where freeze-out and hopping curvature live); `arrhenius_ea_spread_mev` = max−min over unflagged rungs; spread > max(3σ, 1 meV floor) ⇒ `window_sensitive`. **On Mott-VRH data the Arrhenius fit still returns a slope on every window — the drift across windows is how it self-identifies as the wrong model** (measured on the VRH control, T₀ = 10⁶ K: E_a runs 24.0 → 45.1 meV across rungs — spread 16.0 meV over unflagged rungs against per-rung σ ≈ 0.2–0.4 meV — while Arrhenius r² is 0.968 on the full window, so **r² cannot warn you; only the window drift does**). The floor is deliberately loose, not tuned: verdicts are identical for any floor 0.5–2 meV (pinned by test, the 1/χ-guard convention).
  - **Declines rather than fabricates** (`ARRHENIUS_DECLINE_FLAGS`): `insufficient_rho_span` — under **one e-fold** of ρ change in the window, an exponential deviates from its chord by < ~12%, indistinguishable from a straight line at instrument scatter. Measured justification: the corpus' one real insulating channel changes ×1.3 over 3–340 K (0.28 e-folds), fits at r² = 0.10, and its "E_a" runs 0.054 → 7.96 meV purely with the window — a bad metal with dρ/dT < 0, not activated conduction; the honest result there is a decline. `ea_unresolved` — σ ≥ |E_a|. Declined ⇒ blank CSV cells, no fit line, no gap, reason in `arrhenius_flags`; `window_sensitive` annotates but never declines.
  - **Model discrimination is not claimed**: `arrhenius_alt_models` reports r² of ln ρ against 1/T (Arrhenius), T^(−1/4) (Mott 3-D VRH) and T^(−1/2) (Efros–Shklovskii) on the same window, *with* the note that r² over one temperature window cannot select a conduction mechanism — discrimination needs a wide range or an independent probe (Hall, thermopower).
  - Plot kind `resistivity_arrhenius`: log ρ vs 1000/T (industry convention), fit line over the fitted window only, E_a ± σ + the intrinsic-labelled gap on the figure; a declined fit draws no line and states why.
- **Recognized-but-deferred** (reported in `capabilities[]`, not yet fitted): Mott VRH ρ∝exp[(T₀/T)^(1/4)] as a *fitted* model (it is currently only an r² comparison in `arrhenius_alt_models`), Kondo ρ=ρ₀−c·ln T, full Bloch-Grüneisen, Kohler's rule.

### Heat Capacity

**Main model — Debye-Einstein (7 parameters):**
- Cp(T) = γT + n·C_Debye(T, θ_D) + m₁·C_Einstein(T, θ_E1) + m₂·C_Einstein(T, θ_E2)
  - γ = electronic coefficient (mJ/mol·K²)
  - n = number of atoms per formula unit
  - θ_D = Debye temperature
  - θ_E1, θ_E2 = Einstein temperatures
  - m₁, m₂ = Einstein mode coefficients

**Low-temperature models (Cp/T vs T² representation):**
1. Electronic + Debye T³: Cp/T = γ + βT²
2. Electronic + Debye T³+T⁵: Cp/T = γ + βT² + δT⁴
3. Spin fluctuation (non-interacting): Cp/T = γ + βT² + AT²·ln(T₀/T)
4. Spin fluctuation (weakly interacting): Cp/T = γ + βT² + AT²·(1 + T²/T₀²)

**Debye temperature from β**: θ_D = (12π⁴nR / 5β)^(1/3)

**Magnetic-entropy Rln match verdict** (uncertainty honesty, 2026-08-10): `suggest_rln` picks the nearest R·ln(2J+1) plateau to the magnetic-entropy saturation, but the suggestion carries a **verdict**, not just a label — `matched` iff |S_sat − Rln(2J+1)|/Rln(2J+1) ≤ 25 % (`rel_err`, `distance`, `tol` all reported). A negative saturation is unphysical and is always unmatched. The nearest ladder value is a nearest-neighbour, **not evidence of a doublet**: an always-on warning says so whenever a suggestion is present and unmatched. GUI reads `R ln3 (matched, 2% off)` vs `R ln2 (NOT matched — S_mag saturation is 113% away)`; the entropy CSV appends `rln_label`/`rln_matched`/`rln_rel_err`. Measured on the four real HC files: `matched = False` on all of them (`rel_err` 1.1265 / 0.7191 / 1.4878 / n/a).

### Hall Effect

**3-stage pipeline:**
- **Stage A** (temp-dependent): R_H from linear fit of R_xy vs H at each temperature
- **Stage B** (field-dependent): Zero-field subtraction → antisymmetrization R_xy(H) = [R(+H) − R(−H)] / 2 → linear fit
- **Stage C** (derived quantities):
  - Hall coefficient: R_H = slope × thickness
  - Carrier concentration: n = 1 / (e · |R_H|)
  - Mobility: μ = |R_H| · σ (requires longitudinal resistivity data)

**As-built (separately-invoked, capability-driven):**
- Hall uses the SAME PPMS Resistivity-option file as plain resistivity (only wiring differs) → detected as `resistivity`; the Hall analyzer is invoked explicitly via `cryosweep hall <file> --hall-channel N --thickness T [--thickness-unit mm|um|nm] [--long-channel M | --long-file F --long-channel M] [--geometry-sign ±1]` or `cryosweep analyze <file> --probe hall …`.
- Transverse signal read from `Bridge N Resistance (Ohms)` (R_xy); field Oe→T = /10000; **R_H = slope(R_xy vs B) × thickness × geometry_sign**.
- **Stage B antisymmetrization is essential** — on real data the Hall (odd-in-H) signal can be ~1% of the even R_xx admixture; `R_asym(H)=[R(+H)−R(−H)]/2` (interpolated, tolerant of non-symmetric sweeps) removes it. Stage A (raw fit) reported for transparency; Stage B is trusted.
- Stage C: n = 1/(e·|R_H|), e = 1.602176634e-19 C, carrier sign = sign(R_H); μ = |R_H|/ρ_xx where ρ_xx(T) comes from a longitudinal channel (same file) or a separate longitudinal file (interpolated over T). Mobility is capability-gated on longitudinal data.
- Units SI: R_H m³/C, n 1/m³, μ m²/(V·s).
- **Excitation current and current density** (2026-09-05): the temp-dep Hall points report the
  instrument's **excitation current I** per temperature (`Bridge N Excitation (uA)`, local
  median over the measured rows near each grid T — never interpolated). I answers the question
  the file can answer alone: was the drive constant across the sweep, and low enough not to
  heat the sample. **Caveat: I is what the instrument reports, which need not equal the
  requested drive.** **Current density J = I/(w·t)** (A/m²; numerically µA/mm²) is a
  capability that activates only when sample width (`--width-mm`, the shared SampleGeometry
  route) AND thickness are both supplied — an ungated J on unset geometry would be
  scale-arbitrary, the exact failure the resistivity geometry-unset warning names. A
  constant-drive file (the shipped real Hall example: 7999.997 µA throughout) draws flat
  I and J lines; that is the correct result.

### Anomalous Hall effect (recognized, deferred)

In a magnetic conductor the transverse resistivity has two parts:

**ρ_xy = R₀·B + R_s·μ₀·M(B, T)**

- **R₀** — the ordinary Hall coefficient: the Lorentz-force term, linear in the applied
  field B, carrying the carrier density and sign exactly as in the pipeline above.
- **R_s·μ₀·M** — the anomalous term: proportional to the sample's own magnetization, not
  the field. It grows with M, then **saturates where M saturates**; R_s is typically much
  larger than R₀.

**Separating the two in practice.** Where M is saturated (high field, low enough T), the
anomalous term is a constant offset: a linear fit of the antisymmetrized ρ_xy over the
saturated region gives **R₀ from its slope** and **R_s·μ₀·M_sat from its zero-field
intercept**. Substituting a measured M(B, T) instead lets both terms be fitted over the
full field range. Either way the decomposition leans on M — the first to *demonstrate*
saturation, the second explicitly.

**What partial extraction without M(H) would and would not license.** Reporting the
high-field slope as R₀ is defensible *only when M-saturation is demonstrated there* — and
demonstrating it needs the magnetization data, so without M this claim rests on an
unverifiable assumption. Publishing the zero-field intercept of a high-field fit as "the
anomalous Hall resistivity" without M is **not** defensible: the intercept lumps a real
anomalous term together with ordinary multi-band curvature and magnetoresistance leakage,
and cannot tell them apart. This is why `anomalous_hall` is reported in `capabilities[]`
as `applicable: false` with the missing measurement as the reason, rather than shipped as
a fit that cannot be validated.

**The scaling relation, and why it matters.** R_s is not a constant of the material the
way R₀ is: empirically R_s ∝ ρ_xx (skew scattering) or R_s ∝ ρ_xx² (side-jump and the
intrinsic Berry-phase mechanism). Measuring which power law R_s follows against the
sample's own ρ_xx(T) is how the microscopic mechanism is identified — so a serious AHE
analysis needs ρ_xx(T) alongside ρ_xy and M, and a fitting-model choice (constant R_s vs
the two scalings) that the user must own, because the three conventions give different
numbers.

**The measurement prerequisite** — the part most treatments leave implicit: AHE
decomposition needs **the same sample measured in both configurations** — ρ_xy(B, T) in a
Hall-wired transport setup AND M(B, T) in a magnetometer (VSM/MPMS), on the same field
axis of the same orientation, at matching temperatures, into clear M-saturation. No such
pair exists in this corpus: no magnetization file accompanies any Hall-wired sample.

**Indication from the shipped example's source** (method stated; an indication, not a
settled result): antisymmetrizing the Hall channel of the real measurement behind
`examples/hall_mixed_sweeps.dat` on an interpolated ±B grid and comparing the OLS slope
over |B| ≤ 20 kOe with the slope over |B| ≥ 50 kOe gives low/high ratios of **0.76 at
2 K and 0.74 at 50 K** (an independent re-derivation with different windows gave
0.79/0.93 — the numbers move with window choice, the direction does not). A saturating
anomalous term inflates the *low*-field slope, so its signature is a ratio **above 1**;
this sample stiffens at high field instead. Two temperatures, one sample, raw Bridge-1
signal — an argument that the deferral loses nothing *here*, not a claim about the
material.

### Fit vs instrument uncertainty

Two σ families ride on Hall quantities, and they answer different questions — they are never
interchangeable and are never reported under a shared name:

- **`*_sigma` (residual σ, "fit scatter")** — the OLS standard error of the fitted slope,
  computed from the residuals of the R_xy-vs-B fit. It measures **fit quality**: how well a
  straight line describes the antisymmetrized points. It exists only with ≥ 3 antisym points
  (2 points have zero residual degrees of freedom → `r_h_sigma = None`, `sigma_zero_dof =
  True`; an honest None, never 0.0).
- **`*_sigma_instrument` (σ_inst, "instrument noise")** — the instrument's per-point repeat
  noise (the file's std-dev columns) propagated exactly through the same estimator. It is a
  **weaker, different claim**: what the slope uncertainty would be if the only noise were the
  instrument's own scatter, saying nothing about whether the line is the right model. In the
  GUI it is always labeled "σ_inst (instrument noise, not fit quality)".

Propagation (through-the-estimator, exact linear):

- **OLS with intercept** (antisym fit, points (B_i, R_asym,i) with instrument σ_asym,i):
  w_i = (B_i − B̄) / Σ_j (B_j − B̄)², var(slope) = Σ_i w_i² σ²_asym,i, σ_inst = √var.
  Antisymmetrization halves-and-sums pair noise: σ²_asym = (σ²_+H + σ²_−H)/4.
- **Through-origin 2-point fallback** (zero-field-subtracted, y_i = R(B_i) − R(0)):
  every y_i shares the **same** R(0), so the y_i are correlated and the shared term does not
  sum independently —
  var(slope) = [ Σ_i B_i² σ²_B,i + (Σ_i B_i)² σ²_B=0 ] / (Σ_i B_i²)², σ_inst = √var.
  Residual σ stays None here — zero residual DOF by construction.
  (On a symmetric ±B pair Σ B_i = 0, so the zero-field noise cancels out of the slope
  entirely — as it must, since a common offset shifts every y_i equally and a
  through-origin slope is blind to it.)
- Both slope σ families convert to R_H by × thickness (thickness and geometry sign carry no
  σ), and to carrier-n by pure relative propagation: σ_n/n = σ_RH/|R_H|.
- The instrument columns arrive in resistivity units; the per-row `Resistance/Resistivity`
  ratio of the file itself bridges them to Ω — internally self-consistent whatever the
  header geometry setting was (it is not a claim about absolute resistivity).
- A ≥ 50 % relative σ on R_H produces an explicit noise warning rather than a silent number.

The same discipline holds outside Hall: window-sensitivity spreads (Curie-Weiss θ ladder,
resistivity power-law n ladder, TTO κ_ph ladder) are **not error bars** and are never written
with `±`; the statistical σ is labeled `σ_stat (fit scatter only)` and listed last.

### Dilatometry (not yet implemented)

**Thermal expansion:**
- Linear coefficient: α_L = (1/L)(dL/dT) (K⁻¹)
- Volumetric: β = (1/V)(dV/dT) ≈ 3α_L (isotropic)
- Grüneisen parameter: γ_G = βV/(C_V κ_T) = βB_S/(ρC_P), typically ~1–3

**Magnetostriction:**
- Spontaneous: λ_s = ΔL/L at magnetic ordering
- Field-induced: λ_∥ (parallel), λ_⊥ (perpendicular), volume ω = λ_∥ + 2λ_⊥
- Saturation: λ_s = (2/3)(λ_∥ − λ_⊥)

**Phase transitions:**
- Ehrenfest (2nd order): dT_c/dP = V·T_c·Δβ/ΔC_P
- Clausius-Clapeyron (1st order): dT/dP = T·ΔV/ΔH

**Elastic:**
- Debye temperature from sound: θ_D = (ℏ/k_B)(6π²n/V)^(1/3) · v_m


