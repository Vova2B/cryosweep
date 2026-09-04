from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class SampleGeometry(BaseModel):
    width_mm: float | None = None
    thickness_mm: float | None = None
    length_mm: float | None = None

    def complete(self) -> bool:
        vals = (self.width_mm, self.thickness_mm, self.length_mm)
        return all(v is not None and v > 0 for v in vals)


class HallCfg(BaseModel):
    hall_channel: int | None = None          # bridge carrying the transverse (Hall) signal
    longitudinal_channel: int | None = None  # bridge for R_xx in the SAME file (mobility)
    longitudinal_file: str | None = None     # OR a separate file path for R_xx (mobility)
    thickness_mm: float | None = None        # sample thickness -> R_H magnitude
    geometry_sign: int = 1                    # +1 / -1 wiring/geometry sign
    temp_interval: float = 1.0               # common-T-grid spacing (K) for fixed-field interpolation
    # < this many distinct antisym ± pairs -> point low_confidence. Default 1 (2026-09-02,
    # KNOWN-ISSUES #18): a single symmetric ± pair IS a complete antisymmetrization (the
    # even-in-B admixture cancels exactly), so it is a trusted R_H, not a degraded one.
    # The old default 3 was calibrated when < 2 pairs meant NO antisym fit existed at all;
    # after the single-pair fit landed it would re-flag every single-pair point and pin
    # confidence at 0.0 on sparse-field files. Raise to 3 to demand enough pairs for an
    # in-point linearity check (residual sigma). Genuinely unpaired points (the "2point"
    # zero-subtracted fallback) are always low_confidence regardless of this knob.
    tdep_min_antisym_points: int = 1
    tdep_two_point_fallback: bool = True     # zero-field-subtracted 2-point R_H(T) fallback (Sub-feature B)


class ResistivityCfg(BaseModel):
    exclude_hall_channel: bool = True         # route clear odd-in-B winner out of resistivity
    hall_channel_override: int | None = None  # force-exclude this channel (user override)


class StabilityCfg(BaseModel):
    schema_version: int = 1
    window: int = 16
    std_max: dict[str, float] = Field(default_factory=dict)
    drift_max: dict[str, float] = Field(default_factory=lambda: {"temperature": 0.25, "field": 50.0})
    cluster_rel_tol: float = 0.05
    trend_slope_max: dict[str, float] = Field(default_factory=dict)
    monotone_fraction_min: float = 0.8
    span_drift_ratio_min: float = 5.0
    min_segment_len: int = 8
    activity_min: float = 1.0

class QualityCfg(BaseModel):
    exclude_outliers: bool = False     # opt-in: drop robust outliers from curves+fits+CSV
    outlier_k: float = 8.0             # median ± k·MAD half-width
    # DQ-B setpoint grouping / separation-sanity (defaults grounded on the real Hall/Resistivity
    # measurement files, not on synthetic data)
    setpoint_threshold_k: float = 10.0  # below: bin to nearest half-int; at/above: nearest int
    setpoint_unstable_k: float = 0.5    # flag a group whose raw per-segment setpoints span > this
    setpoint_near_dup_k: float = 0.5    # flag two adjacent groups whose raw intervals are closer

class HeatCapacityCfg(BaseModel):
    # NOTE: low-T parsimony threshold stays on RunConfig.hc_parsimony_r2 (canonical); do not re-add here.
    full_init: dict[str, float] = Field(default_factory=lambda: {
        "theta_D": 100.0, "n": 7.0, "gamma": 0.007,
        "theta_E1": 50.0, "theta_E2": 150.0, "m1": 1.0, "m2": 2.0})
    full_fixed: dict[str, bool] = Field(default_factory=lambda: {
        "theta_D": False, "n": True, "gamma": False,
        "theta_E1": False, "theta_E2": False, "m1": False, "m2": False})
    full_fit_min_k: float | None = None
    full_fit_max_k: float | None = None
    lowt_fit_min_k: float | None = None
    lowt_fit_max_k: float | None = None
    full_max_t_min_k: float = 50.0    # require T_max >= this for the full-range fit to be available
    full_min_points: int = 15         # points floor for the 7-param fit
    full_min_r2: float = 0.9          # full-range fit below this r² is treated as non-converged (not presented)
    # --- slice 2 multi-field engine (all inert defaults) ---
    field_bin_koe: float = 1.0          # |field| bin width for grouping (matches frozen primary path)
    min_lowt_per_field: int = 5         # a field group needs >= this many low-T points to fit
    identifiability_rel_sigma: float = 1.0   # param non-identifiable if sigma >= this * |value|
    bound_rail_frac: float = 0.01       # param "railed" if within this fraction of a finite fit bound
    corr_warn: float = 0.99             # fit degenerate if max |param correlation| >= this
    # --- slice 3: Schottky anomaly (all off/neutral by default) ---
    schottky_enabled: bool = False           # master opt-in; off => no Schottky code runs at all
    schottky_r: float = 1.0                  # g0/g1 degeneracy ratio (fixed, not fitted)
    schottky_lattice_t5: bool = False        # add delta*T^5 to the background
    schottky_include_nuclear: bool = False   # request alphaN/T^2 (still sub-gated + AICc-adopted)
    schottky_nuclear_max_tmin_k: float = 2.5 # M2 attempted only when group T_min <= this
    schottky_fit_max_k: float = 15.0         # fit window upper bound (lattice-validity vs peak coverage)
    schottky_delta_max_k: float = 100.0      # Delta upper bound (bound-rail => undetermined)
    schottky_f_max: float = 5.0              # f upper bound
    schottky_peak_corr_max: float = 0.95     # |corr(f, Delta)| gate
    schottky_delta_h_model: str = "none"     # "none" | "zeeman" | "zfs" (plot-time overlay)
    schottky_aicc_margin: float = 2.0        # non-background model must beat background AICc by >= this
    # --- slice 4: transition search (all off/neutral by default) ---
    transitions_enabled: bool = False            # master opt-in; off => no transition code runs
    transition_form: str = "lambda"              # "lambda" | "jump"
    transition_universality: str = "mean_field"  # "mean_field"(α=0,log) | "ising3d"(0.110) | "xy3d"(−0.013)
    transition_lattice_t5: bool = False          # accepted-but-inert (local wing background; advisory emitted)
    transition_wing_mask_k: float = 2.0          # minimum ± inner half-width (K) masked around the candidate
    transition_aicc_margin: float = 2.0          # form must beat background-only AICc by ≥ this
    transition_compare_forms: bool = False       # opt-in λ-vs-jump comparison
    transition_indistinguishable_band: float = 2.0  # |ΔAICc| below this ⇒ "indistinguishable"
    # --- slice 4 hardening: local wing-poly background + hard gates ---
    transition_wing_frac: float = 0.03           # inner half-width W = max(wing_mask_k, wing_frac·Tc_seed)
    transition_span_mult: float = 5.0            # local window = |T − Tc_seed| ≤ span_mult·W
    transition_wing_order: int = 3               # wing polynomial order (2 cannot follow local cubic lattice curvature)
    transition_prominence_n: float = 4.0         # anomaly must stand ≥ n × wing scatter (1.4826·MAD)
    transition_collapse_margin: float = 2.0      # residual ΔAICc ceiling after near-Tc point removal (λ only)
    transition_amp_max_frac: float = 1.0         # amplitude bound = frac × ptp(local Cp)
    # --- slice PQ-5: entropy S(T) ---
    entropy_extrapolate: bool = True             # integrate the low-T model tail (0..T_min) into S(T)
    entropy_lattice_ref_file: str | None = None  # reference .dat whose Cp(T) defines the lattice (Task 3b)
    entropy_rln_j: float | None = None           # override Rln(2J+1) plateau J (None/0 -> auto-suggest)


class RunConfig(BaseModel):
    schema_version: int = 1
    unit_system: Literal["CGS", "SI"] = "CGS"
    stability: StabilityCfg = Field(default_factory=StabilityCfg)
    geometry: SampleGeometry = Field(default_factory=SampleGeometry)
    hall: HallCfg = Field(default_factory=HallCfg)
    resistivity: ResistivityCfg = Field(default_factory=ResistivityCfg)
    heatcapacity: HeatCapacityCfg = Field(default_factory=HeatCapacityCfg)
    quality: QualityCfg = Field(default_factory=QualityCfg)
    confidence_min: float = 0.5
    hc_parsimony_r2: float = 0.99    # heat-capacity low-T parsimony threshold (simplest model with R2>=this)
    probe_override: str | None = None
    presets: dict[str, dict] = Field(default_factory=dict)

    @classmethod
    def load(cls, path=None, **overrides):
        import json, pathlib
        base = {}
        if path:
            base = json.loads(pathlib.Path(path).read_text())
        base.update(overrides)
        return cls(**base)
