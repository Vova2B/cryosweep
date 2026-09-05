from __future__ import annotations
import hashlib, pathlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from cryosweep_core.io.columns import canonicalize_columns, UNIT_OHM_M, UNIT_OHM_CM, UNIT_MICROOHM_CM
from cryosweep_core.detect.sweeps import segment_sweeps
from cryosweep_core.detect.hall_channel import detect_hall_channel
from cryosweep_core.analyzers.resistive_tc import detect_resistive_tc
from cryosweep_core.fitting.transport import (LinearFitModel, PowerLawRhoModel,
                                         RhoT2FermiLiquidModel, fit_rho_powerlaw_ladder)
from cryosweep_core.fitting.uncertainty import rrr_sigma

_HONESTY_FLAGS = {"window_sensitive", "ladder_incomplete"}   # closed O7: annotations
                                       # that do not revoke capability
from cryosweep_core.result import Result, FitResult, Provenance, Diagnostic
from cryosweep_core.robust import outlier_stats, outlier_mask, is_log_space
from cryosweep_core.registry import Need
from cryosweep_core.grouping import group_segments_by_setpoint
from cryosweep_core.fitting.transport import (NO_FIT_LINE_FLAGS,
                                              _RHO_LADDER_RUNGS,
                                              fit_arrhenius_ladder,
                                              ARRHENIUS_DECLINE_FLAGS)

_ZERO_FIELD_OE = 50.0      # |H| below this counts as held "zero field"
_RRR_K = 5                 # nearest-extreme physical points to median for RRR endpoints
_LOWT_MAX_K = 30.0         # low-T window for power-law / linear fit
_MR_NOISE_FLOOR = 1e-9     # Ohm*cm; rho0 below this -> MR low_confidence
_OUTLIER_WARN_FRACTION = 0.005   # outlier fraction at/above which a diagnostic is severity "warning"
_EXCLUDE_MIN_N = 8         # below this, MAD is too unstable -> diagnostic-only, never exclude


# ---- typed result models ---------------------------------------------------
class RhoTCurve(BaseModel):
    model_config = ConfigDict(extra="ignore")
    held_field_oe: float | None = None
    direction: int = 0
    n_points: int = 0
    classification: str = "unknown"     # metallic | insulating | non_monotonic | unknown
    temperature: list[float] = []
    rho: list[float] = []               # Ohm*cm
    # resistive superconducting transition (PQ-4); absent => not detected
    tc_onset_k: float | None = None
    tc_mid_k: float | None = None
    tc_zero_k: float | None = None
    tc_rho_normal: float | None = None
    tc_low_confidence: bool | None = None

class RhoHCurve(BaseModel):
    model_config = ConfigDict(extra="ignore")
    held_temp_k: float | None = None
    direction: int = 0
    n_points: int = 0
    rho_zero_field: float | None = None
    mr_percent_at_max_field: float | None = None
    max_abs_field_oe: float | None = None
    low_confidence: bool = False
    field: list[float] = []
    rho: list[float] = []               # Ohm*cm
    # constituent-ramp sweep directions before DQ-B display grouping (PQ-4 arrow craft)
    directions: list[int] = []

class BridgeResult(BaseModel):
    model_config = ConfigDict(extra="ignore")
    channel: int
    rho_source: str                     # "geometry" | "instrument_column"
    rho_unit: str = "Ohm*cm"
    classification: str = "unknown"
    rrr: float | None = None
    rrr_t_high: float | None = None
    rrr_t_low: float | None = None
    residual_rho: float | None = None   # rho0 from the power-law fit (Ohm*cm)
    rho_t_curves: list[RhoTCurve] = []
    rho_h_curves: list[RhoHCurve] = []
    power_law: FitResult | None = None
    low_t_linear: FitResult | None = None
    rho_t2_linear: FitResult | None = None   # forced-n=2 Fermi-liquid (rho=rho0+beta*T^2), metallic ramp only
    # --- 2026-08-10 uncertainty-honesty additive field (declared LAST: append-only JSON key
    # order). sigma_RRR propagated from the instrument's per-row Std. Dev. column via the
    # shared fitting/uncertainty helpers, on the SAME ramp rows and endpoint policy the
    # shipped RRR uses. None when the std column is absent (e.g. bare-TAB dc-rho files),
    # non-finite in the endpoint windows, or rrr is None. Reporting-only (spec §4). ---
    rrr_std: float | None = None
    # Power-law cutoff ladder (spec §3, closed O7): the primary fit (power_law above) stays
    # byte-identical; the ladder + spread are additive honesty surfaces. window_sensitive on
    # the primary is an annotation, NOT a fit failure (O7). ---
    power_law_ladder: list[dict] | None = None
    power_law_n_spread: float | None = None
    # --- 2026-09-05 activated transport (appended last; additive-only contract). E_a is
    # reported AS MEASURED; the only gap field is params["e_g_assuming_intrinsic_mev"] -
    # the intrinsic assumption travels in the name (extrinsic factor is 1, and transport
    # alone cannot tell the regimes apart). ---
    arrhenius: FitResult | None = None
    arrhenius_ladder: list[dict] | None = None
    arrhenius_ea_spread_mev: float | None = None
    arrhenius_alt_models: dict | None = None

class Capability(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    applicable: bool
    reason: str = ""

class ResistivityData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    probe: str = "resistivity"
    rho_source: str = ""                # "geometry" | "instrument_column" | "mixed" | ""
    bridges: list[BridgeResult] = []
    capabilities: list[Capability] = []
    excluded_hall_channel: int | None = None   # bridge routed out as Hall-wired (None = none)
    excluded_hall_source: str = ""             # "detected" | "override" | ""


# ---- pure helpers ----------------------------------------------------------
def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""

def _physical_mask(rho) -> np.ndarray:
    rho = np.asarray(rho, float)
    return np.isfinite(rho) & (rho > 0)

def _clean_mask(rho, cfg=None, T=None) -> np.ndarray:
    """Physical mask plus rejection of absurd high-side sentinel rows (rho >> curve median).
    When cfg.quality.exclude_outliers is set, ALSO drop robust median±k·MAD outliers so flagged
    points leave curves, MR endpoints, and fits together. Exclusion is skipped on sparse curves
    (n < _EXCLUDE_MIN_N) where MAD is unstable. cfg=None -> byte-identical to the pre-DQ-A path.

    SC exemption (owner-approved 2026-09-01): when `T` is given (rho(T) call sites) and the
    pre-exclusion curve carries a detected resistive transition, points below its tc_onset_k
    are exempt from exclusion — the superconducting state is a phase, not bad data. An accepted
    transition NEEDS a near-flat normal state, so its floor always sits far outside the robust
    band and would otherwise be silently dropped from curves, fits and the CSV. The robust
    center/scale are then computed over the normal state (T >= onset) only, so genuine
    normal-state spikes are still caught. No detected transition (or no T) -> byte-identical."""
    rho = np.asarray(rho, float)
    m = _physical_mask(rho)
    if not m.any():
        return m
    med = float(np.median(rho[m]))
    if med > 0:
        m = m & (rho < med * 1e4)
    q = getattr(cfg, "quality", None)
    if q is not None and q.exclude_outliers and int(m.sum()) >= _EXCLUDE_MIN_N:
        vals = rho
        if T is not None:
            Tarr = np.asarray(T, float)
            tc = detect_resistive_tc(Tarr[m], rho[m])
            onset = (tc or {}).get("tc_onset_k")
            if onset is not None:
                vals = np.where(Tarr < onset, np.nan, rho)   # exempt + keep stats normal-state-only
        mm = m & np.isfinite(vals)
        if int(mm.sum()) >= _EXCLUDE_MIN_N:
            log_space = is_log_space(vals[mm])
            om = outlier_mask(vals, k=q.outlier_k, log_space=log_space)
            m = m & ~om
    return m

def _endpoint(T, rho, lowest: bool, k: int, cfg=None):
    """Median rho over the k physical points nearest the min (lowest) / max T."""
    m = _clean_mask(rho, cfg, T=T) & np.isfinite(T)
    T, rho = np.asarray(T, float)[m], np.asarray(rho, float)[m]
    if T.size == 0:
        return float("nan"), float("nan")
    kk = min(k, T.size)
    order = np.argsort(T)
    sel = order[:kk] if lowest else order[-kk:]
    return float(np.median(T[sel])), float(np.median(rho[sel]))

def _rrr(T, rho, k: int = _RRR_K, cfg=None):
    """RRR = rho(T_high)/rho(T_low). Returns (rrr, t_high, t_low) or (None,None,None)."""
    t_lo, r_lo = _endpoint(T, rho, True, k, cfg)
    t_hi, r_hi = _endpoint(T, rho, False, k, cfg)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo <= 0:
        return None, None, None
    return float(r_hi / r_lo), float(t_hi), float(t_lo)

def _classify(T, rho, cfg=None) -> str:
    """metallic if rho rises with T (robust endpoints), insulating if it falls."""
    t_lo, r_lo = _endpoint(T, rho, True, _RRR_K, cfg)
    t_hi, r_hi = _endpoint(T, rho, False, _RRR_K, cfg)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo <= 0:
        return "unknown"
    ratio = r_hi / r_lo
    if ratio > 1.02:
        return "metallic"
    if ratio < 0.98:
        return "insulating"
    return "non_monotonic"

def _mr_percent(H, rho, cfg=None):
    """MR% = [rho(Hmax)-rho(0)]/rho(0)*100, rho(0) interpolated at H=0.
    Returns (rho0, mr_percent, max_abs_field, low_confidence)."""
    H = np.asarray(H, float); rho = np.asarray(rho, float)
    m = _clean_mask(rho, cfg) & np.isfinite(H)
    H, rho = H[m], rho[m]
    if H.size < 3 or H.min() > 0 or H.max() < 0:
        return None, None, None, True               # cannot bracket H=0
    order = np.argsort(H)
    Hs, rs = H[order], rho[order]
    rho0 = float(np.interp(0.0, Hs, rs))
    j = int(np.argmax(np.abs(Hs)))
    hmax, rmax = float(Hs[j]), float(rs[j])
    if rho0 <= 0:
        return None, None, hmax, True
    mr = (rmax - rho0) / rho0 * 100.0
    low = rho0 < _MR_NOISE_FLOOR
    return rho0, float(mr), float(abs(hmax)), bool(low)


def _outlier_diagnostics(bridges, k: float) -> list:
    """Emits one Diagnostic per ρ(T)/ρ(H) curve with >=1 outlier; severity 'warning' when
    the outlier fraction >= _OUTLIER_WARN_FRACTION, else 'info'. Non-destructive — reads
    curves only."""
    diags = []
    for b in bridges:
        for kind_label, curves in (("ρ(T)", b.rho_t_curves), ("ρ(H)", b.rho_h_curves)):
            for c in curves:
                vals = c.rho
                onset = getattr(c, "tc_onset_k", None)
                if onset is not None:
                    # SC exemption (owner-approved): the superconducting state is a phase, not
                    # bad data — flag only the normal state (T >= onset), whose robust center/
                    # scale are then also computed without the below-Tc floor.
                    tarr = np.asarray(c.temperature, float)
                    vals = [r for t, r in zip(tarr, c.rho) if t >= onset]
                st = outlier_stats(vals, k=k)
                if st["n_outliers"] <= 0:
                    continue
                if kind_label == "ρ(T)":
                    setp = f"{c.held_field_oe:.1f} Oe" if c.held_field_oe is not None else "?"
                else:
                    setp = f"{c.held_temp_k:.1f} K" if c.held_temp_k is not None else "?"
                sev = "warning" if st["fraction"] >= _OUTLIER_WARN_FRACTION else "info"
                diags.append(Diagnostic(
                    kind="outliers", severity=sev,
                    scope=f"bridge{b.channel} {kind_label} {setp}",
                    message=f"{st['n_outliers']} outlier point(s) "
                            f"({st['fraction'] * 100:.1f}%); max/median {st['max_over_median']:.0f}×",
                    data=st))
    return diags


def _duplicate_setpoint_diagnostics(fsegs, cfg) -> list:
    """Always-on separation-sanity check over the field setpoints (bridge-independent).
    'unstable hold' — a single setpoint group's RAW per-segment temperatures span more than
    cfg.quality.setpoint_unstable_k (drift, or two setpoints wrongly merged). 'near-duplicate'
    — two adjacent groups whose RAW temperature intervals are closer than
    cfg.quality.setpoint_near_dup_k (rounding split one hold across a bin boundary). Comparing
    raw intervals — not the quantized keys, which are always >= half the bin width apart.
    Non-destructive: reads grouping only."""
    q = cfg.quality
    groups = group_segments_by_setpoint(fsegs, "temperature", q.setpoint_threshold_k)
    diags = []
    reps = []  # (key, raw_min, raw_max) for groups with at least one finite raw setpoint
    for key, segs in groups:
        raws = [t for s in segs
                if (t := s.setpoint.get("temperature")) is not None and np.isfinite(t)]
        if not raws:
            continue
        lo, hi = float(min(raws)), float(max(raws))
        reps.append((key, lo, hi))
        if len(raws) >= 2 and (hi - lo) > q.setpoint_unstable_k:
            diags.append(Diagnostic(
                kind="duplicate_setpoints", severity="warning",
                scope=f"field setpoint {key:g} K",
                message=f"unstable hold: {len(raws)} segments span {hi - lo:.3f} K "
                        f"(> {q.setpoint_unstable_k:g} K)",
                data={"setpoint": float(key), "spread": hi - lo, "n_segments": len(raws)}))
    for (k0, _lo0, hi0), (k1, lo1, _hi1) in zip(reps, reps[1:]):
        gap = lo1 - hi0
        if gap < q.setpoint_near_dup_k:
            diags.append(Diagnostic(
                kind="duplicate_setpoints", severity="warning",
                scope=f"field setpoints {k0:g}/{k1:g} K",
                message=f"near-duplicate setpoints: groups {k0:g} K and {k1:g} K are "
                        f"{gap:.3f} K apart (< {q.setpoint_near_dup_k:g} K) — likely one hold",
                data={"setpoints": [float(k0), float(k1)], "delta": float(gap)}))
    return diags


_BRIDGES = (1, 2, 3, 4)


def _hall_routing(df, cmap, segs, cfg):
    """Decide which channel (if any) is Hall-wired and should leave the bridge loop.
    Returns (channel|None, source, det) where source is "detected"|"override" and det is
    the raw detect_hall_channel result (kept so an override/detection disagreement is
    visible in the capability reason). Feature fully off -> (None, "", None) with
    detection not even run (byte-identical behavior)."""
    rcfg = getattr(cfg, "resistivity", None)
    if rcfg is None or not (rcfg.exclude_hall_channel or rcfg.hall_channel_override is not None):
        return None, "", None
    det = detect_hall_channel(df, cmap, segs)
    if rcfg.hall_channel_override is not None:
        return int(rcfg.hall_channel_override), "override", det
    if det is not None:
        return int(det[0]), "detected", det
    return None, "", det


def _routing_reason(ch, source, det):
    if source == "override":
        if det is None:
            note = "detection found no clear winner"
        elif det[0] == ch:
            note = f"detection agrees (odd-in-B fraction {det[1]:.2f})"
        else:
            note = f"detection picked Ch{det[0]} (odd-in-B fraction {det[1]:.2f})"
        return (f"Ch{ch} excluded from resistivity by user override ({note}) — "
                f"analyze it in the Hall tab")
    return (f"Ch{ch} classified as Hall-wired (odd-in-B fraction {det[1]:.2f}) — "
            f"excluded from resistivity; analyze it in the Hall tab")


def _bridge_rho(df, cmap, cfg, ch):
    """Return (rho_ohm_cm ndarray | None, source). Prefer geometry recompute; else
    fall back to the instrument's pre-computed resistivity column (Ohm-m -> Ohm*cm)."""
    res_key = f"resistance_ch{ch}"
    rty_key = f"resistivity_ch{ch}"
    geom = cfg.geometry
    if geom.complete() and res_key in cmap.logical:
        R = pd.to_numeric(df[cmap.logical[res_key]], errors="coerce").to_numpy(float)
        # rho[Ohm*m] = R * A/L ; A = w*t (mm^2 -> m^2), L (mm -> m); then *100 -> Ohm*cm
        A_m2 = geom.width_mm * geom.thickness_mm * 1e-6
        L_m = geom.length_mm * 1e-3
        rho_ohm_cm = R * A_m2 / L_m * 100.0
        if _physical_mask(rho_ohm_cm).any():
            return rho_ohm_cm, "geometry"
    if rty_key in cmap.logical:
        col = pd.to_numeric(df[cmap.logical[rty_key]], errors="coerce").to_numpy(float)
        u = cmap.unit.get(rty_key, UNIT_OHM_M)
        factor = (100.0 if u == UNIT_OHM_M else
                  1.0 if u == UNIT_OHM_CM else
                  1e-6 if u == UNIT_MICROOHM_CM else None)
        if factor is not None:
            rho = col * factor
            if _physical_mask(rho).any():
                return rho, "instrument_column"
    return None, ""


def _bridge_rho_std(df, cmap, ch):
    """(std_ohm_cm ndarray | None, rho_inst_ohm_cm ndarray | None) for channel ch.

    The std column rides the instrument's own resistivity scale, so it is paired with the
    INSTRUMENT resistivity column (same unit factor as the rho path: Ohm-m x100 / Ohm-cm x1).
    RRR is scale-invariant, so the relative sigmas from this self-consistent pair apply to
    the shipped RRR whichever rho_source (geometry / instrument_column) produced it."""
    std_key = f"rho_std_bridge{ch}" if f"rho_std_bridge{ch}" in cmap.logical else f"rho_std_ch{ch}"
    rty_key = f"resistivity_ch{ch}"
    if std_key not in cmap.logical or rty_key not in cmap.logical:
        return None, None
    u_std = cmap.unit.get(std_key, UNIT_OHM_M)
    f_std = 100.0 if u_std == UNIT_OHM_M else 1.0
    u_rho = cmap.unit.get(rty_key, UNIT_OHM_M)
    f_rho = (100.0 if u_rho == UNIT_OHM_M else
             1.0 if u_rho == UNIT_OHM_CM else
             1e-6 if u_rho == UNIT_MICROOHM_CM else None)
    if f_rho is None:
        return None, None
    std = pd.to_numeric(df[cmap.logical[std_key]], errors="coerce").to_numpy(float) * f_std
    rho = pd.to_numeric(df[cmap.logical[rty_key]], errors="coerce").to_numpy(float) * f_rho
    return std, rho


def _rrr_std_for_ramp(df, cmap, cfg, ch, T, idx, rrr):
    """sigma_RRR on the widest zero-field ramp rows, endpoint policy mirroring _rrr (spec §4).

    Pre-masks with resistivity's OWN _clean_mask so the shared helper's selection sees exactly
    the rows behind the shipped RRR endpoints; _endpoint (with cfg) is passed as the endpoint
    function so the k-nearest-median policy is _rrr's own."""
    if rrr is None:
        return None
    std, rho_inst = _bridge_rho_std(df, cmap, ch)
    if std is None:
        return None
    Ti, si, ri = np.asarray(T, float)[idx], std[idx], rho_inst[idx]
    m = _clean_mask(ri, cfg, T=Ti) & np.isfinite(Ti)
    if not m.any():
        return None
    return rrr_sigma(Ti[m], ri[m], si[m], rrr, _RRR_K,
                     lambda Tv, rv, lowest, k: _endpoint(Tv, rv, lowest, k, cfg))


def _build_t_curve(T, rho, seg, cfg=None) -> RhoTCurve:
    idx = seg.idx
    m = _clean_mask(rho[idx], cfg, T=T[idx]) & np.isfinite(T[idx])
    keep = idx[m]
    tc = detect_resistive_tc(T[keep], rho[keep]) if keep.size else None
    return RhoTCurve(held_field_oe=seg.setpoint.get("field"), direction=int(seg.direction),
                     n_points=int(keep.size), classification=_classify(T[keep], rho[keep], cfg),
                     temperature=T[keep].tolist(), rho=rho[keep].tolist(), **(tc or {}))


def _build_h_curve_grouped(H, rho, group_segs, key, cfg=None) -> RhoHCurve:
    """One combined display loop per setpoint. Points = cleaned concatenation of the group's
    field segments (direction=0). MR%/rho0/etc. are computed on the WIDEST constituent ramp
    (largest field span = the full bidirectional sweep) so the numbers are byte-identical to
    the pre-grouping per-ramp value (owner decision: grouping is display-only, MR stays per-ramp)."""
    idx = np.concatenate([s.idx for s in group_segs])
    m = _clean_mask(rho[idx], cfg) & np.isfinite(H[idx])
    keep = idx[m]
    rep = max(group_segs, key=lambda s: float(np.ptp(H[s.idx])) if s.idx.size else 0.0)
    rho0, mr, hmax, low = _mr_percent(H[rep.idx], rho[rep.idx], cfg)
    return RhoHCurve(held_temp_k=float(key), direction=0,
                     n_points=int(keep.size), rho_zero_field=rho0,
                     mr_percent_at_max_field=mr, max_abs_field_oe=hmax, low_confidence=low,
                     field=H[keep].tolist(), rho=rho[keep].tolist(),
                     directions=sorted({int(s.direction) for s in group_segs}))


def _widest_zero_field_ramp(T, tsegs):
    zf = [s for s in tsegs if abs(s.setpoint.get("field") or 0.0) < _ZERO_FIELD_OE]
    if not zf:
        return None
    return max(zf, key=lambda s: float(np.ptp(T[s.idx])) if s.idx.size else 0.0)


def _bridge_result(df, cmap, cfg, ch, T, H, tsegs, fgroups) -> BridgeResult | None:
    rho, source = _bridge_rho(df, cmap, cfg, ch)
    if rho is None:
        return None
    t_curves = [_build_t_curve(T, rho, s, cfg) for s in tsegs]
    t_curves = [c for c in t_curves if c.n_points >= 2]
    h_curves = [_build_h_curve_grouped(H, rho, segs, key, cfg) for key, segs in fgroups]
    h_curves = [c for c in h_curves if c.n_points >= 3]
    br = BridgeResult(channel=ch, rho_source=source, rho_t_curves=t_curves, rho_h_curves=h_curves)
    ramp = _widest_zero_field_ramp(T, tsegs)
    if ramp is not None:
        idx = ramp.idx
        rrr, t_hi, t_lo = _rrr(T[idx], rho[idx], cfg=cfg)
        br.rrr, br.rrr_t_high, br.rrr_t_low = rrr, t_hi, t_lo
        br.rrr_std = _rrr_std_for_ramp(df, cmap, cfg, ch, T, idx, rrr)
        br.classification = _classify(T[idx], rho[idx], cfg)
        # low-T fits on the physical low-T portion of the zero-field ramp
        m = _clean_mask(rho[idx], cfg, T=T[idx]) & np.isfinite(T[idx]) & (T[idx] > 0) & (T[idx] <= _LOWT_MAX_K)
        # SC gate (owner-approved, same rule as the outlier exemption): these fits describe
        # the METALLIC normal state, and below a detected transition the ramp is
        # superconducting - fitting rho0 + A*T^n through the drop yields a meaningless
        # exponent (measured: n = 0.83 at r2 = 0.54 on the SC example, neither Fermi-liquid
        # nor phonon). Fit the normal state (T >= tc_onset_k) only; no transition -> unchanged.
        _mfull = _clean_mask(rho[idx], cfg, T=T[idx]) & np.isfinite(T[idx])
        _tc_ramp = detect_resistive_tc(T[idx][_mfull], rho[idx][_mfull]) if _mfull.any() else None
        _onset = (_tc_ramp or {}).get("tc_onset_k")
        if _onset is not None:
            m = m & (T[idx] >= _onset)
        Tk, Rk = T[idx][m], rho[idx][m]
        if Tk.size >= 2 and np.ptp(Tk) > 0:
            try:
                br.low_t_linear = LinearFitModel().fit(Tk, Rk)
            except ValueError:
                pass
        if br.classification == "metallic" and Tk.size >= 4:
            try:
                # Tk is already masked to <= _LOWT_MAX_K (30 K), so the ladder's primary
                # (30 K) rung reproduces the pre-ladder shipped fit exactly (U1).
                # Ladder rungs must span the window ACTUALLY fitted. The fixed 10/15/20/30 K
                # cutoffs are measured from 0, so when a transition gates the window from below
                # (onset 8.8 K on the SC example) three of the four rungs land inside a 1-11 K
                # stub and pin at the search bound - the ladder then reports a spread of 0.118
                # for an exponent that truly runs 0.5->0.99. Relative rungs only when gated, so
                # every ungated curve keeps the absolute cutoffs byte-identically.
                rungs = _RHO_LADDER_RUNGS
                if _onset is not None and Tk.size:
                    lo, hi = float(Tk.min()), float(Tk.max())
                    if hi > lo:
                        rungs = tuple(lo + f * (hi - lo)
                                      for f in (0.25, 0.5, 0.75)) + (_LOWT_MAX_K,)
                pl, pl_ladder, pl_spread = fit_rho_powerlaw_ladder(Tk, Rk, rungs=rungs)
                br.power_law = pl
                br.power_law_ladder = pl_ladder or None
                br.power_law_n_spread = pl_spread
                if not (set(pl.quality_flags) & NO_FIT_LINE_FLAGS):
                    br.residual_rho = pl.params["rho0"]   # only report a resolved residual
            except (ValueError, RuntimeError):
                pass
            try:
                br.rho_t2_linear = RhoT2FermiLiquidModel().fit(Tk, Rk)
            except ValueError:
                pass
        if br.classification == "insulating":
            # Activated transport is a WHOLE-WINDOW statement (no 30 K cap: the exponential
            # regime is the high-T side); fit the full physical zero-field ramp.
            mi = (_clean_mask(rho[idx], cfg, T=T[idx]) & np.isfinite(T[idx])
                  & (T[idx] > 0))
            Ti, Ri = T[idx][mi], rho[idx][mi]
            if Ti.size >= 4 and np.ptp(Ti) > 0:
                try:
                    fit, ladder, spread, alts = fit_arrhenius_ladder(Ti, Ri)
                    br.arrhenius = fit
                    br.arrhenius_ladder = ladder or None
                    br.arrhenius_ea_spread_mev = spread
                    br.arrhenius_alt_models = alts
                except (ValueError, RuntimeError):
                    pass
    return br


_ARRHENIUS_DECLINE_TEXT = {
    "insufficient_rho_span": ("under one e-fold of rho change in the window (an exponential "
                              "is indistinguishable from a straight line there)"),
    "ea_unresolved": "sigma swamps E_a (slope not resolved by this window)",
}


def _activated_transport_cap(bridges, has_insulating) -> Capability:
    fits = [b.arrhenius for b in bridges if b.arrhenius is not None]
    clean = [f for f in fits if not (set(f.quality_flags) & ARRHENIUS_DECLINE_FLAGS)]
    if clean:
        return Capability(
            name="activated_transport", applicable=True,
            reason=("Arrhenius E_a reported as measured; E_g = 2*E_a only under the "
                    "intrinsic-conduction assumption (extrinsic factor is 1) - transport "
                    "alone cannot tell the regimes apart"))
    if fits:
        why = "; ".join(sorted({_ARRHENIUS_DECLINE_TEXT.get(fl, fl)
                                for f in fits
                                for fl in set(f.quality_flags) & ARRHENIUS_DECLINE_FLAGS}))
        return Capability(name="activated_transport", applicable=False,
                          reason=f"insulating rho(T) but the Arrhenius fit declined: {why}")
    return Capability(name="activated_transport", applicable=False,
                      reason=("insulating rho(T) detected but no fittable zero-field ramp"
                              if has_insulating
                              else "not applicable: no insulating rho(T) segment"))


def _capabilities(bridges, tsegs, fsegs) -> list[Capability]:
    has_zero_field_ramp = any(b.rrr is not None for b in bridges)
    has_field_loop = len(fsegs) > 0
    has_metallic = any(b.classification == "metallic" for b in bridges)
    has_insulating = any(c.classification == "insulating"
                         for b in bridges for c in b.rho_t_curves)
    # Closed O7: window_sensitive is an honesty annotation, not a fit failure — without the
    # carve-out the flag would flip power_law_fit to non-applicable on all three real files.
    has_clean_powerlaw = any(
        b.power_law and not (set(b.power_law.quality_flags) - _HONESTY_FLAGS)
        for b in bridges)
    has_tc = any(c.tc_mid_k is not None for b in bridges for c in b.rho_t_curves)
    caps = [
        Capability(name="curve_separation", applicable=True,
                   reason=f"{len(tsegs)} temperature sweeps + {len(fsegs)} field sweeps"),
        Capability(name="RRR", applicable=has_zero_field_ramp,
                   reason="zero-field temperature ramp present" if has_zero_field_ramp
                   else "no wide zero-field temperature ramp"),
        Capability(name="magnetoresistance", applicable=has_field_loop,
                   reason="field sweep(s) present" if has_field_loop else "no field sweep present"),
        Capability(name="power_law_fit", applicable=has_clean_powerlaw,
                   reason=("clean metallic low-T power-law fit obtained" if has_clean_powerlaw
                           else "metallic ramp present but residual rho0 unresolved (low-T window too high)"
                           if has_metallic
                           else "no metallic ramp (rho falls with T)")),
        Capability(name="linear_fit", applicable=len(tsegs) > 0,
                   reason="temperature ramp present" if tsegs else "no temperature ramp"),
        # recognized-but-not-yet-implemented analyses: report data applicability + reason
        _activated_transport_cap(bridges, has_insulating),
        Capability(name="superconducting_transition", applicable=has_tc,
                   reason=("resistive drop detected" if has_tc
                           else "no resistive drop")),
    ]
    return caps


def _header_geometry_unset(header, ch: int) -> bool:
    """Spec §11: `SampleN Cross Section = 1` / `Length = 1` (or absent) means the user did
    not change the geometry settings in the PPMS software — UNSET data, not a malformed
    file. With unity geometry the instrument's resistivity column is resistance times an
    arbitrary factor."""
    info = getattr(header, "info", None) or {}

    def _unset(v):
        if v is None:
            return True
        try:
            return float(str(v)) == 1.0
        except (TypeError, ValueError):
            return True
    return (_unset(info.get(f"Sample{ch} Cross Section"))
            and _unset(info.get(f"Sample{ch} Length")))


def _geometry_unset_warnings(bridges, header) -> list[str]:
    """One §11 warning per instrument-column bridge whose header geometry is unset.
    Always-on, purely additive — nothing already reported changes in value. States what the
    arbitrary scale does and does not touch, and names the app's own remedy."""
    warns = []
    for b in bridges:
        if b.rho_source == "instrument_column" and _header_geometry_unset(header, b.channel):
            warns.append(
                f"ch{b.channel}: ρ scale is arbitrary — the file's resistivity column was "
                "computed with unset instrument geometry (Cross Section = Length = 1) and no "
                "sample geometry was supplied. Ratios are unaffected (RRR, MR%); absolute ρ "
                "and the power-law residual ρ₀ are scale-arbitrary. Enter width / thickness / "
                "length in the resistivity panel to recompute ρ from geometry.")
    return warns


class ResistivityAnalyzer:
    probe = "resistivity"
    needs = (Need("sample_width_mm", scope="sample", required=False),
             Need("sample_thickness_mm", scope="sample", required=False),
             Need("sample_length_mm", scope="sample", required=False))

    def analyze(self, rawtable, cfg) -> Result:
        df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
        header = rawtable.header
        prov = Provenance(file=getattr(rawtable, "path", None) or header.title or "",
                          sha256=_sha256(getattr(rawtable, "path", None)),
                          app_version=header.app_version, config=cfg.model_dump(mode="json"))
        T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float) \
            if "temperature" in cmap.logical else None
        H = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float) \
            if "field" in cmap.logical else None
        if T is None:
            return Result(status="error", errors=["resistivity needs a temperature column"],
                          data={"probe": "resistivity"}, provenance=prov)
        if H is None:
            # Bare zero-field rho(T) export (no Field column): treat as H=0 so the single
            # ramp yields rho(T)/RRR; no field loops -> magnetoresistance reports inapplicable.
            H = np.zeros(len(df), float)
        segs = segment_sweeps(df, cmap, cfg)
        tsegs = [s for s in segs if s.swept.name == "temperature"]
        fsegs = [s for s in segs if s.swept.name == "field"]
        fgroups = group_segments_by_setpoint(fsegs, "temperature", cfg.quality.setpoint_threshold_k)
        bridges = []
        for ch in _BRIDGES:
            br = _bridge_result(df, cmap, cfg, ch, T, H, tsegs, fgroups)
            if br is not None:
                bridges.append(br)
        if not bridges:
            return Result(status="error", errors=["no populated resistivity bridge found"],
                          data={"probe": "resistivity"}, provenance=prov)
        # Hall-channel routing: a clear odd-in-B winner (or user override) is Hall-wired,
        # not longitudinal — route it out of resistivity. Never leaves zero bridges.
        excl_ch, excl_src, det = _hall_routing(df, cmap, segs, cfg)
        excluded_channel, excluded_source, routing_cap = None, "", None
        if excl_ch is not None and any(b.channel == excl_ch for b in bridges):
            kept = [b for b in bridges if b.channel != excl_ch]
            reason = _routing_reason(excl_ch, excl_src, det)
            if kept:
                bridges = kept
                excluded_channel, excluded_source = excl_ch, excl_src
                routing_cap = Capability(name="hall_channel_excluded", applicable=True,
                                         reason=reason)
            else:
                routing_cap = Capability(
                    name="hall_channel_excluded", applicable=False,
                    reason=f"Ch{excl_ch} scores as Hall-wired but is the only populated "
                           f"bridge — kept in resistivity (exclusion would leave no bridges)")
        sources = {b.rho_source for b in bridges}
        top_source = next(iter(sources)) if len(sources) == 1 else "mixed"
        caps = _capabilities(bridges, tsegs, fsegs)
        if routing_cap is not None:
            caps.append(routing_cap)
        rdata = ResistivityData(probe="resistivity", rho_source=top_source,
                                bridges=bridges, capabilities=caps,
                                excluded_hall_channel=excluded_channel,
                                excluded_hall_source=excluded_source)
        # confidence: only CLEAN power-law fits (no quality flags) count; otherwise fall
        # back to RRR presence. A marginal/unresolved fit must not inflate confidence.
        clean_r2s = [b.power_law.r2 for b in bridges
                     if b.power_law and b.power_law.r2 is not None
                     and not (set(b.power_law.quality_flags) - _HONESTY_FLAGS)]
        if clean_r2s:
            conf = float(np.mean(clean_r2s))
        elif any(b.rrr is not None for b in bridges):
            conf = 0.7
        else:
            conf = 0.4
        diags = (_outlier_diagnostics(bridges, cfg.quality.outlier_k)
                 + _duplicate_setpoint_diagnostics(fsegs, cfg))
        status = "ok" if conf >= cfg.confidence_min else "low_confidence"
        return Result(status=status, confidence=conf,
                      confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                        "fit": (float(np.mean(clean_r2s)) if clean_r2s else None)},
                      data=rdata.model_dump(mode="json"), provenance=prov,
                      warnings=_geometry_unset_warnings(bridges, header),
                      diagnostics=diags)
