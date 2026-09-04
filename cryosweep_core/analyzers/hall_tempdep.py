"""hall_tempdep.py — result models + fixed-field interpolation helper for SP-7 temp-dep Hall.

Responsibility: Pydantic result models (Capability, HallTDepPoint, HallTDepStage,
DualMethodPoint, InterpCurve, HallTempDepData) and the pure helper
_interp_fixed_field_curves that groups temperature-ramp segments by held field and
interpolates each onto a common T grid masked to its native range.

The reconstruction core and HallTempDepAnalyzer come in Tasks 5/6.
"""
from __future__ import annotations
import hashlib, math, pathlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from cryosweep_core.detect.sweeps import segment_sweeps
from cryosweep_core.analyzers.hall import (_carrier_n, _mobility,
                                      _long_rho_xx, field_sweep_points)
from cryosweep_core.fitting.transport import LinearFitModel
from cryosweep_core.result import Result, Provenance
from cryosweep_core.registry import Need
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.grouping import cluster_field_setpoints

_OE_PER_T = 10000.0


# ---- typed result models ---------------------------------------------------

class Capability(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    applicable: bool
    reason: str = ""


class HallTDepPoint(BaseModel):
    """One temperature point in the temp-dep Hall reconstruction."""
    model_config = ConfigDict(extra="ignore")
    temperature: float
    field_count: int = 0
    antisym_points: int = 0
    slope_pos_ohm_per_T: float | None = None
    slope_neg_ohm_per_T: float | None = None
    slope_ohm_per_T: float | None = None
    r2: float | None = None
    R_H: float | None = None              # m^3/C
    carrier_n: float | None = None        # 1/m^3
    carrier_type: str | None = None
    rho_xx: float | None = None           # Ohm*m
    sigma: float | None = None            # S/m
    mobility: float | None = None         # m^2/(V*s)
    current_density_J: float | None = None  # A/m^2 (optional)
    antisymmetrized: bool = False
    low_confidence: bool = False
    # Sub-feature B (append-only): 2-point zero-subtracted R_H fallback provenance
    r_h_method: str | None = None          # "antisym" | "2point" | None
    slope_2point_ohm_per_T: float | None = None
    # --- 2026-08-10 uncertainty-honesty additive fields (spec §2.2, closed O4; declared
    # LAST: append-only JSON key order). NB `sigma` above is CONDUCTIVITY in S/m (U5) —
    # uncertainty fields never reuse that name. ---
    # Residual (fit-quality) sigma — None unless >= 3 antisym points (zero-DOF: U4)
    slope_sigma_ohm_per_T: float | None = None
    r_h_sigma: float | None = None
    carrier_n_sigma: float | None = None
    mobility_sigma: float | None = None
    # Instrument repeat-noise sigma (closed O4) — a WEAKER, DIFFERENT claim than the
    # residual sigma above: propagated from the file's Bridge N Std. Dev. column, it
    # measures instrument noise, NOT fit quality. The _instrument suffix is load-bearing.
    slope_sigma_instrument_ohm_per_T: float | None = None
    r_h_sigma_instrument: float | None = None
    carrier_n_sigma_instrument: float | None = None
    mobility_sigma_instrument: float | None = None
    sigma_zero_dof: bool = False


class HallTDepStage(BaseModel):
    """Intermediate stage data at one temperature (for diagnostics / plotting)."""
    model_config = ConfigDict(extra="ignore")
    temperature: float
    fields_T: list[float] = []
    R_raw: list[float] = []
    R_zero_sub: list[float] = []
    R_asym: list[float] = []
    fit_slope: float | None = None
    fit_intercept: float | None = None


class DualMethodPoint(BaseModel):
    """Combined result at one temperature from both the field-sweep and temp-dep methods."""
    model_config = ConfigDict(extra="ignore")
    temperature: float
    R_H_tempdep: float | None = None
    R_H_fieldsweep: float | None = None
    R_H_combined: float | None = None
    n_combined: float | None = None


class InterpCurve(BaseModel):
    """Interpolated R(T) curve at one fixed field, on the common temperature grid."""
    model_config = ConfigDict(extra="ignore")
    field_oe: float
    temperature: list[float] = []
    R: list[float] = []                   # interpolated transverse resistance (Ohm)


class HallTempDepData(BaseModel):
    """Top-level result container for temp-dep Hall analysis."""
    model_config = ConfigDict(extra="ignore")
    probe: str = "hall_tdep"
    hall_channel: int | None = None
    thickness_m: float | None = None
    geometry_sign: int = 1
    temp_interval: float = 1.0
    longitudinal_source: str | None = None
    points: list[HallTDepPoint] = []
    stages: list[HallTDepStage] = []
    interp_curves: list[InterpCurve] = []
    dual_method: list[DualMethodPoint] = []
    capabilities: list[Capability] = []


# ---- pure helpers ----------------------------------------------------------

def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""


def _interp_fixed_field_curves(
    df, cmap, cfg, hall_channel: int, temp_interval: float
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    """Map held_field_oe -> (T_grid, R_grid) for each fixed-field temperature ramp.

    Each curve is masked to its native [T_min, T_max] — no extrapolation.
    Duplicate T values within a field group are collapsed by mean before interpolation
    so that np.interp receives a monotone grid.

    Uses segment_sweeps to identify temperature-swept segments and reads the held-field
    setpoint from the segment metadata — the same production path on real files. Fields
    whose ramp is shorter than the stability window are not resolved by the segmenter and
    are simply absent from the result (the reconstruction's sparsity guard handles this).

    Parameters
    ----------
    df          : DataFrame from canonicalize_columns
    cmap        : ColumnMap from canonicalize_columns
    cfg         : RunConfig (passed to segment_sweeps)
    hall_channel: bridge number carrying the transverse (Hall) signal
    temp_interval: spacing (K) for the common temperature grid

    Returns
    -------
    dict mapping float(field_oe) -> (T_grid, R_grid), both 1-D arrays of equal length >= 2.
    Fields with fewer than 2 usable points are omitted.
    """
    T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    R = pd.to_numeric(df[cmap.logical[f"resistance_ch{hall_channel}"]], errors="coerce").to_numpy(float)

    # Segmenter-based grouping: each fixed-field temperature ramp surfaces as a
    # temperature-swept segment carrying its held-field setpoint.
    tsegs = [s for s in segment_sweeps(df, cmap, cfg) if s.swept.name == "temperature"]
    # #19 (2026-09-02): held fields are clustered across segments, not round()-binned —
    # two ramps of the same physical field whose medians straddle an integer edge (e.g.
    # 40000.887 / 39999.586 Oe, the measured VSM case) must form ONE curve. Same
    # cluster-to-group / setpoint_key-to-label rule and defaults as the VSM fix (3d722ff);
    # verified identical grouping to the old round() on every in-tree and real file.
    withF = [(s, float(f)) for s in tsegs
             if (f := s.setpoint.get("field")) is not None]
    flabels = cluster_field_setpoints([f for _s, f in withF])
    by_field: dict[float, list] = {}
    for (s, _f), lab in zip(withF, flabels):
        if np.isfinite(lab):
            by_field.setdefault(float(lab), []).append(s.idx)

    curves: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for f, idxs in by_field.items():
        idx = np.concatenate(idxs)
        Ti, Ri = T[idx], R[idx]
        m = np.isfinite(Ti) & np.isfinite(Ri)
        Ti, Ri = Ti[m], Ri[m]
        if Ti.size < 2:
            continue
        order = np.argsort(Ti)
        Ti, Ri = Ti[order], Ri[order]
        # collapse duplicate T values by mean -> monotone grid for np.interp
        uT = np.unique(Ti)
        uR = np.array([Ri[Ti == t].mean() for t in uT])
        grid = np.arange(uT.min(), uT.max() + temp_interval, temp_interval)
        grid = grid[(grid >= uT.min()) & (grid <= uT.max())]   # no extrapolation
        if grid.size < 2:
            continue
        curves[float(f)] = (grid, np.interp(grid, uT, uR))

    return curves


_RATIO_CONSTANCY_TOL = 1e-6   # hardening 1 (2026-08-10): required rel spread of R/rho


def _interp_fixed_field_sigma_curves(df, cmap, cfg, hall_channel: int, temp_interval: float):
    """Instrument sigma_R curves per held field, aligned with _interp_fixed_field_curves.

    sigma_R (Ohm) = std_column (Ohm-m) * (Resistance/Resistivity) — the file's own exact
    geometry factor. Measured 2026-08-10 on both real files: the per-row ratio is constant
    to ~1e-12 relative spread (1000 exactly); the header geometry fields are NOT a valid
    source (unset dummies predicting a 10x-wrong factor).

    HARDENING (required): the ratio's relative spread must be < 1e-6, else this function
    DECLINES (returns None) and every *_sigma_instrument field stays None — the measured
    constancy is a runtime gate, not an assumption. Also None when any needed column is
    absent. Collapse of duplicate T rows uses the mean (conservative vs mean/sqrt(k))."""
    std_key = (f"rho_std_bridge{hall_channel}"
               if f"rho_std_bridge{hall_channel}" in cmap.logical
               else f"rho_std_ch{hall_channel}")
    res_key = f"resistance_ch{hall_channel}"
    rty_key = f"resistivity_ch{hall_channel}"
    if (std_key not in cmap.logical or res_key not in cmap.logical
            or rty_key not in cmap.logical):
        return None
    Rr = pd.to_numeric(df[cmap.logical[res_key]], errors="coerce").to_numpy(float)
    Rh = pd.to_numeric(df[cmap.logical[rty_key]], errors="coerce").to_numpy(float)
    SD = pd.to_numeric(df[cmap.logical[std_key]], errors="coerce").to_numpy(float)
    mr = np.isfinite(Rr) & np.isfinite(Rh) & (Rh != 0.0)
    if not mr.any():
        return None
    ratios = Rr[mr] / Rh[mr]
    med = float(np.median(ratios))
    if med == 0.0 or not np.isfinite(med):
        return None
    spread = float((np.max(ratios) - np.min(ratios)) / abs(med))
    if not (spread < _RATIO_CONSTANCY_TOL):
        return None                                # DECLINE, never emit a shaky sigma
    sigma_R = SD * med                             # per-row; med == per-row ratio (gated)

    T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    tsegs = [s for s in segment_sweeps(df, cmap, cfg) if s.swept.name == "temperature"]
    # #19 (2026-09-02): held fields are clustered across segments, not round()-binned —
    # two ramps of the same physical field whose medians straddle an integer edge (e.g.
    # 40000.887 / 39999.586 Oe, the measured VSM case) must form ONE curve. Same
    # cluster-to-group / setpoint_key-to-label rule and defaults as the VSM fix (3d722ff);
    # verified identical grouping to the old round() on every in-tree and real file.
    withF = [(s, float(f)) for s in tsegs
             if (f := s.setpoint.get("field")) is not None]
    flabels = cluster_field_setpoints([f for _s, f in withF])
    by_field: dict[float, list] = {}
    for (s, _f), lab in zip(withF, flabels):
        if np.isfinite(lab):
            by_field.setdefault(float(lab), []).append(s.idx)
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for f, idxs in by_field.items():
        idx = np.concatenate(idxs)
        Ti, Si = T[idx], sigma_R[idx]
        m = np.isfinite(Ti) & np.isfinite(Si)
        Ti, Si = Ti[m], Si[m]
        if Ti.size < 2:
            continue
        order = np.argsort(Ti)
        Ti, Si = Ti[order], Si[order]
        uT = np.unique(Ti)
        uS = np.array([Si[Ti == t].mean() for t in uT])
        grid = np.arange(uT.min(), uT.max() + temp_interval, temp_interval)
        grid = grid[(grid >= uT.min()) & (grid <= uT.max())]
        if grid.size < 2:
            continue
        out[float(f)] = (grid, np.interp(grid, uT, uS))
    return out or None


def _reconstruct_points(
    curves: dict[float, tuple[np.ndarray, np.ndarray]],
    thickness_m: float | None,
    geometry_sign: int,
    min_antisym: int,
    want_stages: bool,
    two_point_fallback: bool = False,
    sd_curves: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[list[HallTDepPoint], list[HallTDepStage]]:
    """Reconstruct Hall coefficients at each common temperature across all fixed-field curves.

    At each common T:
    1. Collect R(T, field) for every curve via np.interp (tolerant of offset T grids).
    2. Antisymmetrize over fuzzy-matched +/-|B| pairs (tolerance 200 Oe — handles real-
       instrument rounding like +20000 vs -19999 Oe).
    3. Linear-fit R_asym vs B (B in Tesla) → slope.
    4. R_H = slope * thickness_m * geometry_sign.
    5. Set field_count, antisym_points, low_confidence, carrier.

    R_H = None when antisym_points == 0 (no ± pair to antisymmetrize) or thickness_m
    is None. A single ± pair (#18) fits through the origin (R_asym(0) = 0 exactly).
    By antisym construction slope_pos == slope_neg == slope_avg; all three set to slope.

    Common T grid: integer-step range spanning the strict intersection of paired-field
    T ranges (zero/near-zero fields not constrained — avoids spurious zero-field-only
    ramps from pulling the common range to a non-overlapping region).
    """
    if not curves:
        return [], []

    _PAIR_TOL = 200.0   # Oe tolerance for fuzzy ±B pairing

    fields = sorted(curves)

    # --- fuzzy antisym pairing ------------------------------------------------
    # For each positive field, claim the closest still-unclaimed negative field
    # within _PAIR_TOL. Each negative is consumed once (del below) so two positive
    # setpoints within tolerance of one negative can't both pair to it (which would
    # double-count a magnitude and inflate antisym_points).
    neg_fields: dict[float, float] = {abs(f): f for f in fields if f < 0}
    pos_fields = sorted(f for f in fields if f > 0)
    paired_mags: list[tuple[float, float]] = []   # (pos_field_oe, neg_field_oe)
    for pf in pos_fields:
        if not neg_fields:
            break
        closest_mag = min(neg_fields.keys(), key=lambda m, _pf=pf: abs(m - _pf))
        if abs(closest_mag - pf) <= _PAIR_TOL:
            paired_mags.append((pf, neg_fields[closest_mag]))
            del neg_fields[closest_mag]            # consume — never reuse a negative

    paired_field_set = {f for pair in paired_mags for f in pair}

    # --- common T grid: paired ranges, optionally widened by (zero ∩ other) -----
    # Zero/near-zero fields alone must NEVER constrain the grid (spurious zero-only
    # ramps would pull it to a non-overlapping region). When two_point_fallback is on,
    # widen to include temperatures where a near-zero field AND >=1 other field overlap.
    zero_fields = [f for f in fields if abs(f) <= _PAIR_TOL]
    other_fields = [f for f in fields if abs(f) > _PAIR_TOL]

    lo_hi: list[tuple[float, float]] = []
    if paired_field_set:
        pl = max(float(curves[f][0].min()) for f in paired_field_set)
        ph = min(float(curves[f][0].max()) for f in paired_field_set)
        if ph >= pl:
            lo_hi.append((pl, ph))
    if two_point_fallback and zero_fields and other_fields:
        for zf in zero_fields:
            zlo, zhi = float(curves[zf][0].min()), float(curves[zf][0].max())
            for of in other_fields:
                olo = max(zlo, float(curves[of][0].min()))
                ohi = min(zhi, float(curves[of][0].max()))
                if ohi >= olo:
                    lo_hi.append((olo, ohi))            # zero ∩ other, never zero-alone
    if not lo_hi and not paired_field_set:
        # legacy no-paired fallback: intersection of ALL curves
        allo = max(float(curves[f][0].min()) for f in fields)
        alhi = min(float(curves[f][0].max()) for f in fields)
        if alhi >= allo:
            lo_hi.append((allo, alhi))
    if not lo_hi:
        return [], []

    T_lo = min(lo for lo, _ in lo_hi)
    T_hi = max(hi for _, hi in lo_hi)

    T_start = int(np.ceil(T_lo - 1e-9))
    T_end = int(np.floor(T_hi + 1e-9))
    common_T = [float(t) for t in range(T_start, T_end + 1)]
    if not common_T:
        return [], []

    pts: list[HallTDepPoint] = []
    stages: list[HallTDepStage] = []

    for T in common_T:
        # Collect R at T from every curve that covers T, via interpolation
        Rmap: dict[float, float] = {}
        for f in fields:
            Tg, Rg = curves[f]
            if float(Tg.min()) - 1e-9 <= T <= float(Tg.max()) + 1e-9:
                Rmap[f] = float(np.interp(T, Tg, Rg))

        field_count = len(Rmap)

        # O4: interpolate the instrument sigma_R onto the SAME per-T grid (SDmap ~ Rmap)
        SDmap: dict[float, float] = {}
        if sd_curves:
            for f, (Tg, Sg) in sd_curves.items():
                if float(Tg.min()) - 1e-9 <= T <= float(Tg.max()) + 1e-9:
                    v = float(np.interp(T, Tg, Sg))
                    if np.isfinite(v):
                        SDmap[f] = v

        # Antisymmetrize over fuzzy-matched ±B pairs (both must be in Rmap at this T)
        B: list[float] = []
        Rasym: list[float] = []
        Rraw: list[float] = []
        Sasym: list[float | None] = []   # per-pair instrument sigma of the antisym point
        for pf, nf in paired_mags:
            if pf in Rmap and nf in Rmap:
                B.append(pf / _OE_PER_T)                      # use positive magnitude, Oe → T
                Rasym.append((Rmap[pf] - Rmap[nf]) / 2.0)    # antisymmetric part
                Rraw.append(Rmap[pf])
                Sasym.append(math.sqrt(SDmap[pf] ** 2 + SDmap[nf] ** 2) / 2.0
                             if (pf in SDmap and nf in SDmap) else None)

        antisym_points = len(B)

        pt = HallTDepPoint(
            temperature=float(T),
            field_count=field_count,
            antisym_points=antisym_points,
            antisymmetrized=antisym_points >= 1,
            low_confidence=antisym_points < min_antisym,
        )

        if antisym_points >= 2:
            fit = LinearFitModel().fit(np.array(B), np.array(Rasym), xunit="T", yunit="Ohm")
            slope = float(fit.params["slope"])
            pt.slope_ohm_per_T = slope
            pt.slope_pos_ohm_per_T = slope    # by antisym construction pos==neg==avg
            pt.slope_neg_ohm_per_T = slope
            pt.r2 = float(fit.r2)
            pt.R_H = (slope * thickness_m * geometry_sign) if thickness_m is not None else None
            pt.carrier_n, pt.carrier_type = _carrier_n(pt.R_H)
            pt.r_h_method = "antisym"
            # Residual sigma (spec §2.2): >= 3 antisym points -> linregress stderr; exactly
            # 2 -> zero residual DOF, stderr is 0.0 -> None + sigma_zero_dof (U4).
            if antisym_points >= 3:
                ssig = float(fit.sigma["slope"])
                pt.slope_sigma_ohm_per_T = ssig if np.isfinite(ssig) else None
            else:
                pt.sigma_zero_dof = True
            # Instrument sigma (closed O4): exact linear propagation through the same
            # OLS-with-intercept estimator: w_i = (B_i - Bbar)/sum((B - Bbar)^2),
            # var_slope = sum(w_i^2 sigma_asym_i^2).
            if all(s is not None for s in Sasym):
                Ba = np.array(B)
                dev = Ba - float(Ba.mean())
                denom = float(np.sum(dev ** 2))
                if denom > 0:
                    var = float(np.sum((dev / denom) ** 2 * np.array(Sasym, float) ** 2))
                    if np.isfinite(var):
                        pt.slope_sigma_instrument_ohm_per_T = math.sqrt(var)

        elif antisym_points == 1:
            # KNOWN-ISSUES #18 (2026-09-02): a single symmetric ± pair IS an
            # antisymmetrization — R_asym = [R(+B) − R(−B)]/2 cancels the even-in-B
            # admixture exactly, and the antisym construction forces R_asym(0) = 0,
            # so the fit is anchored through the origin: slope = R_asym/B. On the
            # real Hall file 121 of 138 T points live here and were previously
            # mislabelled "2point"/low_confidence (bit-identical numbers, disowned).
            slope = float(Rasym[0] / B[0])
            pt.slope_ohm_per_T = slope
            pt.slope_pos_ohm_per_T = slope
            pt.slope_neg_ohm_per_T = slope
            pt.R_H = (slope * thickness_m * geometry_sign) if thickness_m is not None else None
            pt.carrier_n, pt.carrier_type = _carrier_n(pt.R_H)
            pt.r_h_method = "antisym"
            # r2 stays None (one point through the origin makes no linearity claim)
            # and residual sigma stays None — zero residual DOF (U4).
            pt.sigma_zero_dof = True
            # Instrument sigma through the same through-origin estimator:
            # var(slope) = sigma_asym^2 / B^2.
            if Sasym[0] is not None:
                pt.slope_sigma_instrument_ohm_per_T = float(Sasym[0] / B[0])

        # --- Sub-feature B: zero-field-subtracted 2-point R_H fallback -----------
        # Only where NO antisym fit ran (r_h_method is None, i.e. zero ± pairs), a
        # near-zero field plus >=1 other field are present at this T. Through-origin
        # least squares over zero-subtracted points: slope = Σ(B_i·y_i)/Σ(B_i²) with
        # y_i = R(f_i) − R(0). The single (0, one-field) case reduces to (R(B)−R(0))/B.
        # Never overwrites an antisym R_H (single-pair included, #18).
        if two_point_fallback and pt.r_h_method is None:
            zero_here = [f for f in Rmap if abs(f) <= _PAIR_TOL]
            other_here = [f for f in Rmap if abs(f) > _PAIR_TOL]
            if zero_here and other_here:
                R0 = Rmap[min(zero_here, key=abs)]
                Bx = np.array([f / _OE_PER_T for f in other_here])
                yy = np.array([Rmap[f] - R0 for f in other_here])
                denom = float(np.sum(Bx * Bx))
                if denom > 0:
                    slope2 = float(np.sum(Bx * yy) / denom)
                    pt.slope_2point_ohm_per_T = slope2
                    pt.r_h_method = "2point"
                    pt.low_confidence = True
                    if thickness_m is not None:
                        pt.R_H = slope2 * thickness_m * geometry_sign
                        pt.carrier_n, pt.carrier_type = _carrier_n(pt.R_H)
                    # Instrument sigma (closed O4), through-origin estimator on
                    # y_i = R(B_i) - R(0). Every y_i shares the SAME R(0), so the y_i are
                    # CORRELATED and the shared term does not sum independently:
                    #   var(slope) = [ sum_i B_i^2 sigma_{B_i}^2 + (sum_i B_i)^2 sigma_0^2 ]
                    #                / (sum_i B_i^2)^2
                    # F4 (final-review): this used (sum B_i^2) sigma_0^2 in place of
                    # (sum B_i)^2 sigma_0^2, i.e. treated the shared zero-field point as k
                    # independent measurements. On the real Hall file the fallback runs at 121 of 138
                    # points with k = 2 fields at exactly +-9 T, where sum B_i = 0 and the
                    # correct coefficient is 0 — so sigma_inst was sqrt(2) too large on 88 %
                    # of the R_H(T) curve. (Identical at k = 1, which is why the synthetic
                    # closed-form fixture could not see it.) The estimator is now exact, as
                    # the code comment and physics-reference.md always claimed.
                    # Residual sigma stays None — zero residual DOF by construction (U4).
                    zf = min(zero_here, key=abs)
                    if zf in SDmap and all(f in SDmap for f in other_here):
                        s0sq = SDmap[zf] ** 2
                        sB2 = np.array([SDmap[f] ** 2 for f in other_here])
                        var2 = float((np.sum(Bx ** 2 * sB2)
                                      + float(np.sum(Bx)) ** 2 * s0sq) / denom ** 2)
                        if np.isfinite(var2):
                            pt.slope_sigma_instrument_ohm_per_T = math.sqrt(var2)

        # Derived sigma companions, each family independently (spec §2.2): x thickness for
        # R_H (|sign| = 1, thickness carries no sigma — U9), pure relative for carrier n.
        if thickness_m is not None:
            if pt.slope_sigma_ohm_per_T is not None:
                pt.r_h_sigma = pt.slope_sigma_ohm_per_T * thickness_m
            if pt.slope_sigma_instrument_ohm_per_T is not None:
                pt.r_h_sigma_instrument = pt.slope_sigma_instrument_ohm_per_T * thickness_m
        if pt.R_H and pt.carrier_n is not None:
            if pt.r_h_sigma is not None:
                pt.carrier_n_sigma = float(pt.carrier_n * pt.r_h_sigma / abs(pt.R_H))
            if pt.r_h_sigma_instrument is not None:
                pt.carrier_n_sigma_instrument = float(
                    pt.carrier_n * pt.r_h_sigma_instrument / abs(pt.R_H))

        if want_stages:
            stages.append(HallTDepStage(
                temperature=float(T),
                fields_T=list(B),
                R_raw=list(Rraw),
                R_zero_sub=list(Rraw),   # zero-field subtraction is identity here
                R_asym=list(Rasym),
                fit_slope=pt.slope_ohm_per_T,
                fit_intercept=None,
            ))

        pts.append(pt)

    return pts, stages


# ---- derived quantities helper --------------------------------------------

def _sigma_mu_J(pt, rho_fn):
    """Fill sigma / mobility in-place on a HallTDepPoint from a rho_fn(T) callable.
    Returns pt for convenience (mutation is the primary effect)."""
    if rho_fn is not None:
        pt.rho_xx = rho_fn(pt.temperature)
        if pt.rho_xx and pt.rho_xx > 0:
            pt.sigma = 1.0 / pt.rho_xx
            pt.mobility = _mobility(pt.R_H, pt.rho_xx)
            # sigma companions (each family; rho_xx sigma NOT folded — deferred §10)
            if pt.mobility is not None and pt.R_H:
                if pt.r_h_sigma is not None:
                    pt.mobility_sigma = float(pt.mobility * pt.r_h_sigma / abs(pt.R_H))
                if pt.r_h_sigma_instrument is not None:
                    pt.mobility_sigma_instrument = float(
                        pt.mobility * pt.r_h_sigma_instrument / abs(pt.R_H))
    return pt


# ---- capabilities assembler -----------------------------------------------

def _capabilities(points, has_thickness, long_source, has_dual, min_antisym_pts=3):
    """Assemble a list of Capability objects describing what this analysis can offer."""
    any_RH = any(p.R_H is not None for p in points)
    any_anti = any(p.antisym_points >= 1 for p in points)
    any_mu = any(p.mobility is not None for p in points)
    enough = any(p.antisym_points >= 2 and not p.low_confidence for p in points)
    caps = [
        Capability(name="hall_coefficient", applicable=any_RH,
                   reason="R_asym(B) line fits with thickness" if any_RH
                   else ("thickness required" if not has_thickness else "no fittable T point")),
        Capability(name="antisymmetrization", applicable=any_anti,
                   reason="fixed-field family spans +/-B" if any_anti else "no +/-B pairs"),
        Capability(name="carrier_concentration", applicable=any_RH,
                   reason="n=1/(e|R_H|)" if any_RH else "needs R_H"),
        Capability(name="mobility", applicable=any_mu,
                   reason=f"mu=|R_H|/rho_xx ({long_source})" if any_mu
                   else "no longitudinal channel/file"),
        Capability(name="dual_method", applicable=has_dual,
                   reason="field-sweep loops also present" if has_dual
                   else "no field sweeps in file"),
    ]
    n_two_point = sum(1 for p in points if getattr(p, "r_h_method", None) == "2point")
    caps.append(Capability(name="two_point_extended", applicable=n_two_point > 0,
                reason=(f"{n_two_point} extra T via 0-field+1 estimate" if n_two_point
                        else "no 0-field+1 fallback points added")))
    if not enough:
        caps.append(Capability(name="rich_field_recommended", applicable=True,
                    reason="no confident T point has >= 2 antisym pairs (a single-pair "
                           "R_H cannot check its own linearity); supply a Hall file "
                           "with more fixed fields"))
    return caps


# ---- analyzer class --------------------------------------------------------

class HallTempDepAnalyzer:
    probe = "hall_tdep"
    needs = (Need("hall_channel", scope="sample", required=True),
             Need("thickness_mm", scope="sample", required=False),
             Need("longitudinal_channel", scope="sample", required=False),
             Need("longitudinal_file", scope="sample", required=False))

    def analyze(self, rawtable, cfg) -> Result:
        df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
        header = rawtable.header
        hc = cfg.hall
        prov = Provenance(file=getattr(rawtable, "path", None) or header.title or "",
                          sha256=_sha256(getattr(rawtable, "path", None)),
                          app_version=header.app_version, config=cfg.model_dump(mode="json"))

        # --- guards ---
        if hc.hall_channel is None:
            return Result(status="error", errors=["hall_channel required"],
                          data={"probe": "hall_tdep"}, provenance=prov)
        rkey = f"resistance_ch{hc.hall_channel}"
        if rkey not in cmap.logical or "temperature" not in cmap.logical or "field" not in cmap.logical:
            return Result(status="error",
                          errors=[f"hall channel {hc.hall_channel} resistance / T / H not found"],
                          data={"probe": "hall_tdep"}, provenance=prov)

        thickness_m = (hc.thickness_mm * 1e-3) if hc.thickness_mm else None

        # --- longitudinal source for sigma / mobility ---
        long_df = long_cmap = None
        long_source = None
        if hc.longitudinal_file:
            lrt = load_dat(hc.longitudinal_file)
            long_df, long_cmap = canonicalize_columns(lrt.df, lrt.header)
            long_source = f"file:{pathlib.Path(hc.longitudinal_file).name}:ch{hc.longitudinal_channel}"
        elif hc.longitudinal_channel is not None:
            long_source = f"same_file:ch{hc.longitudinal_channel}"
        rho_fn = _long_rho_xx(df, cmap, hc.longitudinal_channel, long_df, long_cmap)

        # --- build fixed-field curves → reconstruct temp-dep Hall points ---
        curves = _interp_fixed_field_curves(df, cmap, cfg, hc.hall_channel, hc.temp_interval)
        # O4: instrument sigma_R curves from the file's own std column + Resistance/
        # Resistivity ratio (None -> every *_sigma_instrument field stays None)
        sd_curves = _interp_fixed_field_sigma_curves(df, cmap, cfg, hc.hall_channel,
                                                     hc.temp_interval)
        points, stages = _reconstruct_points(curves, thickness_m, hc.geometry_sign,
                                             hc.tdep_min_antisym_points, want_stages=True,
                                             two_point_fallback=hc.tdep_two_point_fallback,
                                             sd_curves=sd_curves)
        for p in points:
            _sigma_mu_J(p, rho_fn)

        # --- dual-method: field sweeps present in same file ---
        fsegs = [s for s in segment_sweeps(df, cmap, cfg) if s.swept.name == "field"]
        dual = []
        if fsegs and thickness_m is not None:
            fs_pts = field_sweep_points(df, cmap, cfg, hc, thickness_m, rho_fn)
            fs_by_T = {round(p.temperature, 1): p.R_H for p in fs_pts if p.R_H is not None}
            for p in points:
                rf = fs_by_T.get(round(p.temperature, 1))
                if p.R_H is not None or rf is not None:
                    vals = [v for v in (p.R_H, rf) if v is not None]
                    comb = float(np.mean(vals)) if vals else None
                    n_c, _ = _carrier_n(comb)
                    dual.append(DualMethodPoint(
                        temperature=p.temperature,
                        R_H_tempdep=p.R_H,
                        R_H_fieldsweep=rf,
                        R_H_combined=comb,
                        n_combined=n_c,
                    ))

        has_dual = len(dual) > 0
        caps = _capabilities(points, thickness_m is not None, long_source, has_dual,
                             hc.tdep_min_antisym_points)
        interp = [InterpCurve(field_oe=float(f), temperature=Tg.tolist(), R=Rg.tolist())
                  for f, (Tg, Rg) in sorted(curves.items())]
        data = HallTempDepData(
            probe="hall_tdep",
            hall_channel=hc.hall_channel,
            thickness_m=thickness_m,
            geometry_sign=hc.geometry_sign,
            temp_interval=hc.temp_interval,
            longitudinal_source=long_source,
            points=points,
            stages=stages,
            interp_curves=interp,
            dual_method=dual,
            capabilities=caps,
        )

        # thickness omitted -> R_H is unscaled (all None); report the true cause, not "no fit"
        if thickness_m is None:
            return Result(status="low_confidence", confidence=0.4,
                          warnings=["thickness required for R_H (slope-only reconstruction)"],
                          data=data.model_dump(mode="json"), provenance=prov)

        fitted = [p for p in points if p.R_H is not None]
        if not fitted:
            return Result(status="low_confidence", confidence=0.2,
                          warnings=["no fittable T point (need >=2 antisym points)"],
                          data=data.model_dump(mode="json"), provenance=prov)

        # D8: confidence = fraction of non-low_confidence fitted points; NEVER mean r².
        # Basis = the TRUSTED antisym points only. The 2-point fallback (B) EXTENDS coverage
        # with honestly-flagged low_confidence tail points; counting them in the denominator
        # would let extra coverage deflate status (backwards). Antisym-only frac keeps
        # "antisym_fraction" literally accurate. If there are no antisym points at all
        # (2-point coverage only), the result is genuinely low_confidence (frac 0).
        # #18 (2026-09-02): single-pair points are labelled "antisym" (they are one) and
        # so now count in this basis — the fraction covers the points actually fitted.
        antisym_fitted = [p for p in fitted if p.r_h_method != "2point"]
        frac = (sum(1 for p in antisym_fitted if not p.low_confidence) / len(antisym_fitted)
                if antisym_fitted else 0.0)
        conf = float(frac)
        status = "ok" if frac >= 0.5 else "low_confidence"
        # Closed O4 + hardening 2: honest aggregate warning when the instrument sigma says
        # the R_H(T) points are noise-dominated (> 50 % relative). EXPECTED to fire on the
        # real Hall file's channel (nV-level signal, median std/rho 61 %) — flag, never drop.
        warns: list[str] = []
        rels = [p.r_h_sigma_instrument / abs(p.R_H) for p in fitted
                if p.r_h_sigma_instrument is not None and p.R_H]
        noisy = [x for x in rels if x > 0.5]
        if noisy:
            warns.append(
                f"{len(noisy)}/{len(rels)} R_H(T) points carry > 50% relative instrument "
                f"sigma (median {float(np.median(rels)) * 100:.0f}%) — instrument noise, "
                f"not fit quality; treat these R_H as noise, not a carrier density")
        return Result(
            status=status,
            confidence=conf,
            warnings=warns,
            confidence_parts={
                "detector": 1.0,
                "segmentation": 1.0,
                "antisym_fraction": float(frac),
            },
            data=data.model_dump(mode="json"),
            provenance=prov,
        )
