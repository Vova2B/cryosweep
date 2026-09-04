"""Thermal Transport Option (TTO) analyzer. Mirrors the ACMS analyzer's shape.

ANALYZER ISOLATION (project convention): `Capability`, `_cluster_1d` and `_DIR` are copied
module-locally rather than imported from another analyzer. Do NOT "de-duplicate" them.
"""
from __future__ import annotations

import hashlib
import math
import pathlib

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from cryosweep_core.detect.vsm_blocks import ramps_from_temps
from cryosweep_core.fitting.thermal import fit_kappa_ph_ladder
from cryosweep_core.fitting.uncertainty import (
    MEDIAN_SE as _MEDIAN_SE,
    endpoint_sigma as _shared_endpoint_sigma,
    rrr_sigma as _shared_rrr_sigma,
    straddles_threshold as _shared_straddles_threshold,
)
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.result import Gate, Provenance, Result

_MIN_PTS = 5                 # field groups smaller than this are dropped + logged
_RAMP_MIN_LEN = 15           # same ramp-segmentation floor the ACMS analyzer pins
_DIR = {"warming": "up", "cooling": "down"}
_FIELD_ABS_TOL_OE = 5.0      # field clustering: abs 5 Oe / rel 1% (spec §2 step 3)
_FIELD_REL_TOL = 0.01
_L0 = 2.443e-8               # Sommerfeld Lorenz number, W*Ohm*K^-2
_RRR_K = 5                   # nearest-extreme physical points to median for RRR endpoints
_ZERO_FIELD_OE = 50.0        # |H| below this counts as held "zero field"
_KAPPA_PH_PRIMARY_K = 10.0   # primary kappa_ph fit window: highest r2 AND a low-T asymptotic
_KAPPA_PH_MIN_PTS = 10       # same floor the fitting module enforces on the primary rung
# _MEDIAN_SE now lives in cryosweep_core/fitting/uncertainty.py (U6 extraction) — imported above.
_DT_OVER_T_FRAC = 0.05       # |Delta T| / T above this is warned about
_S_OSC_MIN_COUNT = 5         # reversals needed before the low-T sign structure is called out
_S_OSC_MAX_T_K = 20.0        # only reversals below this T are counted (NEVER 12 K)
_S_OSC_MAX_WINDOW_K = 5.0    # ... and only when they are packed into a window this narrow
_CLASS_HI, _CLASS_LO = 1.02, 0.98    # the same thresholds _classify uses
# Flags that make the reported exponent NOT a measurement, so the fit is DECLINED rather than
# reported (final-review C1). `n_at_bound` means the optimizer parked on the edge of the search
# space ([0.5, 6.0]) — the returned n is the bound, not what the data says; `degenerate_window`
# means the window holds a single distinct T and nothing was fitted (n comes back as p0).
# `kappa_e_dominant` and `window_sensitive` are NOT here: those describe a real fit that the
# reader must interpret carefully, and they are surfaced as words on every surface instead.
_FIT_FATAL_FLAGS = ("n_at_bound", "degenerate_window")


def _cluster_1d(values, rel_tol, abs_tol: float = 0.0):
    """Cluster a 1-D value array by proximity. Sort unique-order the values; start a new
    cluster whenever the gap to the previous value exceeds max(abs_tol, rel_tol*|value|).
    Pure numpy, deterministic. Returns (labels_per_row, representatives) where
    representatives[label] = median of that cluster's member values. (Copied verbatim from
    acms.py per the analyzer-isolation convention — do not import it across analyzers.)"""
    v = np.asarray(values, float)
    order = np.argsort(v, kind="stable")           # stable -> deterministic
    sv = v[order]
    csorted = np.zeros(sv.size, dtype=int)
    c = 0
    for i in range(1, sv.size):
        gap = sv[i] - sv[i - 1]
        tol = max(abs_tol, rel_tol * abs(sv[i]))
        if gap > tol:
            c += 1
        csorted[i] = c
    labels = np.empty(v.size, dtype=int)
    labels[order] = csorted
    reps = np.array([float(np.median(v[labels == k])) for k in range(c + 1)], float)
    return labels, reps


class Capability(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    applicable: bool
    reason: str = ""


class TTOCurve(BaseModel):
    model_config = ConfigDict(extra="ignore")
    field_oe: float
    direction: str                              # "up" | "down" | "mixed"
    n_points: int = 0
    t: list[float] = []                         # K, finite, ascending
    kappa: list[float] = []                     # W/(K*m), finite > 0
    kappa_std: list[float | None] | None = None
    seebeck: list[float | None] | None = None   # microvolt per kelvin
    seebeck_std: list[float | None] | None = None
    rho: list[float | None] | None = None       # Ohm*m (from rho_tto)
    rho_std: list[float | None] | None = None
    zt: list[float | None] | None = None
    zt_std: list[float | None] | None = None
    kappa_e: list[float | None] | None = None       # L0*T/rho (D7 validity)
    kappa_ph: list[float | None] | None = None      # kappa - kappa_e (may be negative)
    lorenz_ratio: list[float | None] | None = None  # (kappa*rho)/(L0*T), dimensionless L/L0
    power_factor: list[float | None] | None = None  # W/(K^2*m) = (S*1e-6)^2/rho
    # --- 2026-08-10 uncertainty-honesty additive fields (spec §5, closed O6; declared LAST:
    # append-only JSON key order). Element-wise, None where any input is None/non-finite:
    #   kappa_e_std      = kappa_e * (rho_std/rho)          (kappa_e = L0*T/rho — exact
    #                                                        single-variable propagation)
    #   kappa_ph_std     = sqrt(kappa_std^2 + kappa_e_std^2) (kappa and rho treated
    #                      UNCORRELATED — the instrument measures them in one pass, so this
    #                      is an upper-bound assumption, recorded not hidden)
    #   lorenz_ratio_std = lorenz_ratio * sqrt((kappa_std/kappa)^2 + (rho_std/rho)^2) ---
    kappa_e_std: list[float | None] | None = None
    kappa_ph_std: list[float | None] | None = None
    lorenz_ratio_std: list[float | None] | None = None


class RRRBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rrr: float
    t_high_k: float
    t_low_k: float
    classification: str            # metallic | insulating | non_monotonic | unknown
    rrr_std: float | None = None   # None, NEVER NaN (bare float, `_san` does not walk it)


class KappaPhFit(BaseModel):
    """Phonon power law kappa_ph = B*T^n on the primary (<=10 K) window, WITH the two
    sensitivities that dominate the statistical sigma on real data (spec §1)."""
    model_config = ConfigDict(extra="ignore")
    n: float
    n_sigma: float                 # statistical only -- NOT the honest error bar
    n_spread: float | None         # max-min across the window ladder; the DOMINANT uncertainty
                                   # None (never 0.0) when < 2 curve_fit rungs fitted (I1)
    n_loglog: float | None         # same primary window, log-log OLS (method sensitivity)
    n_method_delta: float | None   # abs(n - n_loglog)
    b: float
    b_sigma: float
    r2: float
    n_points: int
    window_k: list[float]          # [T_min, primary cutoff] actually fitted
    ladder: list[dict]             # keys cutoff_k/method/n/sigma/r2/n_points, loglog LAST
    quality_flags: list[str]


class TTOSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    pf_at_thigh: float | None = None    # PF median over the 5 valid points nearest max T
    zt_peak: float | None = None
    zt_peak_t_k: float | None = None
    # True when the reported maximum sits at an end of the measured T range (no interior
    # maximum was observed — the sweep stopped). None when there is no ZT at all.
    zt_peak_at_edge: bool | None = None
    zt_peak_std: float | None = None    # the zt_std at the peak row (I4)


class TTOData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    probe: str = "tto"
    sample: dict = {}
    curves: list[TTOCurve] = []
    dropped_groups: list[dict] = []
    rrr: RRRBlock | None = None
    summary: TTOSummary = TTOSummary()
    kappa_ph_fit: KappaPhFit | None = None
    n_error_rows: int = 0
    capabilities: list[Capability] = []


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""


def _num(df, cmap, key):
    """Numeric numpy array for a logical column, or None when the column is absent."""
    col = cmap.logical.get(key)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def _san(values):
    """D11 finiteness sanitiser + emission rule. Maps None/NaN/inf -> None and everything
    else to float; returns None when EVERY entry is None (an all-missing optional array is
    emitted as absent, not as a list of nulls). Pydantic does not map NaN -> None for
    `float | None`, and json.dumps would then emit the invalid token NaN."""
    if values is None:
        return None
    out = []
    for v in values:
        if v is None:
            out.append(None)
            continue
        f = float(v)
        out.append(None if not math.isfinite(f) else f)
    return None if all(v is None for v in out) else out


def _as_float_array(values, n):
    """A length-n float array from a raw numpy array or a list that may hold None.
    Missing/absent -> all-NaN, so the D7 validity masks below simply fail everywhere."""
    if values is None:
        return np.full(n, np.nan)
    return np.array([np.nan if v is None else float(v) for v in values], float)


def _derive_wf(t, kappa, rho):
    """Wiedemann-Franz decomposition + Lorenz ratio (spec §3).

    kappa_e = L0*T/rho ; kappa_ph = kappa - kappa_e ; L/L0 = kappa*rho/(L0*T).
    D7 validity: rho finite AND rho > 0 AND T > 0; invalid points come back NaN (the caller's
    _san turns them into JSON null). kappa_ph MAY be negative when L/L0 < 1 (inelastic
    scattering) — reported as computed, NEVER clipped."""
    T = _as_float_array(t, len(t))
    K = _as_float_array(kappa, len(t))
    R = _as_float_array(rho, len(t))
    ok = np.isfinite(R) & (R > 0) & np.isfinite(T) & (T > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa_e = np.where(ok, _L0 * T / R, np.nan)
        kappa_ph = np.where(ok, K - kappa_e, np.nan)
        lorenz = np.where(ok, K * R / (_L0 * T), np.nan)
    return kappa_e, kappa_ph, lorenz


def _derive_wf_std(cur):
    """The three derived-quantity _std companions (spec §5 formulas — see TTOCurve field
    comment). NaN where any input is missing/non-finite; the caller's _san maps NaN->None."""
    n = len(cur.t)
    K = _as_float_array(cur.kappa, n); Ks = _as_float_array(cur.kappa_std, n)
    R = _as_float_array(cur.rho, n); Rs = _as_float_array(cur.rho_std, n)
    Ke = _as_float_array(cur.kappa_e, n); L = _as_float_array(cur.lorenz_ratio, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        ok_e = np.isfinite(Ke) & np.isfinite(Rs) & np.isfinite(R) & (R > 0)
        ke_std = np.where(ok_e, Ke * (Rs / R), np.nan)
        kph_std = np.where(np.isfinite(Ks) & np.isfinite(ke_std),
                           np.sqrt(Ks ** 2 + ke_std ** 2), np.nan)
        ok_l = (np.isfinite(L) & np.isfinite(Ks) & np.isfinite(K) & (K > 0)
                & np.isfinite(Rs) & np.isfinite(R) & (R > 0))
        l_std = np.where(ok_l, L * np.sqrt((Ks / K) ** 2 + (Rs / R) ** 2), np.nan)
    return ke_std, kph_std, l_std


def _derive_pf(seebeck, rho, n):
    """Power factor PF = S^2/rho in W/(K^2*m); S arrives in microvolt per kelvin, so it is
    converted to volts first. D7 validity: rho finite AND rho > 0 AND S finite."""
    S = _as_float_array(seebeck, n)
    R = _as_float_array(rho, n)
    ok = np.isfinite(R) & (R > 0) & np.isfinite(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(ok, (S * 1e-6) ** 2 / R, np.nan)


def _endpoint(T, rho, lowest: bool, k: int = _RRR_K):
    """Median rho over the k physical points nearest the min (lowest) / max T.

    D10 DOCUMENTED DIVERGENCE from resistivity.py's _endpoint: the mask here is
    `isfinite & > 0` only. resistivity.py additionally applies a `rho >= median*1e4` sentinel
    guard and an optional MAD-outlier exclusion, because raw bridge files carry sentinel rows.
    TTO rho is instrument-computed and sentinel-free, so those guards would only add
    unexplained behaviour. A cross-probe test pins that both agree on sentinel-free input."""
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    m = np.isfinite(rho) & (rho > 0) & np.isfinite(T)
    T, rho = T[m], rho[m]
    if T.size == 0:
        return float("nan"), float("nan")
    kk = min(k, T.size)
    order = np.argsort(T, kind="stable")           # stable -> deterministic on duplicate T
    sel = order[:kk] if lowest else order[-kk:]
    return float(np.median(T[sel])), float(np.median(rho[sel]))


def _rrr(T, rho, k: int = _RRR_K):
    """RRR = rho(T_high)/rho(T_low). Returns (rrr, t_high, t_low) or (None, None, None)."""
    t_lo, r_lo = _endpoint(T, rho, True, k)
    t_hi, r_hi = _endpoint(T, rho, False, k)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo <= 0:
        return None, None, None
    val = float(r_hi / r_lo)
    if not np.isfinite(val):        # subnormal r_lo can overflow the quotient to inf; a
        return None, None, None     # non-finite RRR would break json.dumps(allow_nan=False)
    return val, float(t_hi), float(t_lo)


def _classify(T, rho) -> str:
    """metallic if rho rises with T (robust endpoints), insulating if it falls."""
    _, r_lo = _endpoint(T, rho, True, _RRR_K)
    _, r_hi = _endpoint(T, rho, False, _RRR_K)
    if not (np.isfinite(r_lo) and np.isfinite(r_hi)) or r_lo <= 0:
        return "unknown"
    ratio = r_hi / r_lo
    if ratio > 1.02:
        return "metallic"
    if ratio < 0.98:
        return "insulating"
    return "non_monotonic"


def _rrr_curve(curves):
    """The RRR-selection curve: the zero-field (|H| < 50 Oe) curve with the widest T span.
    Ties resolve to the first by curve order (max() keeps the first maximum)."""
    zf = [c for c in curves if abs(c.field_oe) < _ZERO_FIELD_OE]
    if not zf:
        return None
    return max(zf, key=lambda c: float(np.ptp(np.asarray(c.t, float))) if c.t else 0.0)


def _kappa_ph_fit_curve(curves, primary: float = _KAPPA_PH_PRIMARY_K):
    """The kappa_ph-fit curve: the one with the MOST finite kappa_ph > 0 points at T <= primary.

    I7: deliberately NOT `_rrr_curve`. That helper selects on T span among |H| < 50 Oe curves
    with no reference to kappa_ph, so the widest zero-field curve may be the one WITHOUT rho
    (kappa_ph all None -> the fit declines blaming point count), and a field-only file would
    get no fit at all even though kappa_ph is well defined per curve. Ties break towards a
    zero-field curve, then by curve order. None when no curve clears the 10-point floor."""
    best, best_key = None, None
    for i, c in enumerate(curves):
        # `_as_float_array` honours `n` only for a MISSING array, so a kappa_ph of the wrong
        # length would reach the mask below and raise a bare broadcast ValueError OUTSIDE the
        # caller's try (analyze() would die). Unreachable through this analyzer's own
        # construction (every per-point array is built from the same index slice) — structural
        # safety only, so a future caller cannot turn a length bug into a crash.
        if c.kappa_ph is not None and len(c.kappa_ph) != len(c.t):
            continue
        T = np.asarray(c.t, float)
        kph = _as_float_array(c.kappa_ph, len(c.t))
        n = int(np.count_nonzero(np.isfinite(T) & np.isfinite(kph) & (kph > 0)
                                 & (T > 0) & (T <= primary)))
        if n < _KAPPA_PH_MIN_PTS:
            continue
        key = (-n, 0 if abs(c.field_oe) < _ZERO_FIELD_OE else 1, i)
        if best_key is None or key < best_key:
            best, best_key = c, key
    return best


def _fit_decline_reason(bad, fr) -> str:
    """The capability reason for a kappa_ph fit DECLINED on integrity grounds (C1).

    Names the specific defect and carries the two numbers that prove it, so the capabilities
    CSV / GUI hint strip states a fact about THIS file rather than a bare "declined". Written
    plainly: this string is the only thing the reader gets in place of an exponent."""
    n_val, r2 = float(fr.params["n"]), float(fr.r2)
    why = {
        "n_at_bound": f"the exponent pinned at the search bound (n = {n_val:.3g})",
        "degenerate_window": "the fit window holds a single distinct T, so nothing was fitted",
        "worse_than_a_constant": f"the power law fits worse than a constant (r2 = {r2:.3g})",
    }
    return ("kappa_ph is not a power law below "
            f"{_KAPPA_PH_PRIMARY_K:g} K — " + "; ".join(why[b] for b in bad)
            + " — no exponent is reported")


def _valid_count(arr) -> int:
    if not arr:
        return 0
    return sum(1 for v in arr if v is not None and math.isfinite(v))


def _any_curve_has(curves, attr, n: int = 5) -> bool:
    return any(_valid_count(getattr(c, attr)) >= n for c in curves)


def _pf_at_thigh(curve):
    """Median power factor over the 5 VALID points nearest max T on the given curve."""
    if curve is None or not curve.power_factor:
        return None
    T = np.asarray(curve.t, float)
    pf = _as_float_array(curve.power_factor, len(curve.t))
    m = np.isfinite(pf) & np.isfinite(T)
    if not m.any():
        return None
    Tm, pfm = T[m], pf[m]
    k = min(_RRR_K, Tm.size)
    sel = np.argsort(Tm)[-k:]
    return float(np.median(pfm[sel]))


def _zt_peak(curves):
    """(max valid ZT across all curves, its T, is-at-a-T-range-edge, its zt_std) or
    (None, None, None, None).

    HONESTY FLAG (`at_edge`): on the real gate file the maximum ZT sits exactly at T_max
    (301.370002 K) with ZT still rising — the measurement simply stopped, no interior
    extremum was observed. Calling that a "peak" asserts more than the data supports (the
    same class of over-claim as PQ-4's fabricated Tc on clean metals), so the boundary case
    is flagged and surfaced in the summary table, the CSV and the GUI.

    I4: `std` is tracked in THIS loop rather than recovered afterwards — the winning row index
    is not recoverable from the value, because ties keep the FIRST maximum. None when zt_std
    is absent, shorter than the winning index, or non-finite there."""
    best_v = best_t = best_c = best_std = None
    for c in curves:
        if not c.zt:
            continue
        std = c.zt_std or []
        for i, (tv, zv) in enumerate(zip(c.t, c.zt)):
            if zv is None or not math.isfinite(zv):
                continue
            if best_v is None or zv > best_v:
                best_v, best_t, best_c = float(zv), float(tv), c
                sv = std[i] if i < len(std) else None
                best_std = (float(sv) if sv is not None and math.isfinite(float(sv))
                            else None)
    if best_v is None:
        return None, None, None, None
    # curve `t` is T-ascending, so the extremes are the first and last entries
    at_edge = bool(best_c.t and (best_t == float(best_c.t[0])
                                 or best_t == float(best_c.t[-1])))
    return best_v, best_t, at_edge, best_std


def _error_code_column(df):
    for c in df.columns:
        if str(c).strip().lower() == "error (code)":
            return c
    return None


def _delta_temp_column(df):
    """The RAW `Delta Temp. (K)` column (it is NOT canonicalized), matched case-insensitively
    on the stripped name — the same rule `_error_code_column` uses."""
    for c in df.columns:
        if str(c).strip().lower() == "delta temp. (k)":
            return c
    return None


def _delta_t_warning(delta_t, temp):
    """Warn when any row's |ΔT|/T exceeds 5 %: kappa there is averaged over a wide T window.

    ABSOLUTE value — the ΔT sign is a wiring convention, not information (on the gate file
    every ΔT is positive, 0.0887-5.2496 K, so the file cannot test this). An absent column and
    a present-but-entirely-non-finite column behave identically: None, no error (M5)."""
    dt = np.asarray(delta_t, float)
    T = np.asarray(temp, float)
    m = np.isfinite(dt) & np.isfinite(T) & (T > 0)
    if not m.any():
        return None
    ratio = np.abs(dt[m]) / T[m]
    n = int(np.count_nonzero(ratio > _DT_OVER_T_FRAC))
    if n == 0:
        return None
    i = int(np.argmax(ratio))
    return (f"{n} rows have ΔT/T > {_DT_OVER_T_FRAC * 100:.0f}% "
            f"(max {ratio[i] * 100:.2f}% at {T[m][i]:.3f} K) — "
            f"kappa there is averaged over a wide T window")


def _seebeck_oscillation_warning(curves):
    """One warning per curve whose S reverses sign >= 5 times below 20 K inside a < 5 K window.

    The trigger is DENSITY, not count: five reversals packed into under 5 K is not a physical
    S(T) shape at the sampled resolution, while a curve that crosses zero once or twice, or
    oscillates slowly across a wide range, stays silent. The warning reports an OBSERVATION and
    makes NO claim about noise: `seebeck_std` is deliberately unused, because it is the
    instrument's repeat-scatter on one measurement, not the point-to-point scatter of S — on
    the gate file every bracketing point is 11.4-45.5 sigma from zero, so these crossings are
    real structure (C1)."""
    out = []
    for c in curves:
        if not c.seebeck:
            continue
        T = np.asarray(c.t, float)
        S = _as_float_array(c.seebeck, len(c.t))
        m = np.isfinite(T) & np.isfinite(S) & (T < _S_OSC_MAX_T_K)
        Tm, Sm = T[m], S[m]
        if Tm.size < 2:
            continue
        flips = [i for i in range(Tm.size - 1) if Sm[i] * Sm[i + 1] < 0]
        if len(flips) < _S_OSC_MIN_COUNT:
            continue
        t_lo = float(Tm[flips[0]])
        t_hi = float(Tm[flips[-1] + 1])
        width = abs(t_hi - t_lo)
        if width >= _S_OSC_MAX_WINDOW_K:
            continue
        lo, hi = min(t_lo, t_hi), max(t_lo, t_hi)
        out.append(f"S changes sign {len(flips)} times between {lo:.3f} K and {hi:.3f} K "
                   f"(a {width:.3f} K window) — the low-T sign structure oscillates "
                   f"from point to point")
    return out


def _endpoint_sigma(T, rho, rho_std, lowest: bool, k: int = _RRR_K):
    """Thin wrapper over fitting.uncertainty.endpoint_sigma (U6 extraction, 2026-08-10):
    numerics byte-identical, tto call sites and tests untouched."""
    return _shared_endpoint_sigma(T, rho, rho_std, lowest, k)


def _rrr_sigma(T, rho, rho_std, rrr, k: int = _RRR_K):
    """Thin wrapper over fitting.uncertainty.rrr_sigma, passing tto's own `_endpoint`
    (the "mask must match" law stays enforced per-caller)."""
    return _shared_rrr_sigma(T, rho, rho_std, rrr, k, _endpoint)


def _straddles_threshold(rrr, std) -> bool:
    """True when the +-1 sigma band spans a metal/insulator classification threshold."""
    return _shared_straddles_threshold(rrr, std, _CLASS_LO, _CLASS_HI)


def _sample_block(header):
    """name + the header's geometry INFO passthrough, as floats when parseable (D4: these are
    why no user inputs are needed — kappa/rho are already absolute)."""
    info = getattr(header, "info", None) or {}

    def f(key):
        try:
            return float(info[key])
        except (KeyError, TypeError, ValueError):
            return None

    return {"name": getattr(header, "title", None),
            "material": info.get("SAMPLE_MATERIAL"),
            "vlead_separation": f("SAMPLE_VLEAD_SEPARATION"),
            "ilead_separation": f("SAMPLE_ILEAD_SEPARATION"),
            "cross_section": f("SAMPLE_CROSS_SECTION"),
            "emissivity": f("SAMPLE_EMISSIVITY")}


def _gated(reason_need, reason, prov):
    return Result(status="gated", confidence=0.0, data={"probe": "tto"},
                  gate=[Gate(need=reason_need, reason=reason)], provenance=prov)


class TTOAnalyzer:
    probe = "tto"
    needs = ()   # D4: sample geometry rides in the file header -> no user inputs at all

    def analyze(self, rawtable, cfg) -> Result:
        df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
        header = rawtable.header
        prov = Provenance(file=getattr(rawtable, "path", None) or header.title or "",
                          sha256=_sha256(getattr(rawtable, "path", None)),
                          app_version=header.app_version, config=cfg.model_dump(mode="json"))
        # ---- required columns -> gated on absence (never ok+empty) ----
        if "kappa" not in cmap.logical:
            return _gated("kappa", "TTO file missing a 'Conductivity (W/K-m)' column", prov)
        temp = _num(df, cmap, "temperature")
        if temp is None:
            return _gated("temperature", "TTO file missing a temperature column", prov)
        kappa = _num(df, cmap, "kappa")
        field = _num(df, cmap, "field")
        if field is None:
            field = np.zeros(temp.size, float)          # field optional -> assume 0
        raw = {k: _num(df, cmap, src) for k, src in
               (("kappa_std", "kappa_std"), ("seebeck", "seebeck"),
                ("seebeck_std", "seebeck_std"), ("rho", "rho_tto"),
                ("rho_std", "rho_tto_std"), ("zt", "zt"), ("zt_std", "zt_std"))}

        # ---- 1. row filter (D6): T finite AND kappa finite AND kappa > 0 ----
        # `field` is guarded too even though D6 names only T/kappa: it is the SOLE grouping
        # key, and a NaN field sorts last in `_cluster_1d`'s stable argsort where the gap test
        # `gap > tol` is False for a NaN gap — so the row is absorbed into the last cluster and
        # that cluster's median representative becomes NaN. A single blank
        # `Magnetic Field (Oe)` cell (the QD trailing-comma column shift, c539fdb, produces
        # exactly these) then fragments a real field group, strips its field identity, silently
        # removes it from the |H| < 50 Oe RRR selection, and puts a bare non-finite `float` in
        # `TTOCurve.field_oe` — which the D11 sanitiser never sees (it walks per-point arrays
        # only), breaking `json.dumps(data, allow_nan=False)`. Mirrors `ACMSAnalyzer`, which
        # guards its own grouping keys the same way (acms.py:305-306).
        keep = np.isfinite(temp) & np.isfinite(kappa) & (kappa > 0) & np.isfinite(field)
        n_dropped = int((~keep).sum())

        # ---- 2. instrument error codes (D5): counted, warned, NEVER dropped ----
        n_error_rows = 0
        ecol = _error_code_column(df)
        if ecol is not None:
            codes = pd.to_numeric(df[ecol], errors="coerce").to_numpy(float)[keep]
            n_error_rows = int(np.sum(np.isfinite(codes) & (codes != 0)))

        # ---- 2b. thermal-gradient integrity (spec §2.1). Read on the D6 `keep` rows, BEFORE
        # grouping, so it stays positionally aligned to the filtered temperature array (M4).
        dt_warning = None
        dcol = _delta_temp_column(df)
        if dcol is not None:
            dt_warning = _delta_t_warning(
                pd.to_numeric(df[dcol], errors="coerce").to_numpy(float)[keep], temp[keep])

        temp, kappa, field = temp[keep], kappa[keep], field[keep]
        raw = {k: (v[keep] if v is not None else None) for k, v in raw.items()}

        curves: list[TTOCurve] = []
        dropped: list[dict] = []
        if temp.size:
            # ---- 3. group by field only ----
            labels, reps = _cluster_1d(field, _FIELD_REL_TOL, abs_tol=_FIELD_ABS_TOL_OE)
            buckets: dict[int, list[int]] = {}
            for i in range(field.size):
                buckets.setdefault(int(labels[i]), []).append(i)
            # Belt-and-braces: a non-finite representative can never reach `TTOCurve.field_oe`
            # (which is a bare `float` the D11 sanitiser does not walk). The `isfinite(field)`
            # row guard above already makes this unreachable; it stays as a second line of
            # defence for any future path that reintroduces a non-finite key.
            groups = [{"field_oe": float(reps[k]), "idx": np.asarray(sorted(v), int)}
                      for k, v in buckets.items() if math.isfinite(float(reps[k]))]
            groups.sort(key=lambda g: g["field_oe"])
            for g in groups:
                idx = g["idx"]
                if idx.size < _MIN_PTS:
                    dropped.append({"field_oe": g["field_oe"], "n_points": int(idx.size),
                                    "reason": f"< {_MIN_PTS} points"})
                    continue
                # ---- 4. ramp split. i1 is INCLUSIVE -> +1 (acms.py:329). The "mixed"
                # fallback is spec-mandated defensive depth: ramps_from_temps returns []
                # only for n == 0 today, which the _MIN_PTS filter already excludes.
                ramps = ramps_from_temps(temp[idx].tolist(), min_len=_RAMP_MIN_LEN)
                spans = ([(r["direction"], r["i0"], r["i1"] + 1) for r in ramps]
                         if ramps else [("mixed", 0, idx.size)])
                for direction, a0, a1 in spans:
                    sub = idx[a0:a1]
                    # _MIN_PTS guards the GROUP above; this guards the EMITTED CURVE, so a
                    # stub ramp segment can never become a 2-point "curve". No-op on every
                    # path today (ramps carry min_len=15 and the mixed fallback spans a group
                    # that already passed _MIN_PTS) — defensive depth only.
                    if sub.size < _MIN_PTS:
                        dropped.append({"field_oe": g["field_oe"], "n_points": int(sub.size),
                                        "reason": f"ramp segment < {_MIN_PTS} points"})
                        continue
                    # ONE stable argsort by T applied identically to every parallel array,
                    # so rows stay aligned. `direction` is captured BEFORE the sort — it is
                    # not recoverable from the sorted arrays.
                    sub = sub[np.argsort(temp[sub], kind="stable")]
                    cur = TTOCurve(
                        field_oe=g["field_oe"], direction=_DIR.get(direction, direction),
                        n_points=int(sub.size), t=temp[sub].tolist(),
                        kappa=kappa[sub].tolist(),
                        kappa_std=_san(raw["kappa_std"][sub]) if raw["kappa_std"] is not None else None,
                        seebeck=_san(raw["seebeck"][sub]) if raw["seebeck"] is not None else None,
                        seebeck_std=_san(raw["seebeck_std"][sub]) if raw["seebeck_std"] is not None else None,
                        rho=_san(raw["rho"][sub]) if raw["rho"] is not None else None,
                        rho_std=_san(raw["rho_std"][sub]) if raw["rho_std"] is not None else None,
                        zt=_san(raw["zt"][sub]) if raw["zt"] is not None else None,
                        zt_std=_san(raw["zt_std"][sub]) if raw["zt_std"] is not None else None)
                    # ---- 5. per-point derived (D7 validity, D11 sanitiser, emission rule)
                    kappa_e, kappa_ph, lorenz = _derive_wf(cur.t, cur.kappa, cur.rho)
                    cur.kappa_e = _san(kappa_e)
                    cur.kappa_ph = _san(kappa_ph)
                    cur.lorenz_ratio = _san(lorenz)
                    cur.power_factor = _san(_derive_pf(cur.seebeck, cur.rho, len(cur.t)))
                    ke_std, kph_std, l_std = _derive_wf_std(cur)
                    cur.kappa_e_std = _san(ke_std)
                    cur.kappa_ph_std = _san(kph_std)
                    cur.lorenz_ratio_std = _san(l_std)
                    curves.append(cur)

        # ---- degenerate: no usable curves -> gated (never ok+empty) ----
        if not curves:
            return _gated("tto_data",
                          "no usable thermal-transport data (all rows non-finite or "
                          "kappa <= 0, or every field group dropped)", prov)

        warnings: list[str] = []
        if n_dropped:
            warnings.append(
                f"{n_dropped} rows dropped (non-finite T/kappa/field or kappa <= 0)")
        if n_error_rows:
            warnings.append(f"{n_error_rows} rows carry instrument error codes (kept)")
        if dt_warning:
            warnings.append(dt_warning)
        warnings.extend(_seebeck_oscillation_warning(curves))

        # ---- 6. RRR + classification on the RRR-selection curve ----
        sel = _rrr_curve(curves)
        rrr_block = None
        if sel is not None and sel.rho:
            rho_arr = _as_float_array(sel.rho, len(sel.t))
            value, t_hi, t_lo = _rrr(sel.t, rho_arr)
            if value is not None:
                classification = _classify(sel.t, rho_arr)
                rrr_std = None
                if sel.rho_std is not None:
                    rrr_std = _rrr_sigma(sel.t, rho_arr,
                                         _as_float_array(sel.rho_std, len(sel.t)), value)
                if rrr_std is not None and _straddles_threshold(value, rrr_std):
                    # M7: "unknown" is overloaded (_classify returns it for invalid endpoints
                    # too), so THIS warning is the disambiguator and is emitted in exactly the
                    # band-straddles case.
                    classification = "unknown"
                    warnings.append(
                        f"classification_uncertain: RRR = {value:.4g} ± {rrr_std:.2g} "
                        f"straddles a metal/insulator threshold "
                        f"({_CLASS_HI} / {_CLASS_LO})")
                rrr_block = RRRBlock(rrr=value, t_high_k=t_hi, t_low_k=t_lo,
                                     classification=classification, rrr_std=rrr_std)

        # ---- 7. summary ----
        pf_curve = sel if sel is not None else curves[0]
        zt_peak, zt_peak_t, zt_edge, zt_std = _zt_peak(curves)
        summary = TTOSummary(pf_at_thigh=_pf_at_thigh(pf_curve),
                             zt_peak=zt_peak, zt_peak_t_k=zt_peak_t,
                             zt_peak_at_edge=zt_edge, zt_peak_std=zt_std)

        # ---- 7b. kappa_ph power-law fit (spec §1). A fit failure NEVER breaks analyze().
        kappa_ph_fit = None
        # Two DIFFERENT declines, two different reasons: no curve clears the point floor, or a
        # curve was selected and the fit itself declined. Reporting the point-count reason for
        # the second is a false statement about the file (this slice exists to stop those).
        fit_decline = "needs >=10 finite kappa_ph > 0 points below 10 K"
        fit_curve = _kappa_ph_fit_curve(curves)
        if fit_curve is not None:
            fit_decline = ("kappa_ph power-law fit declined on the selected curve "
                           "(no finite B*T^n solution)")
            n_pts = len(fit_curve.t)
            try:
                fr, ladder = fit_kappa_ph_ladder(
                    np.asarray(fit_curve.t, float),
                    _as_float_array(fit_curve.kappa_ph, n_pts),
                    kappa_e=_as_float_array(fit_curve.kappa_e, n_pts),
                    primary=_KAPPA_PH_PRIMARY_K)
            except (ValueError, RuntimeError):
                fr = None
            if fr is not None:
                cf = [e["n"] for e in ladder if e["method"] == "curve_fit"]
                spread = float(max(cf) - min(cf)) if len(cf) >= 2 else None
                ll = [e["n"] for e in ladder if e["method"] == "loglog"]
                n_loglog = float(ll[0]) if ll else None
                scalars = (fr.params["n"], fr.sigma["n"], fr.params["B"], fr.sigma["B"], fr.r2)
                # These are BARE floats the D11 _san sanitiser never walks (tto.py:122-136).
                # A non-finite one would break the standing json.dumps(allow_nan=False) gate,
                # so a non-finite fit is treated as no fit at all.
                if all(v is not None and math.isfinite(float(v)) for v in scalars):
                    # C1 (final review): a bound-pinned or degenerate fit, or one the power law
                    # describes WORSE THAN A CONSTANT (r2 <= 0), is not a measurement. Measured
                    # on this slice's own tto_deltat_synth.dat: kappa_ph is flat below 10 K, so
                    # curve_fit parks at the lower bound and returns n = 0.5 with
                    # r2 = -3.6e13 and n_spread = 1.1e-16 -- i.e. the numbers read as a PERFECTLY
                    # window-stable exponent, the strongest honesty signal this probe can emit,
                    # precisely because the fit is degenerate. Declining is the only honest
                    # answer; the reason names which of the three tripped so the capability line
                    # is not a bare "declined".
                    bad = [f for f in fr.quality_flags if f in _FIT_FATAL_FLAGS]
                    if float(fr.r2) <= 0.0:
                        bad.append("worse_than_a_constant")
                    if bad:
                        fit_decline = _fit_decline_reason(bad, fr)
                    else:
                        kappa_ph_fit = KappaPhFit(
                            n=float(fr.params["n"]), n_sigma=float(fr.sigma["n"]),
                            n_spread=spread, n_loglog=n_loglog,
                            n_method_delta=(None if n_loglog is None
                                            else abs(float(fr.params["n"]) - n_loglog)),
                            b=float(fr.params["B"]), b_sigma=float(fr.sigma["B"]),
                            r2=float(fr.r2), n_points=int(fr.n_points),
                            window_k=[float(fr.fit_range[0]), float(fr.fit_range[1])],
                            ladder=ladder, quality_flags=list(fr.quality_flags))

        has_seebeck = _any_curve_has(curves, "seebeck")
        has_wf = _any_curve_has(curves, "kappa_e")
        has_pf = _any_curve_has(curves, "power_factor")
        has_zt = _any_curve_has(curves, "zt")
        caps = [
            Capability(name="thermal_conductivity", applicable=True,
                       reason="kappa(T) curves present"),
            Capability(name="seebeck", applicable=has_seebeck,
                       reason="Seebeck data present" if has_seebeck else "no Seebeck data"),
            Capability(name="wiedemann_franz", applicable=has_wf,
                       reason="kappa_e from rho present" if has_wf
                       else "requires finite ρ > 0"),
            Capability(name="power_factor", applicable=has_pf,
                       reason="power factor computed" if has_pf
                       else "requires finite ρ > 0"),
            Capability(name="figure_of_merit", applicable=has_zt,
                       reason="ZT data present" if has_zt else "no ZT data"),
            Capability(name="rrr", applicable=rrr_block is not None,
                       reason="zero-field ρ(T) ramp present" if rrr_block is not None
                       else "no zero-field ρ(T) ramp"),
            Capability(name="kappa_ph_power_fit", applicable=kappa_ph_fit is not None,
                       reason=("kappa_ph = B*T^n fitted on the <=10 K window"
                               if kappa_ph_fit is not None else fit_decline)),
            # Recognized-but-deferred (spec §7): named so the GUI hint strip can say why.
            Capability(name="callaway_fit", applicable=False, reason="deferred"),
            # m10: NOT a bare "deferred". A user reading the capabilities CSV sees
            # `kappa_ph_power_fit: fitted` next to this line and needs to be told they are
            # different claims — the free-n fit measures the exponent (2.0266 on the gate
            # file, which ARGUES AGAINST n = 3); this stub is the n = 3 hypothesis TEST.
            Capability(name="boundary_scattering_fit", applicable=False,
                       reason=("deferred — the n = 3 boundary-scattering hypothesis test, "
                               "distinct from the free-n kappa_ph fit")),
            Capability(name="diffusive_seebeck", applicable=False, reason="deferred"),
            Capability(name="kappa_field_sweep", applicable=False, reason="deferred"),
        ]
        data = TTOData(sample=_sample_block(header), curves=curves, dropped_groups=dropped,
                       rrr=rrr_block, summary=summary, kappa_ph_fit=kappa_ph_fit,
                       n_error_rows=n_error_rows,
                       capabilities=caps).model_dump(mode="json")
        return Result(status="ok", confidence=0.7,
                      confidence_parts={"detector": 1.0, "grouping": 1.0},
                      warnings=warnings, data=data, provenance=prov)
