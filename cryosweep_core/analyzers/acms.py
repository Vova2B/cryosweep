from __future__ import annotations
import numpy as np


def _cluster_1d(values, rel_tol, abs_tol: float = 0.0):
    """Cluster a 1-D value array by proximity. Sort unique-order the values; start a new
    cluster whenever the gap to the previous value exceeds max(abs_tol, rel_tol*|value|).
    Pure numpy, deterministic. Returns (labels_per_row, representatives) where
    representatives[label] = median of that cluster's member values (spec: 'representative =
    median of members'). Non-finite values are placed in a dedicated trailing cluster so they
    never merge with real setpoints (callers pre-filter finite anyway)."""
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


def group_rows(frequency, amplitude, field):
    """Cluster rows by (frequency 1%, amplitude 5%, field abs 5 Oe / rel 1%) and group by the
    joint cluster key. Returns a deterministically-sorted list of group dicts; row order is
    preserved inside each group's idx. NOTE: grouping.py's setpoint_key is temperature-tuned
    (half-integer bins) and is verified to MERGE these amplitude groups -> it is NOT reused."""
    fl, fr = _cluster_1d(frequency, 0.01)
    al, ar = _cluster_1d(amplitude, 0.05)
    xl, xr = _cluster_1d(field, 0.01, abs_tol=5.0)
    buckets: dict[tuple, list[int]] = {}
    for i in range(len(frequency)):
        buckets.setdefault((int(fl[i]), int(al[i]), int(xl[i])), []).append(i)
    groups = []
    for (kf, ka, kx), idx in buckets.items():
        groups.append({"frequency_hz": float(fr[kf]), "amplitude_oe": float(ar[ka]),
                       "field_oe": float(xr[kx]), "idx": np.asarray(sorted(idx), int)})
    groups.sort(key=lambda g: (g["frequency_hz"], g["amplitude_oe"], g["field_oe"]))
    return groups


import hashlib, pathlib
import pandas as pd
from pydantic import BaseModel, ConfigDict
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.vsm_blocks import ramps_from_temps
from cryosweep_core.result import Result, Gate, Provenance
from cryosweep_core.registry import Need

_MIN_PTS = 5            # groups smaller than this are dropped+logged
_RAMP_MIN_LEN = 15      # pinned: yields the real main group's up+down ramps (min_len<15 shatters,
                        # >=20 collapses both into one cooling ramp)
_DIR = {"warming": "up", "cooling": "down"}


class SCTransition(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tc_onset_k: float
    tc_mid_k: float
    drop_emu_per_oe: float
    chi_dprime_peak_t_k: float | None = None
    low_confidence: bool = False
    reasons: list[str] = []


class ChiPeak(BaseModel):
    model_config = ConfigDict(extra="ignore")
    t_f_k: float
    prominence: float
    low_confidence: bool = False
    reasons: list[str] = []


class ChiCurve(BaseModel):
    model_config = ConfigDict(extra="ignore")
    frequency_hz: float
    amplitude_oe: float
    field_oe: float
    direction: str                              # "up" | "down" | "mixed"
    n_points: int = 0
    t: list[float] = []
    chi_prime: list[float] = []                 # emu/Oe
    chi_dprime: list[float] = []
    chi_prime_molar: list[float] | None = None  # emu/mol*Oe
    chi_dprime_molar: list[float] | None = None
    m_dc: list[float] | None = None             # emu
    sc: SCTransition | None = None
    peak: ChiPeak | None = None


class Capability(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    applicable: bool
    reason: str = ""


class ACMSData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    probe: str = "acms"
    sample: dict = {}                           # {molar_mass, mass_mg, name}
    curves: list[ChiCurve] = []
    dropped_groups: list[dict] = []
    sc_transition: SCTransition | None = None
    chi_dprime_peaks: list[ChiPeak] = []
    capabilities: list[Capability] = []


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""


def _num(df, cmap, key):
    col = cmap.logical.get(key)
    if col is None:
        return None
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def _mad(x):
    x = np.asarray(x, float)
    return 1.4826 * float(np.median(np.abs(x - np.median(x)))) if x.size else 0.0


def _chidprime_noise(t, chipp):
    """chi'' noise per spec §3b (the ONE definition, shared by the SC corroboration in §3a and
    the peak detector in §3b): OLS straight line to chi''(T) over the whole (T-sorted) curve;
    residuals r = chi'' - line; sigma = 1.4826*MAD(r). Returns (sigma, r). Inputs must already
    be sorted by T."""
    t = np.asarray(t, float); y = np.asarray(chipp, float)
    A = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    r = y - (slope * t + intercept)
    return _mad(r), r


def _detect_sc(t, chip, chipp, field_oe, t_span):
    """SC screening transition on a chi'(T) diamagnetic drop (spec §3a). Sorts by T, then applies
    preconditions + criteria (a)(b)(c) + first-crossing tc_onset/tc_mid + chi'' corroboration.
    Returns SCTransition or None (decline)."""
    t = np.asarray(t, float); chip = np.asarray(chip, float); chipp = np.asarray(chipp, float)
    if t.size < 20 or (np.nanmax(t) - np.nanmin(t)) < 1.0 or abs(field_oe) >= 50.0:
        return None
    o = np.argsort(t, kind="stable")
    t, chip, chipp = t[o], chip[o], chipp[o]
    span = float(t[-1] - t[0])
    hi = t >= (t[-1] - 0.20 * span)                 # top-20%-of-T baseline
    lo = t <= (t[0] + 0.10 * span)                  # bottom-10%-of-T low level
    if hi.sum() < 5 or lo.sum() < 1:
        return None
    chi_n = float(np.median(chip[hi]))
    chi_low = float(np.median(chip[lo]))
    sigma = _mad(chip[hi])
    drop = chi_n - chi_low
    if sigma <= 0:
        return None
    # (a) 10 sigma drop; (b) truly diamagnetic low level
    if not (drop >= 10 * sigma and chi_low < 0):
        return None
    # (c) tilt guard (extrapolated-baseline; replaces the old full-curve sigma_range test, which
    # created a false-negative dead-band — a textbook step centered in the T window has full-curve
    # MAD ~ step/2, so drop/(6*sigma_range) ~ 0.33 wrongly declined it). Fit an OLS line to the
    # top-20%-of-T baseline window (same `hi` window as (a)) and extrapolate it DOWN to the low-T
    # level's location (median T of the bottom-10% `lo` window). If plain baseline drift already
    # explains chi'_low — it lies within 3*sigma of that extrapolated trend — DECLINE: it is a
    # tilt, not a step. A genuine step has a flat baseline whose extrapolation stays near the
    # plateau, leaving chi'_low far (>3*sigma) below it, so it passes regardless of where the
    # transition sits in the window (no centered-transition dead-band).
    t_lo = float(np.median(t[lo]))
    A_hi = np.vstack([t[hi], np.ones(int(hi.sum()))]).T
    slope_hi, icpt_hi = np.linalg.lstsq(A_hi, chip[hi], rcond=None)[0]
    baseline_at_lo = float(slope_hi) * t_lo + float(icpt_hi)
    if chi_low >= baseline_at_lo - 3.0 * sigma:
        return None
    d = (chip - chi_low) / drop                     # normalized drop, ~1 at high T, ~0 at low T
    # non-monotone-in-the-large guard: crosses the 50% level > 3 times -> decline
    cross50 = int(np.sum(np.diff((d >= 0.5).astype(int)) != 0))
    if cross50 > 3:
        return None

    def _cross(level):                              # first crossing from the low-T side
        for i in range(1, d.size):
            if d[i - 1] < level <= d[i]:
                x0, x1, y0, y1 = t[i - 1], t[i], d[i - 1], d[i]
                return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0)) if y1 != y0 else float(x1)
        return None

    tc_onset = _cross(0.90); tc_mid = _cross(0.50)
    if tc_onset is None or tc_mid is None:
        return None
    reasons: list[str] = []
    low_conf = False
    # points spanning the drop (between 10% and 90% levels)
    in_drop = int(np.sum((d > 0.10) & (d < 0.90)))
    if in_drop < 5:
        low_conf = True; reasons.append("drop spanned by < 5 points")
    if abs(np.median(chip[hi])) > 0 and (_mad(chip[hi]) / abs(np.median(chip[hi]))) > 0.20:
        low_conf = True; reasons.append("baseline relative spread > 20%")
    # chi'' corroboration: interior chi'' max inside [tc_mid - w, tc_onset + w].
    # Noise/prominence use the §3b definition via the shared _chidprime_noise helper
    # (OLS-line residuals, sigma = 1.4826*MAD(r)) — one statistic family, no drift.
    w = max(tc_onset - tc_mid, 0.02 * t_span)
    win = (t >= tc_mid - w) & (t <= tc_onset + w)
    peak_t = None
    sig_pp, r_pp = _chidprime_noise(t, chipp)
    if win.any():
        j = int(np.argmax(r_pp[win]))
        cand_t = float(t[win][j]); cand_h = float(r_pp[win][j])
        if sig_pp > 0 and cand_h >= 3 * sig_pp:
            peak_t = cand_t
    if peak_t is None:
        low_conf = True; reasons.append("no chi'' peak in transition window")
    return SCTransition(tc_onset_k=tc_onset, tc_mid_k=tc_mid, drop_emu_per_oe=drop,
                        chi_dprime_peak_t_k=peak_t, low_confidence=low_conf, reasons=reasons)


def _detect_chipp_peak(t, chipp):
    """Generic chi'' peak -> T_f (spec §3b, scipy-free). Sort by T; noise + residuals come from
    the shared _chidprime_noise helper (Task 4: OLS line over the whole curve, r = chi'' - line,
    sigma = 1.4826*MAD(r)). Candidate = max r in the interior (outside the endpoint 5% of the
    T span); prominence = r(T_peak). Detect when prominence >= 5*sigma."""
    t = np.asarray(t, float); y = np.asarray(chipp, float)
    if t.size < 10:
        return None
    o = np.argsort(t, kind="stable")
    t, y = t[o], y[o]
    sig, r = _chidprime_noise(t, y)
    if sig <= 0:
        return None
    span = float(t[-1] - t[0])
    interior = (t > t[0] + 0.05 * span) & (t < t[-1] - 0.05 * span)
    if not interior.any():
        return None
    ridx = np.where(interior)[0]
    j = ridx[int(np.argmax(r[interior]))]
    prom = float(r[j])
    if prom < 5 * sig:
        return None
    # Spike-strip / width gate (physics integrity): a real susceptibility peak (T_f) has finite
    # width, so its half-max shoulders are also elevated; a lone measurement glitch is a single
    # outlier point sitting on the noise floor. Require >= 3 points (the peak + >= 2 corroborating
    # neighbors) within the T-neighborhood above half prominence. Without this a 1-point glitch in
    # the real (featureless) file is falsely reported as a peak.
    if int(np.sum((np.abs(t - t[j]) <= 0.05 * span) & (r >= 0.5 * prom))) < 3:
        return None
    reasons: list[str] = []
    low_conf = False
    near = int(np.sum(np.abs(t - t[j]) <= 0.05 * span))
    if near < 5:
        low_conf = True; reasons.append("peak spanned by < 5 points")
    if prom < 8 * sig:
        low_conf = True; reasons.append("prominence between 5x and 8x noise")
    return ChiPeak(t_f_k=float(t[j]), prominence=prom, low_confidence=low_conf, reasons=reasons)


def _best_sc(curves):
    """Aggregate per-curve SC transitions (spec §2b tie-break): not-low-confidence first, then
    largest drop, then lowest frequency, then most points."""
    cand = [c for c in curves if c.sc is not None]
    if not cand:
        return None
    best = min(cand, key=lambda c: (c.sc.low_confidence, -c.sc.drop_emu_per_oe,
                                    c.frequency_hz, -c.n_points))
    return best.sc


class ACMSAnalyzer:
    probe = "acms"
    needs = ()   # required data columns are checked inside analyze (-> status="gated"); mass/molar
                 # are NOT needs (D3) -> raw chi is always computable.

    def analyze(self, rawtable, cfg) -> Result:
        df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
        header = rawtable.header
        prov = Provenance(file=getattr(rawtable, "path", None) or header.title or "",
                          sha256=_sha256(getattr(rawtable, "path", None)),
                          app_version=header.app_version, config=cfg.model_dump(mode="json"))
        # --- required columns -> gated on absence ---
        for k in ("m_prime", "m_dprime", "amplitude"):
            if k not in cmap.logical:
                return Result(status="gated", confidence=0.0, data={"probe": "acms"},
                              gate=[Gate(need=k, reason=f"ACMS file missing a '{k}' column")],
                              provenance=prov)
        temp = _num(df, cmap, "temperature")
        m_prime = _num(df, cmap, "m_prime")
        m_dprime = _num(df, cmap, "m_dprime")
        amp = _num(df, cmap, "amplitude")
        freq = _num(df, cmap, "frequency")
        field = _num(df, cmap, "field")
        if temp is None or freq is None:
            miss = "temperature" if temp is None else "frequency"
            return Result(status="gated", confidence=0.0, data={"probe": "acms"},
                          gate=[Gate(need=miss, reason=f"ACMS file missing a '{miss}' column")],
                          provenance=prov)
        if field is None:
            field = np.zeros(temp.size, float)             # field optional -> assume 0
        m_dc = _num(df, cmap, "m_dc")
        # --- row filter ---
        keep = (np.isfinite(temp) & np.isfinite(m_prime) & np.isfinite(m_dprime)
                & np.isfinite(freq) & (amp > 0))
        n_dropped_rows = int((~keep).sum())
        temp, m_prime, m_dprime, amp, freq, field = (a[keep] for a in
                                                     (temp, m_prime, m_dprime, amp, freq, field))
        m_dc = m_dc[keep] if m_dc is not None else None
        mol = header.molar_mass
        mass_g = (header.mass_mg / 1000.0) if header.mass_mg else None
        molar_on = mol is not None and mass_g is not None
        curves: list[ChiCurve] = []
        dropped: list[dict] = []
        if temp.size:
            for g in group_rows(freq, amp, field):
                idx = g["idx"]
                if idx.size < _MIN_PTS:
                    dropped.append({"frequency_hz": g["frequency_hz"], "amplitude_oe": g["amplitude_oe"],
                                    "field_oe": g["field_oe"], "n_points": int(idx.size),
                                    "reason": f"< {_MIN_PTS} points"})
                    continue
                Tg = temp[idx]
                ramps = ramps_from_temps(Tg.tolist(), min_len=_RAMP_MIN_LEN)
                # "mixed" fallback is spec-mandated defensive depth: ramps_from_temps never
                # actually returns [] for n>=5 today (only n==0 does), but the whole-group
                # curve path must exist if that contract ever changes.
                spans = ([(r["direction"], r["i0"], r["i1"] + 1) for r in ramps]
                         if ramps else [("mixed", 0, idx.size)])
                for direction, a0, a1 in spans:
                    sub = idx[a0:a1]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        chip = (m_prime[sub] / amp[sub])
                        chipp = (m_dprime[sub] / amp[sub])
                    cur = ChiCurve(
                        frequency_hz=g["frequency_hz"], amplitude_oe=g["amplitude_oe"],
                        field_oe=g["field_oe"], direction=_DIR.get(direction, direction),
                        n_points=int(sub.size), t=temp[sub].tolist(),
                        chi_prime=chip.tolist(), chi_dprime=chipp.tolist())
                    if molar_on:
                        f = mol / (mass_g)                  # emu/Oe * (g/mol)/g = emu/mol*Oe
                        cur.chi_prime_molar = (chip * f).tolist()
                        cur.chi_dprime_molar = (chipp * f).tolist()
                    if m_dc is not None:
                        md = m_dc[sub]
                        # ALL-finite guard: partial-NaN would leak NaN into JSON (standing
                        # constraint). Rows with M-DC but no finite M' never reach curves
                        # anyway — the row filter requires finite m_prime.
                        if md.size and np.isfinite(md).all():
                            cur.m_dc = md.tolist()
                    t_arr = np.asarray(cur.t, float)
                    t_span = float(t_arr.max() - t_arr.min()) if t_arr.size else 0.0
                    cur.sc = _detect_sc(cur.t, cur.chi_prime, cur.chi_dprime, cur.field_oe, t_span)
                    cur.peak = _detect_chipp_peak(cur.t, cur.chi_dprime)
                    curves.append(cur)
        # --- degenerate: no usable curves -> gated (never ok+empty) ---
        if not curves:
            return Result(status="gated", confidence=0.0, data={"probe": "acms"},
                          gate=[Gate(need="ac_data", reason="no usable AC data "
                                     "(all rows sentinel or every group dropped)")],
                          provenance=prov)
        has_mdc = any(c.m_dc for c in curves)
        best_sc = _best_sc(curves)
        peaks = [c.peak for c in curves if c.peak is not None]
        caps = [
            Capability(name="ac_susceptibility", applicable=True, reason="chi'/chi'' curves present"),
            Capability(name="superconducting_screening", applicable=best_sc is not None,
                       reason="diamagnetic drop detected" if best_sc else "no diamagnetic drop"),
            Capability(name="chi_dprime_peak", applicable=bool(peaks),
                       reason="chi'' peak detected" if peaks else "no chi'' peak"),
            Capability(name="molar_normalization", applicable=molar_on,
                       reason="mass + molar mass supplied" if molar_on
                       else "molar chi: supply --molar-mass/--mass-mg"),
            Capability(name="dc_magnetization", applicable=has_mdc,
                       reason="M-DC data present" if has_mdc else "no M-DC data"),
        ]
        data = ACMSData(
            sample={"molar_mass": mol, "mass_mg": header.mass_mg,
                    "name": getattr(header, "title", None)},
            curves=curves, dropped_groups=dropped, sc_transition=best_sc,
            chi_dprime_peaks=peaks, capabilities=caps).model_dump(mode="json")
        warnings = ([f"{n_dropped_rows} non-finite/sentinel rows dropped"] if n_dropped_rows else [])
        return Result(status="ok", confidence=0.7,
                      confidence_parts={"detector": 1.0, "grouping": 1.0},
                      warnings=warnings, data=data, provenance=prov)
