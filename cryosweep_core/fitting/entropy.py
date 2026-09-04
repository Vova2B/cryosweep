from __future__ import annotations
import numpy as np
from cryosweep_core.fitting.heat_capacity import eval_lowt_cp_over_t

R = 8.314462618

# np.trapz was removed in NumPy 2.0 in favor of np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", None) or np.trapz


def dulong_petit_limit(n_atoms):
    if n_atoms is None or not np.isfinite(n_atoms) or n_atoms <= 0:
        return None
    return 3.0 * float(n_atoms) * R


def _clean_order(T, y):
    """Return the (finite, T>0) mask and the ascending-sort permutation for (T, y).

    Exposed so an aligned lattice array can be masked+sorted with the SAME
    mask/order as (T, cp) — otherwise cmag = cp - lat is misaligned on a
    cooling (descending-T) ramp or when any row is filtered out.
    """
    T = np.asarray(T, float); y = np.asarray(y, float)
    m = np.isfinite(T) & np.isfinite(y) & (T > 0)
    o = np.argsort(T[m])
    return m, o


def _clean(T, y):
    m, o = _clean_order(T, y)
    T = np.asarray(T, float)[m]; y = np.asarray(y, float)[m]
    return T[o], y[o]


def compute_entropy(full_temperature, full_cp, *, lowt_model=None, lattice_cp=None, extrapolate=True):
    out = {"temperature": [], "s_total": [], "s_magnetic": None,
           "extrapolated": False, "lattice_source": None, "reason": ""}
    m, o = _clean_order(full_temperature, full_cp)
    T = np.asarray(full_temperature, float)[m][o]
    cp = np.asarray(full_cp, float)[m][o]
    if T.size < 2:
        out["reason"] = "fewer than 2 finite (T, Cp) points"; return out
    cot = cp / T                                     # Cp/T
    # low-T tail: integral from 0 to T[0] of the chosen model's Cp/T
    tail = 0.0
    if extrapolate and lowt_model is not None:
        key, params = lowt_model
        try:
            tgrid = np.linspace(0.0, T[0], 64)       # integrate the full 0..T[0] tail
            ygrid = np.asarray(eval_lowt_cp_over_t(key, params, tgrid), float)
            # models may divide by T / take log(T0/T) at T=0 -> non-finite endpoint;
            # replace non-finite samples with the nearest finite value (finite Cp/T limit).
            bad = ~np.isfinite(ygrid)
            if bad.any():
                good = np.flatnonzero(~bad)
                if good.size == 0:
                    raise ValueError("no finite tail samples")
                idx = np.clip(np.searchsorted(good, np.flatnonzero(bad)), 0, good.size - 1)
                ygrid[bad] = ygrid[good[idx]]
            tail = float(_trapz(ygrid, tgrid))
            out["extrapolated"] = True
        except Exception:
            tail = 0.0                               # tail is additive; failure => start at 0 at T[0]
            out["extrapolated"] = False
            out["reason"] = "tail extrapolation failed"
    s_total = tail + np.concatenate([[0.0], np.cumsum(0.5 * (cot[1:] + cot[:-1]) * np.diff(T))])
    out["temperature"] = T.tolist(); out["s_total"] = s_total.tolist()
    if lattice_cp is not None:
        lat_raw = np.asarray(lattice_cp, float)
        # Lattice must correspond row-for-row to the ORIGINAL (T, Cp) input so it
        # can be masked+sorted with the SAME finite/T>0 mask AND ascending order.
        # Length guard (fallback): mismatched length => cannot align => skip magnetic.
        aligned = lat_raw[m][o] if lat_raw.shape == m.shape else lat_raw
        # Require the lattice to line up row-for-row with the cleaned/sorted T. A partially
        # non-finite lattice (e.g. a reference file that only overlaps part of the T range)
        # is allowed: out-of-overlap points are GENUINELY truncated -> the s_magnetic entry
        # is None there (JSON null), so the magnetic curve stops at the overlap edges instead
        # of plateauing. The list stays aligned (same length as temperature/s_total) with None
        # at out-of-overlap indices and finite floats inside the overlap.
        if aligned.shape == T.shape and lat_raw.shape == m.shape:
            fin = np.isfinite(aligned)
            if fin.sum() >= 2:
                cmag = cp - aligned
                cot_m = cmag / T
                incr = 0.5 * (cot_m[1:] + cot_m[:-1]) * np.diff(T)
                incr = np.where(fin[1:] & fin[:-1], incr, 0.0)   # zero out non-overlap intervals
                s_mag = np.concatenate([[0.0], np.cumsum(incr)])
                out["lattice_source"] = "provided"
                if fin.all():
                    # full-overlap: plain floats, byte-identical to the pre-truncation behavior
                    out["s_magnetic"] = s_mag.tolist()
                else:
                    # partial overlap: None (never NaN) at out-of-overlap indices
                    out["s_magnetic"] = [float(v) if f else None
                                         for v, f in zip(s_mag, fin)]
                    out["reason"] = ((out["reason"] + "; ") if out["reason"] else "") + \
                        "magnetic entropy truncated to lattice T-overlap"
    return out


_RLN_TOL = 0.25   # closed O5 (2026-08-10): 25% of the matched value. Measured separation:
                  # genuine match 2% vs real-file mismatches >= 70%; the R ln ladder's
                  # tightest half-gap (Rln2 -> Rln3) is 1.686 J/mol/K.


def rln_match_fields(value, s_magnetic, tol=_RLN_TOL):
    """The O5 additive match verdict for a given R ln(2J+1) `value` against the saturation
    of `s_magnetic` (last finite element). Negative magnetic entropy is unphysical -> always
    matched=False (closed O5; never suppressed). No usable saturation -> all-None verdict."""
    out = {"distance": None, "rel_err": None, "matched": False, "tol": tol}
    if not s_magnetic:
        return out
    finite = np.asarray([v for v in s_magnetic if v is not None and np.isfinite(v)], float)
    if finite.size == 0:
        return out
    sat = float(finite[-1])
    distance = abs(sat - value)
    rel_err = distance / value
    # F11 (final-review): `sat > 0.0` is DEFENSIVE-ONLY and provably cannot change the
    # outcome today — the ladder starts at j = 0.5 so `value >= R ln2 = 5.763 > 0`, and
    # `rel_err = |sat - value| / value <= tol (0.25)` already forces `sat >= 0.75*value > 0`.
    # That is why mutation M10 (dropping this term) survived: the survival is CORRECT, not a
    # test gap. It is kept because it makes O5's decision ("negative S_mag saturation is
    # unphysical -> matched = False") explicit at the point of decision rather than an
    # emergent consequence of the tolerance, and it would start binding the moment `tol`
    # rose above 1.0 or the ladder gained a value <= 0. Measured on the two real
    # negative-saturation files: rel_err 1.1265 and 1.4878, both already unmatched.
    out.update(distance=float(distance), rel_err=float(rel_err),
               matched=bool(rel_err <= tol and sat > 0.0))
    return out


def suggest_rln(s_magnetic, jmax=4):
    from math import log
    default = {"j": 0.5, "value": R * log(2.0), "label": "R ln2"}
    if not s_magnetic:
        return {**default, **rln_match_fields(default["value"], s_magnetic)}
    # s_magnetic may carry None at truncated (out-of-overlap) indices; use the last
    # FINITE value as the saturation estimate (never NaN). Full-overlap (all-float) is
    # unaffected: the last element is finite, so behavior is byte-identical.
    finite = np.asarray([v for v in s_magnetic if v is not None and np.isfinite(v)], float)
    if finite.size == 0:
        return {**default, **rln_match_fields(default["value"], None)}
    sat = float(finite[-1])
    best = default; err = abs(sat - default["value"])
    j = 0.5
    while j <= jmax:
        val = R * log(2 * j + 1)
        e = abs(sat - val)
        if e < err:
            err = e; best = {"j": j, "value": val, "label": f"R ln{int(2*j+1)}"}
        j += 0.5
    # Closed O5 (2026-08-10): append-only match verdict — j/value/label semantics unchanged
    # (U7), the renderer keeps drawing exactly what it draws today.
    return {**best, **rln_match_fields(best["value"], s_magnetic)}
