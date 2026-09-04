"""Shared uncertainty helpers (U6, 2026-08-10 spec §4) — extracted from analyzers/tto.py.

The correction and propagation FORMULAS are shared here; the endpoint SELECTION POLICY stays
per-analyzer (tto's `_endpoint` embeds tto's own mask policy; resistivity's differs), which is
why `rrr_sigma` takes the caller's endpoint function as a parameter. Formula lines are copied
character-for-character from tto.py (byte-identical TTO behavior is pinned by the existing
oracles: rrr_std = 0.01742 on the real gate file, 0.00793 on the power-law fixture).

This is the ONE sanctioned cross-analyzer import surface this slice creates: a *fitting*
module importable by any analyzer.
"""
from __future__ import annotations

import math

import numpy as np

MEDIAN_SE = 1.2533          # sqrt(pi/2): standard error of a MEDIAN vs a mean (C3)


def endpoint_sigma(T, rho, rho_std, lowest: bool, k: int):
    """sigma of the median-of-k RRR endpoint: 1.2533 * median(sigma_point) / sqrt(k) (C3).

    The mask and the k-nearest selection MUST match the caller's endpoint function exactly,
    or the selected sigma values would not be the ones behind the selected rho values. The
    1.2533 = sqrt(pi/2) is the median's efficiency penalty relative to a mean; without it
    sigma_RRR is 1.784x too large."""
    T = np.asarray(T, float)
    rho = np.asarray(rho, float)
    sd = np.asarray(rho_std, float)
    m = np.isfinite(rho) & (rho > 0) & np.isfinite(T)
    Tm, sdm = T[m], sd[m]
    if Tm.size == 0:
        return None
    kk = min(k, Tm.size)
    order = np.argsort(Tm, kind="stable")          # stable -> deterministic on duplicate T
    sel = order[:kk] if lowest else order[-kk:]
    s = sdm[sel]
    s = s[np.isfinite(s)]
    if s.size == 0:
        return None
    val = MEDIAN_SE * float(np.median(s)) / math.sqrt(kk)
    return val if math.isfinite(val) else None


def rrr_sigma(T, rho, rho_std, rrr, k: int, endpoint_fn):
    """sigma_RRR = RRR * sqrt((sigma_hi/rho_hi)^2 + (sigma_lo/rho_lo)^2). None — NEVER NaN:
    it is a bare float the D11 `_san` sanitiser does not walk, so a NaN would break the
    standing json.dumps(allow_nan=False) gate.

    `endpoint_fn(T, rho, lowest, k) -> (t, rho_val)` is the caller's own endpoint selector —
    passing it keeps the "mask must match" law enforceable per-caller without importing
    analyzer privates."""
    s_lo = endpoint_sigma(T, rho, rho_std, True, k)
    s_hi = endpoint_sigma(T, rho, rho_std, False, k)
    if s_lo is None or s_hi is None:
        return None
    _, r_lo = endpoint_fn(T, rho, True, k)
    _, r_hi = endpoint_fn(T, rho, False, k)
    if not (math.isfinite(r_lo) and math.isfinite(r_hi)) or r_lo <= 0 or r_hi <= 0:
        return None
    val = float(rrr) * math.sqrt((s_hi / r_hi) ** 2 + (s_lo / r_lo) ** 2)
    return val if math.isfinite(val) else None


def straddles_threshold(rrr, std, lo, hi) -> bool:
    """True when the +-1 sigma band spans a classification threshold (thresholds are the
    caller's — tto passes its metal/insulator 0.98/1.02 pair)."""
    return bool((rrr - std) < hi < (rrr + std)
                or (rrr - std) < lo < (rrr + std))
