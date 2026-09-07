"""Instrument-sigma primitives shared by the two Hall analyzers.

Extracted 2026-09-07 from hall_tempdep._interp_fixed_field_sigma_curves so the field-sweep
analyzer can compute the SAME quantity by the SAME estimator. Instrument sigma is a
WEAKER, DIFFERENT claim than the residual (fit-scatter) sigma: it measures the
instrument's repeat noise, not how well the line fits. The two must never share a label.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

#: The R/rho ratio must be this constant (relative spread) or sigma is DECLINED. Measured
#: 2026-08-10 on both real files: 1e-12 relative, ratio 1000.0 exactly. The header geometry
#: fields are NOT a valid alternative source (unset dummies predict a 10x-wrong factor).
RATIO_CONSTANCY_TOL = 1e-6


def row_sigma_R(df, cmap, channel: int):
    """Per-row instrument sigma of channel `channel`'s RESISTANCE column, in Ohm.

    sigma_R = std_column (Ohm-m) * (Resistance/Resistivity), using the file's own exact
    geometry factor. Returns None -- never a shaky number -- when a needed column is
    absent or the ratio's relative spread exceeds RATIO_CONSTANCY_TOL.
    """
    std_key = (f"rho_std_bridge{channel}"
               if f"rho_std_bridge{channel}" in cmap.logical
               else f"rho_std_ch{channel}")
    res_key = f"resistance_ch{channel}"
    rty_key = f"resistivity_ch{channel}"
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
    if not (spread < RATIO_CONSTANCY_TOL):
        return None                                # DECLINE, never emit a shaky sigma
    return SD * med


def slope_sigma_ols(B, sigma_y):
    """sigma of an OLS-with-intercept slope from per-point sigma of y.

    w_i = (B_i - Bbar) / sum((B - Bbar)^2);  var(slope) = sum(w_i^2 * sigma_i^2).
    Returns None on a degenerate design (zero x-spread) or any non-finite input.
    """
    B = np.asarray(B, float)
    S = np.asarray(sigma_y, float)
    if B.size != S.size or B.size < 2:
        return None
    if not (np.isfinite(B).all() and np.isfinite(S).all()):
        return None
    dev = B - float(B.mean())
    denom = float(np.sum(dev ** 2))
    if denom <= 0:
        return None
    var = float(np.sum((dev / denom) ** 2 * S ** 2))
    return float(np.sqrt(var)) if np.isfinite(var) and var >= 0 else None
