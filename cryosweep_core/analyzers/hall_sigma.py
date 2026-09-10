"""Row-level instrument helpers shared by the two Hall analyzers.

Two related things live here, both concerned with what the instrument reported per row
rather than with how well a model fits:

1. The instrument-sigma primitives (`row_sigma_R`, `slope_sigma_ols`), extracted
   2026-09-07 from hall_tempdep._interp_fixed_field_sigma_curves so the field-sweep
   analyzer computes the SAME quantity by the SAME estimator. Instrument sigma is a
   WEAKER, DIFFERENT claim than the residual (fit-scatter) sigma: it measures the
   instrument's repeat noise, not how well the line fits. The two must never share a
   label.
2. `skip_row_warning` (2026-09-10), which judges whether a row the operator skipped
   looked physical. It is not a sigma primitive; it lives here because it makes the same
   per-row comparison against the file's own reported std-dev, and both Hall analyzers
   need one copy. The module scope is deliberately "per-row instrument facts" rather than
   "sigma" alone -- if something lands here that is neither, it belongs elsewhere.
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


#: |R| or reported instrument sigma more than this many times the KEPT rows' median marks
#: a skipped leading row unphysical rather than merely noisy. Measured 2026-09-10 on real
#: files: corrupted leading rows sit at ~1e11-1e13 on both ratios, genuinely physical ones
#: at ~1 (docs/physics-reference.md, "Leading-row skip") -- the separation is enormous, so
#: this is not a tuning exercise; a test pins the choice's insensitivity across many orders
#: of magnitude either side.
SKIP_PHYSICAL_RATIO = 1e6


def skip_row_warning(df, cmap, channel: int, skip_rows: int) -> str | None:
    """Judge whether any of the first `skip_rows` rows of channel `channel` looks
    PHYSICAL against the rest of the file, and if so return a warning naming the exact
    remedy. Shared by both Hall analyzers so `--skip-rows` means one thing across the
    probe (task 4b).

    This is never a gate and never decides what to exclude -- `skip_rows` is the
    operator's call. It only tells them when their choice (commonly the default of 1)
    may have thrown away good data. A row counts as PHYSICAL unless its |R| or its
    reported instrument sigma sits more than SKIP_PHYSICAL_RATIO times the KEPT rows'
    median -- the judgement is about the skipped row alone, never about whether the
    analysis as a whole should proceed.

    With skip_rows > 1 this reports the FIRST physical-looking row in index order and
    names one remedy, even when several of the skipped rows are independently physical.
    That is deliberate rather than an oversight: the remedy (`--skip-rows 0`, then choose
    a smaller N) is the same whichever of them is named, and one warning beats N. A test
    pins this so a future maintainer who wants per-row reporting changes it on purpose.

    Returns None when: skip_rows <= 0 (nothing was skipped); skip_rows >= the row count
    (nothing left to compare against); the resistance column for `channel` is absent; or
    no skipped row carries enough signal (R and/or instrument sigma) to judge at all --
    "say nothing rather than guessing" over declaring a verdict without evidence.
    """
    res_key = f"resistance_ch{channel}"
    if skip_rows <= 0 or res_key not in cmap.logical:
        return None
    R = pd.to_numeric(df[cmap.logical[res_key]], errors="coerce").to_numpy(float)
    if skip_rows >= R.size:
        return None
    sigma = row_sigma_R(df, cmap, channel)
    kept_R = R[skip_rows:]
    med_R = float(np.nanmedian(np.abs(kept_R))) if np.isfinite(kept_R).any() else None
    med_sigma = None
    if sigma is not None:
        kept_sigma = sigma[skip_rows:]
        if np.isfinite(kept_sigma).any():
            med_sigma = float(np.nanmedian(kept_sigma))
    for j in range(skip_rows):
        Rj = R[j]
        r_ratio = (abs(Rj) / med_R
                   if (med_R and med_R > 0 and np.isfinite(Rj)) else None)
        sigmaj = sigma[j] if sigma is not None else None
        sd_ratio = (sigmaj / med_sigma
                    if (sigmaj is not None and np.isfinite(sigmaj)
                        and med_sigma and med_sigma > 0) else None)
        if r_ratio is None and sd_ratio is None:
            continue                              # no judgement possible -- say nothing
        if ((r_ratio is not None and r_ratio > SKIP_PHYSICAL_RATIO)
                or (sd_ratio is not None and sd_ratio > SKIP_PHYSICAL_RATIO)):
            continue                              # unphysical: the default did its job
        r_txt = (f"|R| = {abs(Rj):.2e} Ohm against a file median of {med_R:.2e}"
                 if med_R else "R not comparable")
        sd_txt = (f", reported std {sigmaj:.2e} against a median of {med_sigma:.2e}"
                  if (sigmaj is not None and med_sigma) else "")
        which = "it" if skip_rows == 1 else f"row {j}"
        plural = "" if skip_rows == 1 else "s"
        # "by default" only when skip_rows sits at HallCfg's own default (1) -- an
        # explicit --skip-rows N (N != 1) is the operator's own choice, not a default.
        cause = "by default" if skip_rows == 1 else f"via --skip-rows {skip_rows}"
        return (f"skipped {skip_rows} leading data row{plural} {cause}, but {which} "
                f"looks physical ({r_txt}{sd_txt}). "
                f"Re-run with --skip-rows {j} to include it.")
    return None


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
