"""Row-level instrument helpers shared by the two Hall analyzers.

Two related things live here, both concerned with what the instrument reported per row
rather than with how well a model fits:

1. The instrument-sigma primitives (`row_sigma_R`, `slope_sigma_ols`), extracted
   2026-09-07 from hall_tempdep._interp_fixed_field_sigma_curves so the field-sweep
   analyzer computes the SAME quantity by the SAME estimator. Instrument sigma is a
   WEAKER, DIFFERENT claim than the residual (fit-scatter) sigma: it measures the
   instrument's repeat noise, not how well the line fits. The two must never share a
   label.
2. The leading-row judgement (2026-09-10; made conditional 2026-09-14) -- one ratio
   test with two consumers. `corrupt_leading_rows` decides what `skip_rows="auto"` drops;
   `skip_row_warning` tells an operator who passed an explicit count that a row they
   dropped looked fine. Both read the same `SKIP_PHYSICAL_RATIO` from the same place on
   purpose: a threshold that acts and a threshold that warns must never be able to
   disagree about the same row. Neither is a sigma primitive; they live here because they
   make the same per-row comparison against the file's own reported std-dev, and both Hall
   analyzers need one copy. The module scope is deliberately "per-row instrument facts"
   rather than "sigma" alone -- if something lands here that is neither, it belongs
   elsewhere.
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


#: Auto mode never scans -- and so never drops -- more than this many leading rows,
#: however broken the head of a file is. A bounded blast radius is what makes a detector
#: acceptable here at all: the measured defect is a SINGLE pre-settling row, so ten is
#: already generous, and a file whose first ten rows are all unphysical has a problem no
#: row-skip should be quietly papering over.
AUTO_SCAN_CAP = 10


def _channel_ratios(df, cmap, channel: int, baseline_skip: int):
    """(|R| ratio, sigma ratio) per row for `channel`, against its own baseline median.

    The baseline EXCLUDES the first `baseline_skip` rows so the comparison can never be
    circular: a corrupt row must not be allowed to inflate the median it is judged
    against. Either element is None when that quantity has no comparable baseline.
    """
    res_key = f"resistance_ch{channel}"
    if res_key not in cmap.logical:
        return None, None
    R = pd.to_numeric(df[cmap.logical[res_key]], errors="coerce").to_numpy(float)
    if baseline_skip >= R.size:
        return None, None
    kept_R = R[baseline_skip:]
    med_R = float(np.nanmedian(np.abs(kept_R))) if np.isfinite(kept_R).any() else None
    r_ratio = np.abs(R) / med_R if (med_R and med_R > 0) else None
    sigma = row_sigma_R(df, cmap, channel)
    s_ratio = None
    if sigma is not None:
        kept_sigma = sigma[baseline_skip:]
        if np.isfinite(kept_sigma).any():
            med_sigma = float(np.nanmedian(kept_sigma))
            if med_sigma and med_sigma > 0:
                s_ratio = sigma / med_sigma
    return r_ratio, s_ratio


def _row_is_corrupt(ratios, i: int) -> bool:
    """True only on POSITIVE evidence -- some ratio exists for row `i` and exceeds the
    threshold. A row nothing can judge is never condemned."""
    for r_ratio, s_ratio in ratios:
        for ratio in (r_ratio, s_ratio):
            if ratio is None:
                continue
            v = ratio[i]
            if np.isfinite(v) and v > SKIP_PHYSICAL_RATIO:
                return True
    return False


def corrupt_leading_rows(df, cmap, max_scan: int = AUTO_SCAN_CAP) -> int:
    """How many leading rows are PROVABLY corrupt -- the count `skip_rows="auto"` drops.

    A row counts corrupt when, on ANY resistance channel present, its |R| or its reported
    instrument sigma exceeds SKIP_PHYSICAL_RATIO times that channel's baseline median. Any
    one channel's evidence condemns the whole row because a reading taken before the bridge
    settled is not a reading on any channel; on the one corrupted file measured here both
    channels say so at once (1.16e12x on channel 1, 1.32e11x on channel 2).

    Scanning stops at the first row that is not corrupt, and never passes `max_scan`.
    Absence of evidence is never evidence: a row that cannot be judged -- no usable R, no
    reported sigma -- is KEPT. On every uncorrupted file measured here this returns 0, which
    is the whole point of consulting it instead of dropping a row unconditionally.
    """
    channels = [ch for ch in (1, 2, 3) if f"resistance_ch{ch}" in cmap.logical]
    if not channels or len(df) < 4:
        return 0
    # Enough clean rows must remain to form a median. On a real file (thousands of rows)
    # this is just max_scan; on a short synthetic it backs off rather than refusing to judge.
    baseline_skip = min(max_scan, max(1, len(df) // 3))
    ratios = [_channel_ratios(df, cmap, ch, baseline_skip) for ch in channels]
    n = 0
    while n < max_scan and _row_is_corrupt(ratios, n):
        n += 1
    return n


def resolve_skip_rows(skip_rows, df, cmap) -> tuple[int, bool]:
    """(count, came_from_auto) -- how many leading rows to drop, for both Hall analyzers.

    "auto" consults `corrupt_leading_rows`. An explicit integer is obeyed verbatim AND
    turns detection off: the operator has already decided, and a detector that overrode
    them would make `--skip-rows 0` mean something other than "keep everything".
    """
    if skip_rows == "auto":
        return corrupt_leading_rows(df, cmap), True
    return max(0, int(skip_rows)), False


def auto_skip_warning(df, cmap, n_skip: int) -> str | None:
    """Say what auto dropped and why, naming the evidence and the flag that reverses it.

    Dropping data silently is the one thing this design must never do -- so when auto acts
    it reports the worst-offending channel's own numbers. Silent when it dropped nothing,
    which is the common case and needs no noise.
    """
    if n_skip <= 0:
        return None
    baseline_skip = min(AUTO_SCAN_CAP, max(1, len(df) // 3))
    worst = None                                   # (ratio, channel, value, median, what)
    for ch in (1, 2, 3):
        if f"resistance_ch{ch}" not in cmap.logical:
            continue
        R = pd.to_numeric(df[cmap.logical[f"resistance_ch{ch}"]], errors="coerce").to_numpy(float)
        r_ratio, s_ratio = _channel_ratios(df, cmap, ch, baseline_skip)
        sigma = row_sigma_R(df, cmap, ch)
        for i in range(min(n_skip, R.size)):
            if r_ratio is not None and np.isfinite(r_ratio[i]) and r_ratio[i] > 0:
                # the ratio was formed AS |R[i]|/median, so the median divides back out
                cand = (r_ratio[i], ch, abs(R[i]), abs(R[i]) / r_ratio[i], "|R|", "Ohm")
                if worst is None or cand[0] > worst[0]:
                    worst = cand
            if (s_ratio is not None and sigma is not None
                    and np.isfinite(s_ratio[i]) and s_ratio[i] > 0):
                cand = (s_ratio[i], ch, sigma[i], sigma[i] / s_ratio[i],
                        "reported std", "Ohm")
                if worst is None or cand[0] > worst[0]:
                    worst = cand
    if worst is None:
        return None
    ratio, ch, value, median, what, unit = worst
    plural = "" if n_skip == 1 else "s"
    return (f"dropped {n_skip} leading data row{plural} as unphysical: channel {ch} "
            f"{what} = {value:.2e} {unit} is {ratio:.1e} times the file median "
            f"({median:.2e} {unit}). Re-run with --skip-rows 0 to keep "
            f"{'it' if n_skip == 1 else 'them'}.")


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
