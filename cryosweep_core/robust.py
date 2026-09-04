"""Pure-numpy robust statistics for display-robustness / outlier diagnostics (DQ-A).
matplotlib-free, Qt-free. Single source of truth; all helpers guard degenerate input and never raise."""
from __future__ import annotations
import numpy as np

_MAD_TO_SIGMA = 1.4826
_DECADE_LOG_THRESHOLD = 2.0     # robust span (decades) above which we work in log space


def _finite(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    return v[np.isfinite(v)]


def robust_decade_span(values) -> float:
    """Decades spanned by the *bulk* (2.5–97.5 percentile) of finite, positive values.
    Uses a robust percentile span (not raw min/max) so a heavy tail cannot inflate the
    span and spuriously flip the analysis into log space. Returns 0.0 if undefined."""
    v = _finite(values)
    v = v[v > 0]
    if v.size < 3:
        return 0.0
    lo, hi = np.percentile(v, [2.5, 97.5])
    if lo <= 0 or hi <= 0:
        return 0.0
    return float(np.log10(hi / lo))


def is_log_space(values) -> bool:
    """True when the robust decade span exceeds the log-space threshold (shared decision)."""
    return robust_decade_span(values) > _DECADE_LOG_THRESHOLD


def _center_scale(w: np.ndarray):
    med = float(np.median(w))
    s = _MAD_TO_SIGMA * float(np.median(np.abs(w - med)))
    return med, s


def outlier_mask(values, k: float = 8.0, log_space: bool = False) -> np.ndarray:
    """Boolean mask (aligned to input) — True for points OUTSIDE [med - k·s, med + k·s].
    Non-finite (and non-positive when log_space) positions are always False.
    Empty / degenerate / zero-MAD input -> all-False."""
    v = np.asarray(values, dtype=float)
    out = np.zeros(v.shape, dtype=bool)
    finite = np.isfinite(v)
    if log_space:
        finite = finite & (v > 0)
    fv = v[finite]
    if fv.size == 0:
        return out
    w = np.log10(fv) if log_space else fv
    med, s = _center_scale(w)
    if s == 0:
        return out
    out[finite] = (w < med - k * s) | (w > med + k * s)
    return out


def robust_range(values, k: float = 8.0, log_space: bool = False):
    """(lo, hi) y-limits for the bulk: lo = max(min(v), med - k·s); hi = med + k·s
    (hi intentionally NOT clamped to max(v) so it can be tighter than a tail).
    lo = max(min(v), med - k·s): the robust lower bound, but never below the data minimum
    (so a low tail is clipped symmetrically with the high tail, while the view never shows
    empty space below the data). hi is likewise free to clip a high tail.
    Empty/all-nan -> (nan, nan). Zero-MAD -> (min, max)."""
    fv = _finite(values)
    if log_space:
        fv = fv[fv > 0]
    if fv.size == 0:
        return (float("nan"), float("nan"))
    w = np.log10(fv) if log_space else fv
    med, s = _center_scale(w)
    if s == 0:
        return (float(fv.min()), float(fv.max()))
    if log_space:
        lo = max(float(fv.min()), float(10.0 ** (med - k * s)))
        hi = float(10.0 ** (med + k * s))
    else:
        lo = max(float(fv.min()), float(med - k * s))
        hi = float(med + k * s)
    return (lo, hi)


def outlier_stats(values, k: float = 8.0) -> dict:
    """Diagnostic payload. Auto-selects log space when the robust decade span > 2.
    All keys always present; never raises."""
    v = np.asarray(values, dtype=float)
    fv = _finite(v)
    n = int(fv.size)
    span = robust_decade_span(v)
    log_space = span > _DECADE_LOG_THRESHOLD
    if n == 0:
        return {"n": 0, "n_outliers": 0, "fraction": 0.0,
                "robust_range": [float("nan"), float("nan")],
                "max_over_median": float("nan"), "decade_span": span,
                "log_space": log_space, "outlier_indices": []}
    mask = outlier_mask(v, k=k, log_space=log_space)
    n_outliers = int(mask.sum())
    lo, hi = robust_range(v, k=k, log_space=log_space)
    med = float(np.median(fv))
    mx = float(np.max(fv))
    return {"n": n, "n_outliers": n_outliers, "fraction": n_outliers / n,
            "robust_range": [lo, hi],
            # deliberately ALWAYS the linear max/median severity ratio (not switched to log space)
            "max_over_median": (mx / med) if med > 0 else float("nan"),
            "decade_span": span, "log_space": bool(log_space),
            "outlier_indices": np.nonzero(mask)[0].astype(int).tolist()}
