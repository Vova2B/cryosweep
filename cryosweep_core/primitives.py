from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import linregress, theilslopes

def _finite(values):
    v = np.asarray(values, float)
    return v[np.isfinite(v)]

def normalized_span(values, tol: float) -> float:
    v = _finite(values)
    if v.size == 0:
        return 0.0
    return float(np.ptp(v) / tol) if tol > 0 else float(np.ptp(v))

def monotone_fraction(values) -> float:
    v = _finite(values)
    d = np.diff(v)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    up = np.count_nonzero(d > 0)
    return float(max(up, d.size - up) / d.size)

def robust_slope(values) -> float:
    v = np.asarray(values, float)
    t = np.arange(v.size, dtype=float)
    mask = np.isfinite(v)            # Bug 4: theilslopes returns nan if any input is non-finite
    v, t = v[mask], t[mask]
    if v.size < 2:
        return 0.0
    return float(theilslopes(v, t)[0])

def cluster_setpoints(values, rel_tol: float) -> np.ndarray:
    v = np.asarray(values, float)
    if v.size == 0:                  # Bug 6: empty input must not IndexError
        return np.array([], dtype=int)
    order = np.argsort(v, kind="mergesort")
    labels = np.empty(v.size, dtype=int)
    cur = 0
    labels[order[0]] = 0
    anchor = v[order[0]]
    for i in order[1:]:
        scale = max(abs(anchor), 1e-12)
        if abs(v[i] - anchor) > rel_tol * scale:
            cur += 1
            anchor = v[i]
        labels[i] = cur
    return labels

@dataclass(frozen=True)
class LinFit:
    slope: float
    intercept: float
    sigma_slope: float
    r2: float

def linfit(x, y) -> LinFit:
    r = linregress(np.asarray(x, float), np.asarray(y, float))
    return LinFit(slope=float(r.slope), intercept=float(r.intercept),
                  sigma_slope=float(r.stderr), r2=float(r.rvalue ** 2))
