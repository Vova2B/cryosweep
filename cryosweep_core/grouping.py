"""Pure-numpy setpoint grouping for resistivity field-segment loops (DQ-B).
No plotting library or GUI toolkit imports. Single source of truth; helpers guard degenerate input and never raise.

Magnitude-aware key: below `threshold` a held temperature is binned to the nearest HALF-integer
(low-T setpoints like 2/5/10/15 K and 4.5 vs 5.0 stay distinct); at/above `threshold` to the
nearest INTEGER (so a drifting high-T hold like 199.9 / 200.0 merges). Rounding is half-UP, not
Python's banker's round() (which would send 12.5 -> 12)."""
from __future__ import annotations
import math
import numpy as np

_DEFAULT_THRESHOLD_K = 10.0


def setpoint_key(value, threshold: float = _DEFAULT_THRESHOLD_K) -> float:
    """Magnitude-aware round-half-up bin for a setpoint value. Non-finite -> nan."""
    v = float(value)
    if not np.isfinite(v):
        return float("nan")
    if v < threshold:
        return math.floor(v * 2.0 + 0.5) / 2.0      # nearest half-integer, half up
    return float(math.floor(v + 0.5))                # nearest integer, half up



def cluster_field_setpoints(values, rel_tol: float = 1e-3, abs_floor: float = 1.0):
    """One label per input value; values belonging to the same cluster share a label.

    Single-link clustering over the SORTED finite values: consecutive values join when their
    gap is within max(abs_floor, rel_tol * max(|a|, |b|)). Each cluster is then labelled by
    `setpoint_key(cluster median)` -- cluster to GROUP, round to LABEL, because the label
    reaches legends and CSV cells and the raw median would render 500 Oe as 499.9 Oe.

    Clustering rather than rounding is deliberate. `setpoint_key` alone bins to the nearest
    integer regardless of magnitude, so on a real multi-field M(T) the same 40 kOe ramp
    arrived as 40000.8870 -> 40001.0 and 39999.5860 -> 40000.0 and was drawn as two curves.
    A coarser fixed bin would not fix that: every bin has edges, and the defect is two values
    straddling one. Clustering the values actually present has no edges.

    Non-finite values pass through as nan and never join a cluster.
    """
    vals = [float(v) for v in values]
    out = [float("nan")] * len(vals)
    finite = [(i, v) for i, v in enumerate(vals) if math.isfinite(v)]
    if not finite:
        return out
    finite.sort(key=lambda t: t[1])
    clusters: list[list[tuple[int, float]]] = [[finite[0]]]
    for idx, v in finite[1:]:
        prev = clusters[-1][-1][1]
        tol = max(abs_floor, rel_tol * max(abs(prev), abs(v)))
        if v - prev <= tol:
            clusters[-1].append((idx, v))
        else:
            clusters.append([(idx, v)])
    for c in clusters:
        label = setpoint_key(float(np.median([v for _i, v in c])))
        for i, _v in c:
            out[i] = label
    return out


def group_segments_by_setpoint(segs, axis: str = "temperature",
                               threshold: float = _DEFAULT_THRESHOLD_K):
    """Group segments by setpoint_key(setpoint[axis]). Returns [(key, [segs]), ...] sorted by key.
    Segments whose setpoint[axis] is missing or non-finite are skipped."""
    buckets: dict[float, list] = {}
    for s in segs:
        raw = s.setpoint.get(axis)
        if raw is None:
            continue
        k = setpoint_key(raw, threshold)
        if not np.isfinite(k):
            continue
        buckets.setdefault(k, []).append(s)
    return [(k, buckets[k]) for k in sorted(buckets)]
