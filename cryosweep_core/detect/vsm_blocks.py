"""Dedicated per-block VSM sweep classifier (Qt/mpl-free — numpy/pandas only).

The generic `segment_sweeps` under-covers real mixed VSM files (verified on VSM_N:
it leaves ~62% of rows unsegmented and returns only 1 of >=5 M(T) ramps). VSM `.dat`
files interleave temperature ramps (T moving, H held) and field loops (H moving, T
held) in one file, often with duplicate-T loops (two M(H) branches at the same
setpoint). This classifier splits EVERY row into a contiguous block tagged
"temperature" (T-sweep) or "field" (H-sweep), so ~100% of rows are covered.

Design: per-row local sweep-activity (centered peak-to-peak normalized by the axis'
setpoint-stability tolerance) picks the moving axis; dwell rows inherit their
neighbour; contiguous same-kind runs become coarse blocks; each coarse block is then
split where its HELD axis leaves its plateau (distinct M(T) field groups, distinct
M(H) temperatures), and field blocks are further split by field direction so the two
branches of a loop are separate blocks (row order preserved). Thresholds below are
module constants with justification; calibrated on VSM_N (mixed), MPMS (pure M(T)
ZFC/FC) and vsm_synth (single M(T) ramp).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# --- module constants (justified; calibrated on the 3 real/synth fixtures) ---
# Local-activity window (rows either side). QD VSM logs ~1 pt/few-sec; a real ramp
# moves monotonically across many rows, so a +-8 row window (17 pts) captures motion
# without smearing across a sweep-type boundary.
_WIN = 8
# Setpoint-stability tolerances: QD holds T to << 0.5 K while sweeping field, and
# holds field to ~1 Oe (<< 50 Oe) while ramping T. A moving axis blows far past its
# tolerance; a held axis stays under it. These are the drift_max defaults' scale.
_TOL_T = 0.5   # K
_TOL_F = 50.0  # Oe
# Below this normalized activity on BOTH axes a row is a dwell/settle point and
# inherits the surrounding block's kind (avoids spurious 1-row blocks at turnarounds).
_ACT_MIN = 1.0
# Contiguous runs shorter than this are transition/settle glitches; absorbed into the
# preceding block to keep 100% coverage without minting spurious blocks.
_MIN_BLOCK = 8
# Held-field plateau tolerance for splitting T-blocks into distinct M(T) field groups:
# absolute floor (Oe) OR a fraction of the setpoint (field groups span decades:
# 100/5000/40000/100000 Oe -> always split; within one group field holds to ~1 Oe).
_FIELD_PLATEAU_ABS = 50.0
_FIELD_PLATEAU_REL = 0.1
# Held-temperature plateau tolerance for splitting field blocks by their T setpoint
# (dup-T loops at e.g. 5 K vs 30 K stay separate).
_TEMP_PLATEAU_K = 1.0


@dataclass(frozen=True)
class VSMBlock:
    start: int          # inclusive raw-row index
    end: int            # exclusive raw-row index
    kind: str           # "temperature" | "field"
    setpoint: float     # held-axis setpoint (field Oe for T-blocks; temperature K for field-blocks)


def _row_labels(T: np.ndarray, F: np.ndarray) -> list[str]:
    n = len(T)
    actT = np.zeros(n)
    actF = np.zeros(n)
    for i in range(n):
        a = max(i - _WIN, 0)
        b = min(i + _WIN + 1, n)
        st = T[a:b]
        st = st[np.isfinite(st)]
        sf = F[a:b]
        sf = sf[np.isfinite(sf)]
        actT[i] = (np.ptp(st) / _TOL_T) if st.size else 0.0
        actF[i] = (np.ptp(sf) / _TOL_F) if sf.size else 0.0
    lab: list[str | None] = [None] * n
    for i in range(n):
        if actT[i] < _ACT_MIN and actF[i] < _ACT_MIN:
            lab[i] = None  # dwell/settle
        else:
            lab[i] = "temperature" if actT[i] >= actF[i] else "field"
    # forward-fill then back-fill dwell rows to the nearest decided kind
    last = None
    for i in range(n):
        if lab[i] is None:
            lab[i] = last
        else:
            last = lab[i]
    last = None
    for i in range(n - 1, -1, -1):
        if lab[i] is None:
            lab[i] = last
        else:
            last = lab[i]
    # any remaining None (all-dwell frame) -> "temperature" default
    return [x if x is not None else "temperature" for x in lab]


def _coarse_blocks(lab: list[str], n: int) -> list[list]:
    blocks = []
    s = 0
    for i in range(1, n):
        if lab[i] != lab[s]:
            blocks.append([s, i, lab[s]])
            s = i
    blocks.append([s, n, lab[s]])
    return blocks


def _plateau_split(held: np.ndarray, tol_fn) -> list[tuple[int, int]]:
    """Split index range where the held axis leaves its running-median plateau."""
    n = len(held)
    if n == 0:
        return []
    out = []
    start = 0
    setp = held[start]
    for i in range(1, n):
        if np.isfinite(held[i]) and np.isfinite(setp) and abs(held[i] - setp) > tol_fn(setp):
            out.append((start, i))
            start = i
            setp = held[i]
        else:
            seg = held[start:i + 1]
            seg = seg[np.isfinite(seg)]
            if seg.size:
                setp = float(np.median(seg))
    out.append((start, n))
    return out


def _direction_runs(x: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous monotone-direction runs over x (point index ranges, end-exclusive)."""
    n = len(x)
    if n <= 1:
        return [(0, n)]
    d = np.sign(np.diff(x))
    runs = []
    start = 0
    cur = 0
    for i, s in enumerate(d):
        if s != 0 and cur != 0 and s != cur:
            runs.append((start, i + 1))
            start = i + 1
        if s != 0:
            cur = s
    runs.append((start, n))
    return runs


def classify_vsm_blocks(df, cmap, cfg=None) -> list[VSMBlock]:
    """Classify every row of a VSM frame into contiguous T-sweep / H-sweep blocks.

    Returns blocks in row order covering ~100% of rows. `cfg` is accepted for API
    symmetry with `segment_sweeps` but the thresholds are module constants (VSM-tuned).
    """
    T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    F = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
    n = len(T)
    if n == 0:
        return []
    lab = _row_labels(T, F)
    coarse = _coarse_blocks(lab, n)
    raw: list[list] = []
    for s, e, kind in coarse:
        if kind == "temperature":
            held = F[s:e]
            tol = lambda sp: max(_FIELD_PLATEAU_ABS, _FIELD_PLATEAU_REL * abs(sp))
            for a, b in _plateau_split(held, tol):
                raw.append([s + a, s + b, kind])
        else:
            held = T[s:e]
            tol = lambda sp: _TEMP_PLATEAU_K
            for a, b in _plateau_split(held, tol):
                seg = F[s + a:s + b]
                for da, db in _direction_runs(seg):
                    raw.append([s + a + da, s + a + db, kind])
    # absorb sub-_MIN_BLOCK runs (transition rows) into the preceding block
    merged: list[list] = []
    for blk in raw:
        if merged and (blk[1] - blk[0]) < _MIN_BLOCK:
            merged[-1][1] = blk[1]
        else:
            merged.append(blk)
    # re-merge adjacent same-kind blocks that survived with matching held setpoints
    out: list[VSMBlock] = []
    for s, e, kind in merged:
        held = (F[s:e] if kind == "temperature" else T[s:e])
        held = held[np.isfinite(held)]
        sp = float(np.median(held)) if held.size else float("nan")
        out.append(VSMBlock(start=s, end=e, kind=kind, setpoint=sp))
    return out


def ramps_from_temps(temps, min_len: int = 3) -> list[dict]:
    """Partition a 1-D temperature array (the POST-FILTER exported array) into
    contiguous monotone ramps. Returns [{direction, i0, i1}, ...] where i0/i1 are
    INCLUSIVE point indices into the passed (compacted) array. warming = T rising.

    Operates purely on the array handed in, so callers pass temp[keep] and the indices
    are automatically in post-filter coordinates (no off-by-mask hazard).
    """
    x = np.asarray(temps, float)
    n = x.size
    if n == 0:
        return []
    if n == 1:
        return [{"direction": "warming", "i0": 0, "i1": 0}]
    # point-index runs from direction changes (end-exclusive), then to inclusive spans
    runs = [(a, b - 1) for (a, b) in _direction_runs(x)]  # inclusive point spans
    # merge runs shorter than min_len into the previous run (noise turnarounds)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and (b - a + 1) < min_len:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    spans = []
    for a, b in merged:
        seg = x[a:b + 1]
        warming = float(np.sum(np.sign(np.diff(seg)))) >= 0.0
        spans.append(["warming" if warming else "cooling", int(a), int(b)])
    # coalesce adjacent runs of identical direction (single-diff noise turnarounds
    # inside one physical ramp would otherwise mint a spurious extra ramp).
    coalesced: list[list] = []
    for d, a, b in spans:
        if coalesced and coalesced[-1][0] == d:
            coalesced[-1][2] = b
        else:
            coalesced.append([d, a, b])
    return [{"direction": d, "i0": a, "i1": b} for d, a, b in coalesced]
