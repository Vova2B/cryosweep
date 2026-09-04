from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from cryosweep_core.primitives import normalized_span, robust_slope
from cryosweep_core.model import Axis, Segment

DEFAULT_WHITELIST = ("temperature", "field", "angle", "frequency", "current")

@dataclass(frozen=True)
class Block:
    start: int
    end: int
    swept_axis: str

def _axis_tol(axis, cfg):
    return cfg.stability.drift_max.get(axis, 1.0)

def _col(df, cmap, axis):
    return pd.to_numeric(df[cmap.logical[axis]], errors="coerce").to_numpy(float)

def _candidate_axes(cmap, whitelist):
    return [a for a in whitelist if a in cmap.logical]

def find_blocks(df, cmap, cfg, whitelist=DEFAULT_WHITELIST):
    axes = _candidate_axes(cmap, whitelist)
    n = len(df)
    if n == 0 or not axes:
        return []
    w = cfg.stability.window
    cols = {a: _col(df, cmap, a) for a in axes}
    # per-row rolling activity = local peak-to-peak (centered window) / drift_max
    act = np.zeros((len(axes), n))
    for k, a in enumerate(axes):
        v = cols[a]; tol = _axis_tol(a, cfg)
        for i in range(n):
            seg = v[max(i - w, 0):min(i + w, n)]
            seg = seg[np.isfinite(seg)]
            act[k, i] = (np.ptp(seg) / tol) if seg.size else 0.0
    winner = np.argmax(act, axis=0)
    maxact = np.max(act, axis=0)
    labels = np.where(maxact >= cfg.stability.activity_min, winner, -1)
    last = -1
    for i in range(n):                       # forward-fill settling rows
        if labels[i] == -1:
            labels[i] = last
        else:
            last = labels[i]
    if (labels == -1).all():
        return []
    # group consecutive identical labels
    raw = []
    start = 0
    for i in range(1, n):
        if labels[i] != labels[start]:
            raw.append((start, i, labels[start])); start = i
    raw.append((start, n, labels[start]))
    blocks = []
    for s, e, lab in raw:
        if lab < 0 or e - s < cfg.stability.min_segment_len:
            continue
        if _span_dominates(cols, axes, s, e, cfg):
            blocks.append(Block(s, e, axes[lab]))
            continue
        # The guard rejected this block: a SECOND axis moves across it almost as much as
        # the winner does. Classically a multi-field M(T) -- several temperature ramps at
        # different held fields, run back to back and merged into one label. The guard is
        # right that the MERGED block is ambiguous; nothing had tried splitting it.
        # Strictly ADDITIVE: only blocks already being discarded reach here, so every file
        # that segments today segments identically.
        blocks.extend(_recover_ambiguous_block(cols, axes, s, e, lab, cfg))
    return blocks


def _span_dominates(cols, axes, s, e, cfg) -> bool:
    """Winner must beat the runner-up axis by span_drift_ratio_min over [s, e)."""
    spans = sorted((normalized_span(cols[a][s:e], _axis_tol(a, cfg)) for a in axes), reverse=True)
    return not (len(spans) > 1 and spans[1] > 0
                and spans[0] / spans[1] < cfg.stability.span_drift_ratio_min)


def _plateau_runs(v, tol, min_len):
    """Maximal index runs over which `v` never steps by more than `tol` between rows.

    A held field reads as one run; a ramp steps every row and yields runs shorter than
    min_len, which are dropped. Non-finite rows never trigger a split (they carry no
    evidence of a setpoint change).
    """
    n = len(v)
    if n == 0:
        return []
    runs, start = [], 0
    for i in range(1, n):
        a, b = v[i - 1], v[i]
        if np.isfinite(a) and np.isfinite(b) and abs(b - a) > tol:
            runs.append((start, i))
            start = i
    runs.append((start, n))
    return [(a, b) for a, b in runs if b - a >= min_len]


def _recover_ambiguous_block(cols, axes, s, e, lab, cfg):
    """Split a guard-rejected block on the confounding axis; keep unambiguous pieces.

    The confounding axis is the runner-up by span -- the one whose motion made the block
    ambiguous. Splitting on ITS plateaus isolates the held-setpoint stretches; each is
    then re-tested against the same guard, so nothing is admitted that the guard would
    not admit on its own.
    """
    win = axes[lab]
    ranked = sorted(axes, key=lambda a: normalized_span(cols[a][s:e], _axis_tol(a, cfg)),
                    reverse=True)
    confounder = next((a for a in ranked if a != win), None)
    if confounder is None:
        return []
    runs = _plateau_runs(cols[confounder][s:e], _axis_tol(confounder, cfg),
                         cfg.stability.min_segment_len)
    out = []
    for a, b in runs:
        s2, e2 = s + a, s + b
        if _span_dominates(cols, axes, s2, e2, cfg):
            out.append(Block(s2, e2, win))
    return out

def select_swept_axis(df, cmap, cfg, whitelist=DEFAULT_WHITELIST):
    axes = _candidate_axes(cmap, whitelist)
    if not axes:                              # Bug 6: no whitelisted axes
        return (None, 0.0, ["no candidate axes"])
    spans = {a: normalized_span(_col(df, cmap, a), _axis_tol(a, cfg)) for a in axes}
    ranked = sorted(spans, key=spans.get, reverse=True)
    swept = ranked[0]; conf = 1.0; warn = []
    if len(ranked) > 1 and spans[ranked[1]] > 0 and spans[swept] / spans[ranked[1]] < cfg.stability.span_drift_ratio_min:
        conf = 0.5; warn.append(f"ambiguous swept axis: {swept} vs {ranked[1]}")
    return swept, conf, warn

def split_by_direction(x, cfg):
    x = np.asarray(x, float)
    sign = np.sign(np.diff(x))
    runs = []; start = 0; cur = 0
    for i, s in enumerate(sign):
        if s != 0 and cur != 0 and s != cur:
            runs.append((start, i + 1)); start = i + 1
        if s != 0:
            cur = s
    runs.append((start, len(x)))
    return [r for r in runs if r[1] - r[0] >= cfg.stability.min_segment_len] or [(0, len(x))]

def segment_sweeps(df, cmap, cfg, whitelist=DEFAULT_WHITELIST):
    blocks = find_blocks(df, cmap, cfg, whitelist)
    if not blocks:                            # generic fallback: whole frame, single swept axis
        swept, conf, _ = select_swept_axis(df, cmap, cfg, whitelist)
        if swept is None:                     # Bug 6: no candidate axes -> no segments
            return []
        blocks = [Block(0, len(df), swept)]
    axes = _candidate_axes(cmap, whitelist)
    segs = []
    for blk in blocks:
        sub = df.iloc[blk.start:blk.end]
        swept = blk.swept_axis
        swept_col = cmap.logical[swept]
        xb = pd.to_numeric(sub[swept_col], errors="coerce").to_numpy(float)
        fixed_axes = [a for a in axes if a != swept]
        for (a, b) in split_by_direction(xb, cfg):
            xs = xb[a:b]
            slope = robust_slope(xs)
            direction = (int(np.sign(slope)) if np.isfinite(slope) else 0) or 0
            branch = {1: "up", -1: "down", 0: None}[direction]
            fixed = {}; setpoint = {}; tol = {}
            reject = False                       # Bug 3a: a "fixed" axis that is actually moving
            for ax in fixed_axes:
                cv_raw = pd.to_numeric(sub.iloc[a:b][cmap.logical[ax]], errors="coerce").to_numpy(float)
                cv = cv_raw[np.isfinite(cv_raw)]
                fixed[ax] = float(np.median(cv)) if cv.size else float("nan")
                setpoint[ax] = fixed[ax]
                tol[ax] = float(np.ptp(cv)) if cv.size else 0.0
                # If a non-swept whitelisted axis drifts as much as a real sweep would, this
                # block is a transition / diagonal (e.g. T resetting while field steps), not a
                # clean sweep. Drop it. Real field-loops hold T to span~0.01 K (kept); real
                # T-ramps hold field to ~1 Oe (kept).
                if normalized_span(cv_raw, _axis_tol(ax, cfg)) >= cfg.stability.span_drift_ratio_min:
                    reject = True
            if reject:
                continue
            segs.append(Segment(
                swept=Axis(name=swept, column=swept_col, unit=cmap.unit.get(swept, "")),
                direction=direction, branch=branch, fixed=fixed, tol=tol, setpoint=setpoint,
                idx=np.arange(blk.start + a, blk.start + b), confidence=1.0,
                x=xs, data={}, normalized=set(),
            ))
    return _merge_consecutive(segs, df, cmap, axes, cfg)


def _merge_consecutive(segs, df, cmap, axes, cfg):
    """Bug 3b: merge consecutive (row-order) segments with the same swept axis, same
    direction, and matching fixed setpoints (within cluster_rel_tol). Splits caused by
    settle gaps inside one physical ramp are stitched back into a single segment."""
    if not segs:
        return segs
    rtol = cfg.stability.cluster_rel_tol
    out = [segs[0]]
    for s in segs[1:]:
        p = out[-1]
        same = (p.swept.name == s.swept.name and p.direction == s.direction)
        if same:
            fixed_axes = [a for a in axes if a != s.swept.name]
            for ax in fixed_axes:
                pa, sa = p.setpoint.get(ax), s.setpoint.get(ax)
                if pa is None or sa is None or np.isnan(pa) or np.isnan(sa):
                    continue
                scale = max(abs(pa), abs(sa), 1e-12)
                if abs(pa - sa) > rtol * scale:
                    same = False
                    break
        if same:
            out[-1] = _stitch(p, s, df, cmap, axes)
        else:
            out.append(s)
    return out


def _stitch(p, s, df, cmap, axes):
    idx = np.concatenate([p.idx, s.idx])
    swept = p.swept.name
    swept_col = cmap.logical[swept]
    xs = pd.to_numeric(df.iloc[idx][swept_col], errors="coerce").to_numpy(float)
    fixed = {}; setpoint = {}; tol = {}
    for ax in [a for a in axes if a != swept]:
        cv = pd.to_numeric(df.iloc[idx][cmap.logical[ax]], errors="coerce").to_numpy(float)
        cv = cv[np.isfinite(cv)]
        fixed[ax] = float(np.median(cv)) if cv.size else float("nan")
        setpoint[ax] = fixed[ax]
        tol[ax] = float(np.ptp(cv)) if cv.size else 0.0
    return Segment(
        swept=Axis(name=swept, column=swept_col, unit=cmap.unit.get(swept, "")),
        direction=p.direction, branch=p.branch, fixed=fixed, tol=tol, setpoint=setpoint,
        idx=idx, confidence=min(p.confidence, s.confidence), x=xs, data={}, normalized=set(),
    )
