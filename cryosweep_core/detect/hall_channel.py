from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd
from cryosweep_core.analyzers.hall import _antisymmetrize

_CLEAR_FLOOR = 0.05      # absolute odd-fraction floor for "a real Hall signal exists"
_CLEAR_RATIO = 2.0       # winner must be >= this multiple of the runner-up
_MIN_FINITE = 10         # a channel needs at least this many finite points to be scored


def _clear_winner(scored):
    """scored = list of (odd_fraction, channel). Return (channel, frac) iff the top is a clear
    winner (>= floor AND >= ratio x runner-up), else None."""
    if not scored:
        return None
    scored = sorted(scored, reverse=True)
    top_f, top_ch = scored[0]
    second = scored[1][0] if len(scored) > 1 else 0.0
    if top_f >= _CLEAR_FLOOR and top_f >= _CLEAR_RATIO * second:
        return (int(top_ch), float(top_f))
    return None


def _field_segments_by_T(segs):
    by_T = defaultdict(list)
    for s in segs:
        if s.swept.name != "field":
            continue
        T = s.setpoint.get("temperature")
        if T is not None:
            by_T[round(float(T), 1)].append(s)
    return by_T


def _odd_fraction(df, cmap, by_T, ch):
    """Mean over field loops of mean|odd| / (mean|odd| + mean|even|) for resistance_ch{ch}.
    None if the channel is absent, all-NaN/too-sparse, or has no antisymmetrizable loop."""
    key = f"resistance_ch{ch}"
    if key not in cmap.logical:
        return None
    H = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
    R = pd.to_numeric(df[cmap.logical[key]], errors="coerce").to_numpy(float)
    if np.isfinite(R).sum() < _MIN_FINITE:
        return None
    fracs = []
    for segs in by_T.values():
        idx = np.concatenate([s.idx for s in segs])
        Hh, Rr = H[idx], R[idx]
        Hp, R_asym = _antisymmetrize(Hh, Rr)
        if Hp.size < 2:
            continue
        m = np.isfinite(Hh) & np.isfinite(Rr)
        Hs, Rs = Hh[m], Rr[m]
        order = np.argsort(Hs)
        Hs, Rs = Hs[order], Rs[order]
        even = (np.interp(Hp, Hs, Rs) + np.interp(-Hp, Hs, Rs)) / 2.0
        denom = float(np.mean(np.abs(R_asym)) + np.mean(np.abs(even)))
        if denom > 0:
            fracs.append(float(np.mean(np.abs(R_asym)) / denom))
    return float(np.mean(fracs)) if fracs else None


def detect_longitudinal_channel(df, cmap, segs, hall_channel):
    """Pick the longitudinal (even-in-B) companion bridge for mobility: a channel with data
    that isn't the Hall channel; lowest odd fraction wins (most even-in-B), lowest channel
    number breaks ties and covers files without field loops. Returns int channel or None."""
    by_T = _field_segments_by_T(segs)
    cands = []
    for ch in range(1, 5):
        if ch == hall_channel:
            continue
        key = f"resistance_ch{ch}"
        if key not in cmap.logical:
            continue
        R = pd.to_numeric(df[cmap.logical[key]], errors="coerce").to_numpy(float)
        if np.isfinite(R).sum() < _MIN_FINITE:
            continue
        f = _odd_fraction(df, cmap, by_T, ch) if by_T else None
        cands.append((f if f is not None else 1.0, ch))
    return int(sorted(cands)[0][1]) if cands else None


def detect_hall_channel(df, cmap, segs):
    """Odd-in-B Hall-channel detection over field-sweep loops. Returns (channel, odd_fraction)
    on a clear winner, else None. Qt-free, matplotlib-free, sign-agnostic."""
    by_T = _field_segments_by_T(segs)
    if not by_T:
        return None
    scored = []
    for ch in range(1, 5):
        f = _odd_fraction(df, cmap, by_T, ch)
        if f is not None:
            scored.append((f, ch))
    return _clear_winner(scored)


def hall_field_sweep_applicable(cmap, segs) -> bool:
    """True iff the file structurally supports field-sweep Hall: >=2 resistance channels +
    field + temperature columns AND >=1 field-sweep segment."""
    rcols = [k for k in cmap.logical if k.startswith("resistance_ch")]
    if len(rcols) < 2 or "field" not in cmap.logical or "temperature" not in cmap.logical:
        return False
    return any(s.swept.name == "field" for s in segs)


def hall_tdep_applicable(segs) -> bool:
    """True iff temp-ramp segments exist at >=1 opposite-sign field pair (a positive AND a
    negative held field). A held '-0.0'/0 never counts (it is neither >0 nor <0)."""
    fields = set()
    for s in segs:
        if s.swept.name != "temperature":
            continue
        f = s.setpoint.get("field")
        if f is not None:
            fields.add(round(float(f), 0))
    return any(f > 0 for f in fields) and any(f < 0 for f in fields)
