"""Resistive superconducting-transition detector (pure numpy; Qt/matplotlib-free).

Standard resistive criteria on a rho(T) ramp: normal-state level rho_N = median rho over the
top-20%-of-T plateau (finite rho>0, >=5 pts). Gated: results ONLY when the ramp actually goes
superconducting — min rho over the bottom 10% of the T-range < 2% of rho_N and rho_N above the
noise floor. Tc criteria are the first upward crossings from the low-T side (robust to
re-entrant noise) of 90% (onset) / 50% (mid, THE Tc) / 10% (zero) of rho_N, linearly
interpolated between samples. Deterministic — no RNG, stable sort.

NARROWNESS gate (integrity): the drop-floor gate alone false-positives on clean non-SC metals
whose steep Bloch-Grüneisen falloff (high RRR, low Θ_D) coasts below 2%·rho_N without any
transition. A real resistive transition is NARROW, so we DECLINE (return None) when the
onset→zero width exceeds 0.5×tc_mid (grossly broad = the whole ramp, not a transition). The
existing CV / drop-point-count machinery still flags moderately broad but genuine cases as
low-confidence."""
from __future__ import annotations

import numpy as np

_PLATEAU_FRAC = 0.20      # top fraction of the T-range defining the normal-state plateau
_DROP_FRAC = 0.10         # bottom fraction of the T-range probed for the drop
_DROP_GATE = 0.02         # min rho in the drop window must fall below this fraction of rho_N
_RHO_NOISE_FLOOR = 1e-9   # Ohm*cm; a "normal state" below this is noise
_MIN_PLATEAU_PTS = 5
_MIN_DROP_PTS = 5         # fewer points across onset..zero -> low confidence
_PLATEAU_CV_MAX = 0.20    # plateau std/median above this -> low confidence
_MAX_REL_WIDTH = 0.50     # (onset-zero)/mid above this -> not a transition (decline); a real
                          # resistive transition is narrow, a clean-metal falloff spans the ramp
_CRITERIA = (("tc_onset_k", 0.90), ("tc_mid_k", 0.50), ("tc_zero_k", 0.10))


def detect_resistive_tc(temperature, rho) -> dict | None:
    """Per-ramp resistive Tc. Returns dict(tc_onset_k, tc_mid_k, tc_zero_k, tc_rho_normal,
    tc_low_confidence) or None when no superconducting drop is present."""
    T = np.asarray(temperature, float)
    R = np.asarray(rho, float)
    m = np.isfinite(T) & np.isfinite(R)
    T, R = T[m], R[m]
    if T.size < 2:
        return None
    order = np.argsort(T, kind="stable")
    T, R = T[order], R[order]
    lo_t, hi_t = float(T[0]), float(T[-1])
    span = hi_t - lo_t
    if span <= 0:
        return None
    plateau = R[(T >= hi_t - _PLATEAU_FRAC * span) & (R > 0)]
    if plateau.size < _MIN_PLATEAU_PTS:
        return None
    rho_n = float(np.median(plateau))
    if rho_n <= _RHO_NOISE_FLOOR:
        return None
    drop = R[T <= lo_t + _DROP_FRAC * span]
    if drop.size == 0 or float(np.min(drop)) >= _DROP_GATE * rho_n:
        return None
    out: dict = {"tc_rho_normal": rho_n}
    for key, frac in _CRITERIA:
        out[key] = _first_upward_crossing(T, R, frac * rho_n)
    onset, zero = out["tc_onset_k"], out["tc_zero_k"]
    mid = out["tc_mid_k"]
    cv = float(np.std(plateau)) / rho_n
    # NARROWNESS gate: a grossly broad onset->zero span is a clean-metal falloff, not a
    # transition -> decline. Needs all three crossings resolved and mid>0 to form the ratio.
    # Only fired when the plateau is TRUSTWORTHY (cv <= max): a noisy plateau inflates rho_n and
    # so pushes `onset` into the noise (spuriously broad), which is exactly the case cv already
    # flags low-confidence -> we defer to that rather than decline a genuine noisy transition.
    if (cv <= _PLATEAU_CV_MAX and onset is not None and zero is not None and mid is not None
            and mid > 0 and (onset - zero) > _MAX_REL_WIDTH * mid):
        return None
    n_drop = int(np.count_nonzero(
        (T >= (zero if zero is not None else lo_t))
        & (T <= (onset if onset is not None else hi_t))))
    out["tc_low_confidence"] = bool(n_drop < _MIN_DROP_PTS or cv > _PLATEAU_CV_MAX)
    return out


def _first_upward_crossing(T, R, level):
    """First upward crossing of `level` scanning from low T; linear interpolation between the
    bracketing samples. None when rho never crosses upward (e.g. already above level)."""
    below = R < level
    for i in range(1, T.size):
        if below[i - 1] and not below[i]:
            t0, t1, r0, r1 = float(T[i - 1]), float(T[i]), float(R[i - 1]), float(R[i])
            if r1 == r0:
                return t1
            return t0 + (level - r0) * (t1 - t0) / (r1 - r0)
    return None
