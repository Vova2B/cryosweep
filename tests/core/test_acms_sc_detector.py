import numpy as np
from cryosweep_core.analyzers.acms import _detect_sc


def _curve(tc_mid=5.0, onset=5.5, chi_n=0.0, chi_low=-1.0, npts=200, noise=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.4, 8.0, npts)
    # smooth diamagnetic step: chi_n above onset -> chi_low below (tc_mid = 50% point)
    width = onset - tc_mid
    chip = chi_low + (chi_n - chi_low) / (1 + np.exp(-(t - tc_mid) / (0.4 * width)))
    chipp = 0.3 * (chi_n - chi_low) * np.exp(-((t - (tc_mid + width)) ** 2) / (0.2 ** 2))
    chip += rng.normal(0, noise * abs(chi_n - chi_low), npts)
    chipp += rng.normal(0, noise * abs(chi_n - chi_low), npts) + 1e-6
    return t, chip, chipp


def test_detects_oracle_tc_five_kelvin():
    t, chip, chipp = _curve(tc_mid=5.0)
    sc = _detect_sc(t, chip, chipp, 0.0, t.max() - t.min())
    assert sc is not None
    assert abs(sc.tc_mid_k - 5.0) < 0.3


def test_declines_on_featureless():
    rng = np.random.default_rng(1); t = np.linspace(0.4, 5.0, 200)
    chip = -4e-12 + rng.normal(0, 2e-14, 200); chipp = 2.4e-13 + rng.normal(0, 1e-14, 200)
    assert _detect_sc(t, chip, chipp, 0.0, t.max() - t.min()) is None


def test_declines_positive_decreasing_chi():
    # ferromagnetic tail +10 -> +2: a decreasing but POSITIVE chi' must NOT fire (§3a(b))
    t = np.linspace(0.4, 8.0, 200)
    chip = 2.0 + 8.0 / (1 + np.exp(-(t - 5.0) / 0.4)); chipp = np.full(200, 1e-6)
    assert _detect_sc(t, chip, chipp, 0.0, t.max() - t.min()) is None


def test_declines_when_field_biased():
    t, chip, chipp = _curve(tc_mid=5.0)
    assert _detect_sc(t, chip, chipp, 100.0, t.max() - t.min()) is None


# --- (c) tilt guard: extrapolated-baseline, no centered-transition dead-band ---

def _frac_below(tc_mid, t):
    return (tc_mid - t.min()) / (t.max() - t.min())


def test_detects_centered_transition_closes_deadband():
    # tc_mid at the CENTER of the T window (frac ~ 0.5): the OLD full-curve sigma_range test
    # wrongly declined this (MAD ~ step/2). Pins the dead-band closed.
    t = np.linspace(0.4, 8.0, 200)
    tc_mid = 0.4 + 0.5 * (8.0 - 0.4)                       # ~4.2 K, dead-center
    assert abs(_frac_below(tc_mid, t) - 0.5) < 0.02
    _, chip, chipp = _curve(tc_mid=tc_mid, onset=tc_mid + 0.5)
    sc = _detect_sc(t, chip, chipp, 0.0, t.max() - t.min())
    assert sc is not None
    assert abs(sc.tc_mid_k - tc_mid) < 0.3


def test_detects_off_center_low():
    # step low in the window (frac ~ 0.25)
    t = np.linspace(0.4, 8.0, 200)
    tc_mid = 0.4 + 0.25 * (8.0 - 0.4)                      # ~2.3 K
    assert abs(_frac_below(tc_mid, t) - 0.25) < 0.02
    _, chip, chipp = _curve(tc_mid=tc_mid, onset=tc_mid + 0.5)
    sc = _detect_sc(t, chip, chipp, 0.0, t.max() - t.min())
    assert sc is not None
    assert abs(sc.tc_mid_k - tc_mid) < 0.3


def test_detects_off_center_high():
    # step high in the window (frac ~ 0.7)
    t = np.linspace(0.4, 8.0, 300)
    tc_mid = 0.4 + 0.7 * (8.0 - 0.4)                       # ~5.7 K
    assert abs(_frac_below(tc_mid, t) - 0.7) < 0.02
    _, chip, chipp = _curve(tc_mid=tc_mid, onset=tc_mid + 0.3, npts=300)
    sc = _detect_sc(t, chip, chipp, 0.0, t.max() - t.min())
    assert sc is not None
    assert abs(sc.tc_mid_k - tc_mid) < 0.3


def test_declines_on_linear_diamagnetic_tilt():
    # strong monotone diamagnetic DRIFT (a straight tilt, no step): chi'_low < 0 and the plateau-
    # to-low drop exceeds 10*sigma, but plain baseline extrapolation explains the low level -> the
    # (c) tilt guard must DECLINE (no false SC on featureless-but-tilted drift).
    rng = np.random.default_rng(3)
    t = np.linspace(0.4, 8.0, 200)
    chip = -2.0 + 0.5 * t + rng.normal(0, 1e-3, 200)      # rises with T; chi'_low < 0
    assert chip[t <= 1.16].mean() < 0                      # low-T level is diamagnetic
    chipp = 0.3 * np.exp(-((t - 5.0) ** 2) / 0.2 ** 2) + 1e-6
    assert _detect_sc(t, chip, chipp, 0.0, t.max() - t.min()) is None
