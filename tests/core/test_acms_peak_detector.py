import numpy as np
from cryosweep_core.analyzers.acms import _detect_chipp_peak


def test_detects_interior_peak_at_known_tf():
    t = np.linspace(1.0, 10.0, 300)
    chipp = 1e-3 + 5e-2 * np.exp(-((t - 4.0) ** 2) / (0.3 ** 2))     # peak at T_f = 4.0
    p = _detect_chipp_peak(t, chipp)
    assert p is not None and abs(p.t_f_k - 4.0) < 0.2


def test_declines_flat_noise():
    rng = np.random.default_rng(3); t = np.linspace(1.0, 10.0, 300)
    chipp = 2.4e-13 + rng.normal(0, 1e-14, 300)
    assert _detect_chipp_peak(t, chipp) is None


def test_ignores_endpoint_rise():
    t = np.linspace(1.0, 10.0, 300)
    chipp = 1e-3 + 1e-2 * t                                          # monotone edge ramp, no interior peak
    assert _detect_chipp_peak(t, chipp) is None
