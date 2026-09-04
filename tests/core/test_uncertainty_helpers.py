"""U6 extraction: shared sigma helpers, byte-identical TTO behavior."""
import math

import numpy as np
import pytest

from cryosweep_core.fitting.uncertainty import MEDIAN_SE, endpoint_sigma, rrr_sigma, straddles_threshold


def test_median_se_is_the_shipped_sqrt_pi_over_2_constant():
    # U6 byte-identity: the constant is copied character-for-character from tto.py, which
    # ships the 5-significant-figure 1.2533 (sqrt(pi/2) = 1.25331414...). The existing TTO
    # oracles (rrr_std 0.01742 real / 0.00793 fixture, test_tto_integrity.py) pin 1.2533 —
    # "upgrading" the precision here would change every emitted rrr_std.
    assert MEDIAN_SE == 1.2533
    assert MEDIAN_SE == pytest.approx(math.sqrt(math.pi / 2.0), abs=2e-5)


def test_endpoint_sigma_median_of_k_with_efficiency_penalty():
    T = np.arange(10.0)
    rho = np.ones(10)
    std = np.full(10, 0.2)
    assert endpoint_sigma(T, rho, std, lowest=True, k=5) == pytest.approx(
        MEDIAN_SE * 0.2 / math.sqrt(5))


def test_endpoint_sigma_none_on_all_nonfinite_std():
    T = np.arange(10.0); rho = np.ones(10); std = np.full(10, np.nan)
    assert endpoint_sigma(T, rho, std, lowest=True, k=5) is None


def test_straddles_threshold_parametrized():
    assert straddles_threshold(1.0, 0.05, lo=0.98, hi=1.02)
    assert not straddles_threshold(1.0, 0.001, lo=0.98, hi=1.02)


def test_tto_wrappers_unchanged():
    # the tto module still exposes its wrappers with identical numerics
    from cryosweep_core.analyzers import tto
    T = np.arange(10.0); rho = np.linspace(2.0, 1.0, 10); std = np.full(10, 0.02)
    assert tto._endpoint_sigma(T, rho, std, True) == pytest.approx(
        endpoint_sigma(T, rho, std, lowest=True, k=tto._RRR_K))
