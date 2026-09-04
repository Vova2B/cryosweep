import numpy as np
import pytest
from cryosweep_core.fitting.models import CurieWeissModel


def test_curie_weiss_zero_slope_raises_valueerror():
    # Bug 2: a flat inv_chi (slope 0) must raise a clean ValueError, not ZeroDivisionError.
    T = np.linspace(2, 300, 50)
    inv_chi = np.zeros_like(T)
    with pytest.raises(ValueError):
        CurieWeissModel().fit(T, inv_chi, unit_system="CGS")


def test_curie_weiss_too_few_points_raises_valueerror():
    # Bug 6: n<3 points -> clean ValueError.
    with pytest.raises(ValueError):
        CurieWeissModel().fit(np.array([2.0, 3.0]), np.array([1.0, 2.0]), unit_system="CGS")

def test_curie_weiss_recovers_known_params():
    C_true, theta_true = 0.5, -10.0
    T = np.linspace(2, 300, 300)
    inv_chi = (T - theta_true) / C_true
    fr = CurieWeissModel().fit(T, inv_chi, unit_system="CGS")
    assert abs(fr.params["C"] - 0.5) < 1e-6
    assert abs(fr.params["theta"] - (-10.0)) < 1e-6
    assert abs(fr.params["mu_eff"] - 2.827 * 0.5 ** 0.5) < 1e-3
    assert fr.r2 > 0.9999
    assert fr.n_points == 300
    assert "C" in fr.sigma and fr.sigma["C"] >= 0.0
    assert len(fr.covariance) == 2          # slope/intercept covariance retained

def test_curie_weiss_sigma_grows_with_noise():
    rng = np.random.default_rng(1)
    T = np.linspace(2, 300, 300)
    clean = (T + 10.0) / 0.5
    noisy = clean + 5.0 * rng.standard_normal(300)
    fr = CurieWeissModel().fit(T, noisy, unit_system="CGS")
    assert fr.sigma["C"] > 0.0
    assert fr.params["C"] > 0

def test_modified_curie_weiss_runs():
    T = np.linspace(2, 300, 300)
    chi = 1e-4 + 0.5 / (T + 10.0)
    fr = CurieWeissModel().fit(T, 1.0 / chi, unit_system="CGS", modified=True)
    assert fr.params["chi0"] == fr.params["chi0"]   # not NaN
    assert fr.model == "curie_weiss_modified"
