import numpy as np
import pytest
from cryosweep_core.fitting.transport import LinearFitModel, PowerLawRhoModel

def test_linear_fit_recovers_slope_intercept():
    x = np.linspace(0, 100, 50)
    y = 3.0 * x + 7.0
    fit = LinearFitModel().fit(x, y)
    assert fit.model == "linear"
    assert fit.params["slope"] == pytest.approx(3.0, rel=1e-9)
    assert fit.params["intercept"] == pytest.approx(7.0, abs=1e-7)
    assert fit.r2 == pytest.approx(1.0, abs=1e-12)
    assert fit.n_points == 50

def test_linear_fit_rejects_degenerate_x():
    with pytest.raises(ValueError):
        LinearFitModel().fit([1.0, 1.0, 1.0], [2.0, 3.0, 4.0])

def test_power_law_recovers_rho0_A_n():
    T = np.linspace(2, 60, 80)
    rho = 1.0e-6 + 5.0e-10 * T**2          # Fermi-liquid n=2
    fit = PowerLawRhoModel().fit(T, rho)
    assert fit.model == "power_law_rho"
    assert fit.params["rho0"] == pytest.approx(1.0e-6, rel=5e-3)
    assert fit.params["n"] == pytest.approx(2.0, abs=2e-2)
    assert fit.params["A"] == pytest.approx(5.0e-10, rel=5e-2)
    assert fit.r2 > 0.999

def test_power_law_needs_min_points():
    with pytest.raises(ValueError):
        PowerLawRhoModel().fit([2, 3, 4], [1, 2, 3])

def test_power_law_is_scale_invariant():
    T = np.linspace(2, 60, 80)
    rho = 1.0e-6 + 5.0e-10 * T**2
    a = PowerLawRhoModel().fit(T, rho)
    b = PowerLawRhoModel().fit(T, rho * 1.0e6)     # identical curve shape, 1e6x absolute scale
    assert b.params["n"] == pytest.approx(a.params["n"], rel=1e-6)
    assert b.r2 == pytest.approx(a.r2, rel=1e-6)
    assert b.params["rho0"] == pytest.approx(a.params["rho0"] * 1e6, rel=1e-6)
    assert b.params["A"] == pytest.approx(a.params["A"] * 1e6, rel=1e-6)
    assert b.r2 > 0.999

def test_rho_t2_fermi_liquid_recovers_rho0_beta():
    from cryosweep_core.fitting.transport import RhoT2FermiLiquidModel
    T = np.linspace(2, 30, 40)
    beta, rho0 = 5.0e-9, 1.0e-6
    rho = rho0 + beta * T**2
    fit = RhoT2FermiLiquidModel().fit(T, rho)
    assert fit.model == "rho_t2_linear"
    assert fit.params["rho0"] == pytest.approx(rho0, rel=1e-6)
    assert fit.params["beta"] == pytest.approx(beta, rel=1e-6)
    assert fit.r2 > 0.999
    assert fit.fit_range == pytest.approx([2.0, 30.0])   # stored in K, not K^2

def test_rho_t2_fermi_liquid_rejects_degenerate():
    from cryosweep_core.fitting.transport import RhoT2FermiLiquidModel
    with pytest.raises(ValueError):
        RhoT2FermiLiquidModel().fit([5.0, 5.0, 5.0], [1.0, 2.0, 3.0])
