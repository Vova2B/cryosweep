import numpy as np
from cryosweep_core.fitting.heat_capacity import (
    debye_heat_capacity, einstein_heat_capacity, debye_temp_from_beta, DebyeLowTModel, R)

def test_debye_dulong_petit_limit():
    # genuine T >> theta_D limit (yi = theta/T = 2e-4) -> 3nR
    assert abs(debye_heat_capacity(1.0e6, 200.0, n=1.0) - 3 * R) < 1e-6

def test_debye_hight_expansion_matches_integral():
    # in the expansion region (yi=0.04) the 2-term form must match the quad integral
    from scipy.integrate import quad
    T, th = 5000.0, 200.0; yi = th / T
    integrand = lambda x: x**4 * np.exp(x) / (np.exp(x) - 1) ** 2
    true = 9.0 * R * (T / th) ** 3 * quad(integrand, 0.0, yi, limit=200)[0]
    assert abs(debye_heat_capacity(T, th, n=1.0) - true) < 1e-4

def test_debye_low_t_cube_limit():
    # T << theta_D -> (12 pi^4/5) nR (T/theta)^3
    T, th = 1.0, 400.0
    exp = (12.0/5.0) * np.pi**4 * R * (T/th)**3
    assert abs(debye_heat_capacity(T, th, n=1.0) - exp) / exp < 1e-3

def test_einstein_high_t_limit():
    # T >> theta_E -> m*3R
    assert abs(einstein_heat_capacity(np.array([5000.0]), 100.0, 1.0)[0] - 3*R) < 0.05

def test_debye_temp_from_beta_matches_formula():
    beta = 1e-4
    exp = (12 * np.pi**4 * 1 * R / (5 * beta))**(1.0/3.0)
    assert abs(debye_temp_from_beta(beta, 1) - exp) < 1e-9
    assert np.isnan(debye_temp_from_beta(-1.0, 1))     # non-physical beta

def test_lowt_model_recovers_gamma_beta():
    # Cp/T = gamma + beta*T^2 exactly -> recover both + theta_D + sigma
    gamma, beta, n = 0.01, 5e-4, 1.0
    T = np.linspace(2.0, 8.0, 60); T2 = T**2
    cp_over_t = gamma + beta * T2
    cp = cp_over_t * T
    fr = DebyeLowTModel(n_atoms=n).fit(T, cp)
    assert abs(fr.params["gamma"] - gamma) < 1e-9
    assert abs(fr.params["beta"] - beta) < 1e-12
    assert abs(fr.params["theta_D"] - debye_temp_from_beta(beta, n)) < 1e-6
    assert "gamma" in fr.sigma and fr.n_points == 60
