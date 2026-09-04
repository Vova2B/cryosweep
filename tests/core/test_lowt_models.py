import numpy as np
import pytest
from cryosweep_core.fitting.heat_capacity import fit_lowt_models, DebyeLowTModel, debye_temp_from_beta


def _debye_data(gamma=0.01, beta=5e-4, n=20, tmax=10.0):
    T = np.linspace(2.0, tmax, n)
    cp = gamma * T + beta * T**3                      # Cp/T = gamma + beta*T^2 exactly
    return T, cp


def test_debye_t3_model_key_renamed():
    T, cp = _debye_data()
    fr = DebyeLowTModel(n_atoms=2).fit(T, cp)
    assert fr.model == "debye_t3"


def test_clean_debye_chooses_debye_t3_by_parsimony():
    # all models fit a clean Debye curve, but parsimony must pick the simplest.
    T, cp = _debye_data(gamma=0.01, beta=5e-4, n=20)
    out = fit_lowt_models(T, cp, n_atoms=2)
    assert out["chosen_key"] == "debye_t3"
    assert out["chosen"].params["gamma"] == pytest.approx(0.01, rel=1e-3)
    assert out["chosen"].params["beta"] == pytest.approx(5e-4, rel=1e-3)
    assert len(out["fits"]) == 4
    assert all(f["ok"] for f in out["fits"])           # clean data -> all four succeed


def test_t5_curvature_escalates_past_debye_t3():
    # add a real T^5 term so Debye T^3 cannot reach the parsimony threshold.
    T = np.linspace(2.0, 10.0, 25)
    cp = 0.01 * T + 5e-4 * T**3 + 8e-6 * T**5          # Cp/T = g + b T^2 + d T^4
    out = fit_lowt_models(T, cp, n_atoms=2, parsimony_r2=0.999)
    assert out["chosen_key"] in ("debye_t3_t5", "spin_fluct_noninteracting", "spin_fluct_weak")
    assert out["chosen_key"] != "debye_t3"


def test_spin_fluctuation_recovers_upturn():
    # Cp/T = g + b T^2 + A T^2 ln(T0/T) with negative A -> low-T upturn; Debye T^3 fits poorly.
    T = np.linspace(2.0, 10.0, 25); x = T**2
    cp_over_t = 0.05 + (-1.5e-4) * x + (-5e-4) * x * np.log(10.0 / T)
    cp = cp_over_t * T
    out = fit_lowt_models(T, cp, n_atoms=2, parsimony_r2=0.99)
    debye = next(f for f in out["fits"] if f["key"] == "debye_t3")
    spin = next(f for f in out["fits"] if f["key"] == "spin_fluct_noninteracting")
    assert spin["r2"] > debye["r2"]                    # the richer model fits the upturn better
    assert spin["ok"] and spin["r2"] > 0.95


def test_adjusted_r2_fallback_does_not_blindly_pick_most_params():
    # Force the fallback path with an unreachable threshold: when NO model clears
    # parsimony_r2, selection falls back to highest adjusted R^2 (NOT most params).
    rng_x = np.linspace(2.0, 10.0, 16)
    cp = 0.01 * rng_x + 5e-4 * rng_x**3
    cp[0] *= 1.01; cp[-1] *= 0.99                       # tiny perturbation
    out = fit_lowt_models(rng_x, cp, n_atoms=2, parsimony_r2=1.01)   # unreachable -> force fallback
    ok = [f for f in out["fits"] if f["ok"]]
    assert out["chosen_key"] == max(ok, key=lambda f: f["adj_r2"])["key"]
    # the fallback maximizes ADJUSTED R^2, so the most-parameter (4-param) model must NOT win
    most_params = max(ok, key=lambda f: f["n_params"])["key"]
    assert out["chosen_key"] != most_params


def test_min_points_guard_marks_overparam_models_failed():
    T, cp = _debye_data(n=5)                            # 5 points
    out = fit_lowt_models(T, cp, n_atoms=2)
    by = {f["key"]: f for f in out["fits"]}
    assert by["spin_fluct_noninteracting"]["ok"] is False   # 4 params need >= 6 points
    assert by["debye_t3"]["ok"] is True                     # 2 params OK on 5 points


def test_theta_d_nan_when_beta_negative_but_fit_reported():
    T = np.linspace(2.0, 10.0, 20); x = T**2
    cp = (0.2 - 5e-4 * x) * T                           # beta<0 (pure linear, perfect)
    out = fit_lowt_models(T, cp, n_atoms=2)
    f = next(f for f in out["fits"] if f["key"] == "debye_t3")
    assert f["ok"] and np.isnan(f["theta_D"]) and f["params"]["beta"] < 0


# --------------------------------------------------------------------------- #
# Task-2: extended=True  σ + information criteria                              #
# --------------------------------------------------------------------------- #

def _clean_lowt():
    T = np.linspace(2.0, 10.0, 17)
    cp = 0.005 * T + 2.0e-4 * T**3          # Cp/T = 0.005 + 2e-4 T^2 (exact debye_t3)
    return T, cp


def test_extended_off_is_byte_identical():
    T, cp = _clean_lowt()
    a = fit_lowt_models(T, cp, n_atoms=2.0)
    b = fit_lowt_models(T, cp, n_atoms=2.0)
    assert a == b                                    # determinism
    assert "sigma" not in a["fits"][0]               # no extended keys leak when off


def test_extended_on_adds_sigma_and_ic():
    T, cp = _clean_lowt()
    out = fit_lowt_models(T, cp, n_atoms=2.0, extended=True)
    f0 = out["fits"][0]                              # debye_t3
    assert "sigma" in f0 and "gamma" in f0["sigma"]
    assert np.isfinite(f0["aic"]) and np.isfinite(f0["bic"])
    # AICc valid here (n=17, k=2 -> n-k-2=13>0)
    assert f0["aicc"] is not None and np.isfinite(f0["aicc"])


def test_aicc_none_when_too_few_points():
    # 6 points, spin-fluct k=4 -> n-k-2=0 -> AICc None
    T = np.linspace(2.0, 7.0, 6); cp = 0.005 * T + 2.0e-4 * T**3
    out = fit_lowt_models(T, cp, n_atoms=2.0, extended=True)
    spin = next(f for f in out["fits"] if f["key"] == "spin_fluct_noninteracting")
    assert spin["aicc"] is None


def test_default_return_unchanged_keys():
    T, cp = _clean_lowt()
    out = fit_lowt_models(T, cp, n_atoms=2.0)        # default (extended=False)
    for f in out["fits"]:
        assert set(f) == {"key", "label", "ok", "r2", "adj_r2", "params",
                          "theta_D", "n_params", "t2_grid", "cp_over_t_fit"}


# --------------------------------------------------------------------------- #
# Task-3: identifiability checks                                               #
# --------------------------------------------------------------------------- #

def test_identifiable_clean_debye():
    T, cp = _clean_lowt()
    out = fit_lowt_models(T, cp, n_atoms=2.0, extended=True)
    f0 = out["fits"][0]                               # debye_t3, exact fit
    assert f0["identifiable"] is True
    assert f0["identifiability"]["gamma"]["ok"] is True


def test_railed_T0_flagged_nonidentifiable():
    # noisy-free data where the spin-fluct T0 rails to its upper bound (500)
    T = np.linspace(2.0, 10.0, 17)
    cp = 0.005 * T + 2.0e-4 * T**3                    # pure lattice -> spin term degenerate
    out = fit_lowt_models(T, cp, n_atoms=2.0, extended=True)
    spin = next(f for f in out["fits"] if f["key"] == "spin_fluct_weak")
    # spin model is over-parameterised here -> NOT identifiable (railing or degeneracy)
    assert spin["identifiable"] is False
