# tests/core/test_schottky_fit_models.py
import numpy as np
import pytest
from cryosweep_core.fitting.heat_capacity import (
    _schottky_seed_peak, _fit_schottky_models, R,
)

G0, B0 = 0.005, 2.0e-4

def _bg(T): return G0 * T + B0 * T ** 3
def _sch(T, f, D):                                    # naive form (safe for D<=15, T>=1.9)
    z = D / np.asarray(T, float); ez = np.exp(z)
    return f * R * z ** 2 * ez / (1 + ez) ** 2

def test_seed_finds_interior_peak():
    T = np.linspace(1.9, 15.0, 60)
    cp = _bg(T) + _sch(T, f=0.3, D=8.0)
    s = _schottky_seed_peak(T, cp, G0, B0, fit_max_k=15.0)
    assert s["peak_covered"] is True
    assert s["Delta_seed"] == pytest.approx(8.0, rel=0.25)     # ~2.4*T_star

def test_fit_recovers_delta_and_f_when_peak_covered():
    T = np.linspace(1.9, 15.0, 60)
    cp = _bg(T) + _sch(T, f=0.3, D=8.0)
    s = _schottky_seed_peak(T, cp, G0, B0, fit_max_k=15.0)
    out = _fit_schottky_models(T, cp, s, r=1.0, lattice_t5=False,
                               include_nuclear=False, nuclear_max_tmin_k=2.5,
                               delta_max_k=100.0, f_max=5.0)
    assert out["chosen_key"] == "schottky"
    p = out["models"]["schottky"]["params"]
    assert p["Delta"] == pytest.approx(8.0, rel=0.05)
    assert p["f"] == pytest.approx(0.3, rel=0.05)

def test_flat_background_data_chooses_background():
    T = np.linspace(1.9, 15.0, 60)
    cp = _bg(T)                                                # no anomaly
    s = _schottky_seed_peak(T, cp, G0, B0, fit_max_k=15.0)
    out = _fit_schottky_models(T, cp, s, r=1.0, lattice_t5=False,
                               include_nuclear=False, nuclear_max_tmin_k=2.5,
                               delta_max_k=100.0, f_max=5.0)
    assert out["chosen_key"] == "background"

def test_background_seeds_come_from_debye_t3_not_hardcoded():
    # C2 regression: off-default background (gamma=0.05, beta=5e-4) must reach curve_fit's p0.
    T = np.linspace(1.9, 15.0, 60)
    cp = 0.05 * T + 5e-4 * T ** 3 + _sch(T, 0.3, 8.0)
    s = _schottky_seed_peak(T, cp, 0.05, 5e-4, fit_max_k=15.0)
    out = _fit_schottky_models(T, cp, s, r=1.0, lattice_t5=False, include_nuclear=False,
                               nuclear_max_tmin_k=2.5, delta_max_k=100.0, f_max=5.0)
    assert out["seed"]["gamma"] == pytest.approx(0.05)         # seed wired, not the 0.005 constant
    assert out["seed"]["beta"] == pytest.approx(5e-4)
    assert out["models"]["schottky"]["params"]["Delta"] == pytest.approx(8.0, rel=0.08)

def test_nuclear_model_attempted_only_when_gated():
    T = np.linspace(1.9, 12.0, 50)
    cp = _bg(T) + _sch(T, 0.2, 7.0) + 0.4 / T ** 2
    s = _schottky_seed_peak(T, cp, G0, B0, fit_max_k=15.0)
    # gate OFF -> M2 absent
    off = _fit_schottky_models(T, cp, s, r=1.0, lattice_t5=False, include_nuclear=False,
                               nuclear_max_tmin_k=2.5, delta_max_k=100.0, f_max=5.0)
    assert "schottky_nuclear" not in off["models"]
    # gate ON (T_min=1.9<=2.5) -> M2 present and recovers alphaN
    on = _fit_schottky_models(T, cp, s, r=1.0, lattice_t5=False, include_nuclear=True,
                              nuclear_max_tmin_k=2.5, delta_max_k=100.0, f_max=5.0)
    assert "schottky_nuclear" in on["models"]
    assert on["models"]["schottky_nuclear"]["params"]["alphaN"] == pytest.approx(0.4, rel=0.15)
