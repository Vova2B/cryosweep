import numpy as np
import pytest
from cryosweep_core.fitting.heat_capacity import fit_schottky, R

G0, B0 = 0.005, 2.0e-4
def _bg(T): return G0 * T + B0 * T ** 3
def _sch(T, f, D):
    z = D / np.asarray(T, float); ez = np.exp(z)
    return f * R * z ** 2 * ez / (1 + ez) ** 2

def test_determined_when_peak_covered():
    T = np.linspace(1.9, 15.0, 60); cp = _bg(T) + _sch(T, 0.3, 8.0)
    s = fit_schottky(T, cp, gamma0=G0, beta0=B0)
    assert s["attempted"] and s["chosen_key"] == "schottky"
    assert s["delta_determined"] is True
    assert s["params"]["Delta"] == pytest.approx(8.0, rel=0.05)

def test_rising_tail_only_not_determined():
    # Delta=60 K -> T_peak~25 K, far above the 15 K window: only the rising tail is seen
    T = np.linspace(1.9, 12.0, 40); cp = _bg(T) + _sch(T, 0.3, 60.0)
    s = fit_schottky(T, cp, gamma0=G0, beta0=B0)
    assert s["peak_covered"] is False
    assert s["delta_determined"] is False
    assert "lower-bound" in s["reason"].lower()

def test_low_field_group_flagged():
    T = np.linspace(1.9, 15.0, 60); cp = _bg(T) + _sch(T, 0.3, 8.0)
    s = fit_schottky(T, cp, gamma0=G0, beta0=B0, is_lowest_field=True)
    assert any("kramers" in w.lower() or "zero" in w.lower() for w in s["warnings"])

def test_f_gt_one_flagged():
    T = np.linspace(1.9, 15.0, 60); cp = _bg(T) + _sch(T, 1.5, 8.0)
    s = fit_schottky(T, cp, gamma0=G0, beta0=B0, f_max=5.0)
    assert s["params"]["f"] > 1.0
    assert any("f >" in w or "TLS" in w for w in s["warnings"])

def test_post_fit_peak_above_window_not_determined():
    # Delta≈60 K -> fitted-peak T_peak ~ Delta/2.4 ~25 K, above the T∈[1.9,12] fit window:
    # the fitted Schottky curve's maximum falls outside the window (a bound, not a resolved peak).
    T = np.linspace(1.9, 12.0, 40); cp = _bg(T) + _sch(T, 0.3, 60.0)
    s = fit_schottky(T, cp, gamma0=G0, beta0=B0)
    assert s["delta_determined"] is False
    assert "peak" in s["reason"].lower()

def test_f_railed_not_determined():
    # true f (0.5) exceeds the imposed f_max (0.3) -> the fit rails against the upper bound,
    # so the TLS amplitude (and hence Delta) is not identifiable.
    T = np.linspace(1.9, 15.0, 60); cp = _bg(T) + _sch(T, 0.5, 8.0)
    s = fit_schottky(T, cp, gamma0=G0, beta0=B0, f_max=0.3)
    assert s["delta_determined"] is False
    assert "f_not_railed" in s["reason"]
