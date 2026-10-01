"""Curie-Weiss fit windows, and curves that run from 1/chi = 0 to the top of the data.

The user can bound the points the Curie-Weiss (CW) fit and, separately, the modified CW fit
(chi = chi0 + C/(T - theta)) are judged on. The curve each fit produces does not stop at its
window: it starts where it reaches 1/chi = 0 -- at its own theta -- and runs to the highest
data temperature (or a set limit), as ONE line. `fit_range` and the parallel `in_fit_window`
list say which part was fitted.

Where carrying a curve outside its window would draw something false -- a CW line with C <= 0
or with theta inside its own window (1/chi <= 0 on fitted points), a modified curve whose pole
lies inside the window, or a modified theta so far below zero that T >> |theta| is never
reached -- the curve is the window only and says so (`zero_crossing: false` plus a reason).
"""
import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.fitting.models import CurieWeissModel, cw_curve, fit_cw_ladder

_SI_PER_CGS = 4 * np.pi * 1e-6          # chi_SI (m^3/mol) = chi_CGS (emu/(mol*Oe)) * 4 pi 1e-6


def _cw(theta=-10.0, C=0.5, n=150, tmin=2.0, tmax=300.0):
    T = np.linspace(tmin, tmax, n)
    return T, (T - theta) / C


def _mod(theta=-20.0, C=0.8, chi0=2e-3, n=150, tmin=2.0, tmax=300.0):
    T = np.linspace(tmin, tmax, n)
    return T, 1.0 / (chi0 + C / (T - theta))


def _arr(c, k):
    return np.asarray(c[k], float)


# ------------------------------------------------------------------ config

def test_vsm_config_defaults_to_no_window():
    v = RunConfig().vsm
    assert (v.cw_fit_min_k, v.cw_fit_max_k, v.cw_mod_fit_min_k, v.cw_mod_fit_max_k,
            v.cw_curve_max_k) == (None,) * 5


def test_vsm_config_loads_and_is_in_the_schema():
    cfg = RunConfig.load(vsm={"cw_fit_min_k": 150.0, "cw_curve_max_k": 400.0})
    assert cfg.vsm.cw_fit_min_k == 150.0 and cfg.vsm.cw_curve_max_k == 400.0
    props = RunConfig.model_json_schema()["$defs"]["VSMCfg"]["properties"]
    assert {"cw_fit_min_k", "cw_fit_max_k", "cw_mod_fit_min_k", "cw_mod_fit_max_k",
            "cw_curve_max_k"} <= set(props)


# ------------------------------------------------------------------ cw_curve: plain CW

def test_cw_curve_starts_at_theta_with_zero_and_ends_at_top():
    c = cw_curve({"C": 0.5, "theta": -10.0}, "curie_weiss", 50.0, 200.0, 300.0)
    g, y, f = _arr(c, "t_grid"), _arr(c, "inv_chi_fit"), np.asarray(c["in_fit_window"])
    assert g[0] == -10.0 and y[0] == 0.0
    assert g[-1] == 300.0
    assert np.all(np.diff(g) > 0) and np.all(np.isfinite(y))
    assert c["zero_crossing"] is True
    # both window edges are exact grid points and the flag flips AT them
    assert 50.0 in g and 200.0 in g
    assert f[g == 50.0][0] and f[g == 200.0][0]
    assert not f[g < 50.0].any() and not f[g > 200.0].any()
    assert f[(g >= 50.0) & (g <= 200.0)].all()
    assert len(g) == len(y) == len(f)


def test_cw_curve_positive_theta_below_the_window_starts_at_theta():
    c = cw_curve({"C": 0.5, "theta": 30.0}, "curie_weiss", 60.0, 300.0, 300.0)
    assert c["t_grid"][0] == 30.0 and c["inv_chi_fit"][0] == 0.0
    assert c["zero_crossing"] is True


def test_cw_curve_top_never_cuts_into_the_window():
    c = cw_curve({"C": 0.5, "theta": -10.0}, "curie_weiss", 50.0, 200.0, 120.0)
    assert c["t_grid"][-1] == 200.0


@pytest.mark.parametrize("params, why", [
    ({"C": -0.5, "theta": 400.0}, "C_nonpositive"),
    ({"C": 0.5, "theta": 80.0}, "theta_in_window"),
])
def test_cw_curve_declines_the_extension_and_never_draws_negative_inverse_chi(params, why):
    c = cw_curve(params, "curie_weiss", 50.0, 200.0, 300.0)
    g, y = _arr(c, "t_grid"), _arr(c, "inv_chi_fit")
    assert c["zero_crossing"] is False and c["reason"] == why
    assert g.size and g.min() >= 50.0 and g.max() <= 200.0     # the window only
    assert np.all(y > 0)                                         # clipped to 1/chi > 0
    assert all(c["in_fit_window"])


# ------------------------------------------------------------------ cw_curve: modified

def test_modified_curve_starts_at_its_own_theta_and_ends_at_top():
    p = {"C": 0.8, "theta": -20.0, "chi0": 2e-3}
    c = cw_curve(p, "curie_weiss_modified", 2.0, 300.0, 300.0)
    g, y = _arr(c, "t_grid"), _arr(c, "inv_chi_fit")
    assert g[0] == -20.0 and y[0] == 0.0          # evaluated as (T-theta)/(...): exactly 0
    assert g[-1] == 300.0 and np.all(np.isfinite(y)) and np.all(y[1:] > 0)
    assert c["zero_crossing"] is True and c["reason"] is None


def test_modified_curve_with_negative_chi0_stops_before_its_pole():
    # chi0 < 0: the denominator chi0 (T - theta) + C vanishes at T* = theta - C/chi0
    p = {"C": 0.8, "theta": -20.0, "chi0": -2e-3}
    tstar = -20.0 - 0.8 / -2e-3                   # 380 K, above the window, below top
    c = cw_curve(p, "curie_weiss_modified", 2.0, 300.0, 500.0)
    g, y = _arr(c, "t_grid"), _arr(c, "inv_chi_fit")
    assert g[0] == -20.0 and y[0] == 0.0
    assert g.max() < tstar and g.max() > 300.0
    assert np.all(np.isfinite(y)) and np.all(y[1:] > 0)     # never joins +inf to -inf
    assert c["zero_crossing"] is True


def test_modified_curve_with_its_pole_inside_the_window_is_window_only():
    p = {"C": 0.8, "theta": -20.0, "chi0": -5e-3}          # T* = 140 K, inside 2-300 K
    c = cw_curve(p, "curie_weiss_modified", 2.0, 300.0, 300.0)
    y = _arr(c, "inv_chi_fit")
    assert c["zero_crossing"] is False and c["reason"] == "pole_in_window"
    assert np.all(np.isfinite(y)) and np.all(y > 0)
    assert min(c["t_grid"]) >= 2.0 and max(c["t_grid"]) < 140.0


def test_modified_theta_far_below_zero_is_window_only_and_flagged():
    # T >> |theta| is never reached when theta < -T_max(window): no asymptotic regime
    # the real-file magnitudes (theta -2.5e5 K, pole near 667 K)
    p = {"C": 141524210.47, "theta": -250502.16, "chi0": -563.46}
    c = cw_curve(p, "curie_weiss_modified", 2.0, 300.0, 300.0)
    g = _arr(c, "t_grid")
    assert c["zero_crossing"] is False and c["reason"] == "theta_out_of_range"
    assert g.min() >= 2.0 and g.max() <= 300.0


def test_curve_values_are_json_safe():
    for p, m in (({"C": 0.5, "theta": -10.0}, "curie_weiss"),
                 ({"C": 0.8, "theta": -20.0, "chi0": -2e-3}, "curie_weiss_modified")):
        c = cw_curve(p, m, 2.0, 300.0, 1000.0)
        assert np.all(np.isfinite(_arr(c, "inv_chi_fit")))
        assert all(isinstance(v, bool) for v in c["in_fit_window"])


# ------------------------------------------------------------------ ladder inside a window

def test_ladder_skips_rungs_at_or_below_the_window_minimum():
    T, inv = _cw()
    w = T >= 150.0
    _, ladder, spread, _ = fit_cw_ladder(T[w], inv[w])
    # 25/50/100/150 K would all be exact copies of the primary fit
    assert [e["tmin_k"] for e in ladder] == [200.0]
    assert spread is None                        # < 2 rungs: no spread, never 0.0


def test_ladder_without_a_window_keeps_every_rung():
    T, inv = _cw()
    _, ladder, _, _ = fit_cw_ladder(T, inv)
    assert [e["tmin_k"] for e in ladder] == [25.0, 50.0, 100.0, 150.0, 200.0]


# ------------------------------------------------------------------ modified fit unit invariance

def test_modified_fit_is_unit_invariant():
    """Same data in CGS and SI must give the same theta and the same chi0 (converted). Before
    the fix SI chi0 stuck at its 0.0 start: the parameter scale was ~1e-5 of the CGS one."""
    T, inv_cgs = _mod()
    rng = np.random.default_rng(7)
    inv_cgs = inv_cgs * (1.0 + 0.005 * rng.standard_normal(T.size))
    a = CurieWeissModel().fit(T, inv_cgs, "CGS", modified=True)
    b = CurieWeissModel().fit(T, inv_cgs / _SI_PER_CGS, "SI", modified=True)
    assert a.params["theta"] == pytest.approx(-20.0, abs=2.0)
    assert b.params["theta"] == pytest.approx(a.params["theta"], rel=1e-5)
    assert b.params["chi0"] == pytest.approx(a.params["chi0"] * _SI_PER_CGS, rel=1e-4)
    assert b.params["C"] == pytest.approx(a.params["C"] * _SI_PER_CGS, rel=1e-5)
    assert b.sigma["chi0"] == pytest.approx(a.sigma["chi0"] * _SI_PER_CGS, rel=1e-3)
    assert b.sigma["theta"] == pytest.approx(a.sigma["theta"], rel=1e-3)
