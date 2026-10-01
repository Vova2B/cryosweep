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


# ------------------------------------------------------------------ analyzer

import pathlib

from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.io.loader import load_dat

from tests.core.conftest import require_real

_EX = pathlib.Path(__file__).parents[2] / "examples"


def _an(vsm=None, unit_system="CGS", name="magnetization_vsm.dat"):
    rt = load_dat(str(_EX / name))
    return VSMAnalyzer().analyze(rt, RunConfig.load(unit_system=unit_system, vsm=vsm or {}))


def test_no_window_carries_both_curves_from_theta_to_the_data_top():
    r = _an()
    d = r.data
    T = np.asarray(d["temperature"])
    for key, fkey in (("fit_curve", "fit"), ("fit_modified_curve", "fit_modified")):
        c = d[key]
        assert c["zero_crossing"] is True
        assert c["t_grid"][0] == d[fkey]["params"]["theta"] and c["inv_chi_fit"][0] == 0.0
        assert c["t_grid"][-1] == pytest.approx(T.max())
        assert c["fit_range"] == d[fkey]["fit_range"]
        assert c["window_k"] == [None, None]


def test_curve_keys_are_appended_last():
    keys = list(_an().data)
    i = keys.index("mu_eff_spread")
    assert keys[i + 1:i + 3] == ["fit_curve", "fit_modified_curve"]


def test_cw_window_bounds_the_fit_and_its_ladder_but_not_the_data():
    base = _an().data
    d = _an({"cw_fit_min_k": 150.0}).data
    T = np.asarray(d["temperature"])
    assert len(T) == len(base["temperature"])                 # data are not windowed
    assert d["fit"]["fit_range"][0] >= 150.0
    assert d["fit"]["n_points"] == int((T >= 150.0).sum())
    assert all(e["tmin_k"] > d["fit"]["fit_range"][0] for e in d["cw_ladder"] or [])
    # the curve still starts at theta and runs to the top of the data
    c = d["fit_curve"]
    assert c["t_grid"][0] == d["fit"]["params"]["theta"] and c["t_grid"][-1] == T.max()
    g, f = np.asarray(c["t_grid"]), np.asarray(c["in_fit_window"])
    assert not f[g < d["fit"]["fit_range"][0]].any() and f[g == d["fit"]["fit_range"][0]].all()
    assert c["window_k"] == [150.0, None]
    # the modified fit has its own window: untouched
    assert d["fit_modified"]["fit_range"] == base["fit_modified"]["fit_range"]


def test_modified_window_is_separate():
    d = _an({"cw_mod_fit_min_k": 50.0, "cw_mod_fit_max_k": 200.0}).data
    lo, hi = d["fit_modified"]["fit_range"]
    assert lo >= 50.0 and hi <= 200.0
    assert d["fit"]["fit_range"] == _an().data["fit"]["fit_range"]
    assert d["fit_modified_curve"]["window_k"] == [50.0, 200.0]


def test_curve_limit_extends_and_an_invalid_one_is_ignored_with_a_warning():
    assert _an({"cw_curve_max_k": 400.0}).data["fit_curve"]["t_grid"][-1] == 400.0
    r = _an({"cw_curve_max_k": -5.0})
    assert r.data["fit_curve"]["t_grid"][-1] == max(r.data["temperature"])
    assert any("cw_curve_max_k" in w for w in r.warnings)


def test_a_window_that_leaves_too_few_points_keeps_the_data():
    r = _an({"cw_fit_min_k": 299.5})
    assert r.status == "low_confidence"
    assert r.data["fit"] is None and r.data["fit_curve"] is None
    assert len(r.data["temperature"]) == 300                   # arrays, plots, export survive
    assert any("vsm.cw_fit_min_k" in w for w in r.warnings)
    assert r.data["fit_modified"] is not None                  # its own window is unbounded


def test_a_modified_window_that_leaves_too_few_points_is_non_blocking():
    r = _an({"cw_mod_fit_min_k": 299.5})
    assert r.status == "ok" and r.data["fit"] is not None
    assert r.data["fit_modified"] is None and r.data["fit_modified_curve"] is None
    assert any("vsm.cw_mod_fit_min_k" in w for w in r.warnings)


@pytest.mark.parametrize("vsm", [{"cw_fit_min_k": 200.0, "cw_fit_max_k": 100.0},
                                 {"cw_fit_min_k": float("nan")}])
def test_an_invalid_window_is_ignored_with_a_warning(vsm):
    r = _an(vsm)
    base = _an()
    assert r.data["fit"] == base.data["fit"]
    assert any("cw_fit" in w and "ignored" in w for w in r.warnings)


def test_paramagnetic_regime_warning_uses_the_window_minimum():
    msg = "CW fit window extends below |theta|"
    assert any(msg in w for w in _an().warnings)               # 2 K < |theta| = 10 K
    assert not any(msg in w for w in _an({"cw_fit_min_k": 50.0}).warnings)


def test_si_curve_is_in_si_units():
    d = _an(unit_system="SI").data
    c = d["fit_curve"]
    T = np.asarray(d["temperature"]); y = np.asarray(d["inv_chi"])
    i = int(np.argmax(T))
    assert c["inv_chi_fit"][-1] == pytest.approx(y[i], rel=1e-2)


def test_real_file_far_negative_modified_theta_is_flagged_and_kept():
    """No window: theta_mod = -2.5e5 K. Params stay (decline the curve, not the record)."""
    import dataclasses
    rt = load_dat(str(require_real("vsm_mt")))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0,
                                                            mass_mg=5.0))
    d = VSMAnalyzer().analyze(rt, RunConfig.load()).data
    fm = d["fit_modified"]
    assert "theta_out_of_range" in fm["quality_flags"]
    assert fm["params"]["theta"] < -max(d["temperature"])
    c = d["fit_modified_curve"]
    assert c["zero_crossing"] is False and c["reason"] == "theta_out_of_range"
    assert min(c["t_grid"]) >= fm["fit_range"][0]


# ------------------------------------------------------------------ export

import csv
import json

from cryosweep_core.io.export import export_result


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.reader(f))


def test_model_curves_csv_runs_from_theta_to_the_top_with_window_flags(tmp_path):
    r = _an({"cw_fit_min_k": 150.0})
    out = export_result(r, str(tmp_path / "e"))
    rows = _rows(out["model_curves"])
    assert rows[0] == ["model", "x", "y", "in_fit_window"]     # same header as heat capacity
    cw = [x for x in rows[1:] if x[0] == "curie_weiss"]
    mod = [x for x in rows[1:] if x[0] == "curie_weiss_modified"]
    assert cw and mod
    d = r.data
    assert float(cw[0][1]) == pytest.approx(d["fit"]["params"]["theta"]) and float(cw[0][2]) == 0.0
    assert float(cw[-1][1]) == pytest.approx(max(d["temperature"]))
    flags = [x[3] for x in cw]
    first_in = flags.index("1")
    assert float(cw[first_in][1]) == pytest.approx(d["fit"]["fit_range"][0])
    assert set(flags[:first_in]) == {"0"}
    meta = json.loads(pathlib.Path(out["meta"]).read_text())
    assert meta["units"]["model_curves.y"] == d["inv_chi_unit"]
    assert meta["units"]["model_curves.x"] == "K"


def test_derived_csv_carries_the_modified_fit(tmp_path):
    r = _an()
    rows = _rows(export_result(r, str(tmp_path / "e"))["derived"])
    assert rows[0] == ["quantity", "value", "sigma", "unit", "model"]
    mod = {x[0]: x for x in rows[1:] if x[4] == "curie_weiss_modified"}
    pm = r.data["fit_modified"]["params"]
    assert set(pm) <= set(mod)
    assert float(mod["chi0"][1]) == pytest.approx(pm["chi0"])
    assert mod["chi0"][3] == r.data["fit_modified"]["units"]["chi0"]
    assert "quality_flags" not in mod                           # clean fit: no flags row


def test_export_survives_a_fit_lost_to_the_window(tmp_path):
    r = _an({"cw_fit_min_k": 299.5})
    out = export_result(r, str(tmp_path / "e"))
    rows = _rows(out["model_curves"])
    assert {x[0] for x in rows[1:]} == {"curie_weiss_modified"}
    assert _rows(out["fit_params"])[0] == ["param", "value", "sigma", "unit"]
    assert len(_rows(out["points"])) == 301


def test_model_curves_csv_is_header_only_without_curves(tmp_path):
    r = _an({"cw_fit_min_k": 299.5, "cw_mod_fit_min_k": 299.5})
    rows = _rows(export_result(r, str(tmp_path / "e"))["model_curves"])
    assert rows == [["model", "x", "y", "in_fit_window"]]


def test_derived_csv_names_a_flagged_modified_fit(tmp_path):
    import dataclasses
    rt = load_dat(str(require_real("vsm_mt")))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0,
                                                            mass_mg=5.0))
    r = VSMAnalyzer().analyze(rt, RunConfig.load())
    rows = _rows(export_result(r, str(tmp_path / "e"))["derived"])
    flags = [x for x in rows[1:] if x[0] == "quality_flags" and x[4] == "curie_weiss_modified"]
    assert flags and "theta_out_of_range" in flags[0][1]


# ------------------------------------------------------------------ figure

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cryosweep_core.plotting.catalog import OverlayFile, series_inverse_chi
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec


def _fig(r, **kw):
    return render_kind([r], "inverse_chi", PlotSpec(**kw), GlobalStyle())


def _gid(ax, gid):
    return [ln for ln in ax.lines if ln.get_gid() == gid]


def test_each_fit_is_one_solid_line_from_zero_at_theta_to_the_top():
    r = _an({"cw_fit_min_k": 150.0})
    ax = _fig(r).axes[0]
    assert not _gid(ax, "fit-extrap")                  # no dotted continuation
    fits = _gid(ax, "fit")
    assert len(fits) == 2
    for ln, key in zip(fits, ("fit_curve", "fit_modified_curve")):
        c = r.data[key]
        x = np.asarray(ln.get_xdata(), float); y = np.asarray(ln.get_ydata(), float)
        assert x[0] == c["t_grid"][0] and y[0] == 0.0
        assert x[-1] == pytest.approx(max(r.data["temperature"]))
    assert fits[0].get_linestyle() != fits[1].get_linestyle()   # which model, by style
    plt.close("all")


def test_y_axis_is_framed_on_the_data_and_zero():
    r = _an({"cw_curve_max_k": 900.0})                 # the curve overshoots the data 3x
    ax = _fig(r).axes[0]
    lo, hi = ax.get_ylim()
    y = np.asarray(r.data["inv_chi"])
    assert lo <= 0.0 < y.min()                         # 1/chi = 0 at theta is on the frame
    assert hi < 1.2 * y.max()                          # framed on the data, not the curve
    assert ax.get_xlim()[0] <= r.data["fit"]["params"]["theta"]   # negative theta visible
    plt.close("all")


def test_user_y_limits_still_win():
    ax = _fig(_an(), ymin=10.0, ymax=50.0).axes[0]
    assert ax.get_ylim() == pytest.approx((10.0, 50.0))
    plt.close("all")


def test_old_results_without_curves_keep_the_old_drawing():
    r = _an()
    old = r.model_copy(deep=True)
    for k in ("fit_curve", "fit_modified_curve"):
        old.data.pop(k)
    ax = _fig(old).axes[0]
    assert _gid(ax, "fit-extrap")                      # the dotted continuation, as before
    plt.close("all")


def test_fit_series_are_cataloged_but_never_drawn_as_data():
    r = _an()
    fits = [s for s in series_inverse_chi(r) if s.role == "fit"]
    assert {s.key for s in fits} == {"cw_fit", "cw_modified_fit"}
    c = r.data["fit_curve"]
    s = next(s for s in fits if s.key == "cw_fit")
    assert s.x == c["t_grid"] and s.y == c["inv_chi_fit"]
    ax = _fig(r).axes[0]
    marked = [ln for ln in ax.lines if ln.get_marker() not in (None, "None", "", " ")]
    assert all(len(ln.get_xdata()) != len(c["t_grid"]) for ln in marked)
    plt.close("all")


def test_overlay_draws_no_fit_lines():
    a, b = _an(), _an({"cw_fit_min_k": 150.0})
    ov = [OverlayFile(file_id=0, label="a"), OverlayFile(file_id=1, label="b")]
    ax = render_kind([a, b], "inverse_chi", PlotSpec(), GlobalStyle(), overlay=ov).axes[0]
    assert not _gid(ax, "fit") and not _gid(ax, "fit-extrap")
    assert len(ax.lines) == 2                          # one data series per file
    plt.close("all")


def test_window_only_modified_curve_draws_only_its_window():
    r = _an()
    r = r.model_copy(deep=True)
    lo, hi = r.data["fit_modified"]["fit_range"]
    r.data["fit_modified_curve"] = cw_curve(
        {"C": 0.8, "theta": -20.0, "chi0": -5e-3}, "curie_weiss_modified", lo, hi, hi)
    ax = _fig(r).axes[0]
    mod = [ln for ln in _gid(ax, "fit") if ln.get_label() == "modified CW"]
    x = np.asarray(mod[0].get_xdata(), float)
    assert x.min() >= lo
    plt.close("all")
