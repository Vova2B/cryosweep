"""The fit runs inside the user's temperature window; the curve it produces does not stop there.

A fitted curve that ends at the fit window cannot show what the fit implies outside it — the
gamma intercept at T = 0, or how a Debye-Einstein fit to 20-150 K sits against the data at
300 K. So the Debye-Einstein curve runs from 0 K to the highest data temperature (or a set
limit) and each low-T curve runs from T^2 = 0 to the top of its window, while `fit_range` and
the parallel `in_fit_window` list keep saying exactly which part was fitted.
"""
import csv
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.config import RunConfig
from cryosweep_core.fitting.heat_capacity import (fit_full_range, fit_lowt_models,
                                                  specific_heat_full)
from cryosweep_core.io.export import export_result

_TRUE = {"theta_D": 220.0, "n": 3.0, "gamma": 0.01, "theta_E1": 60.0, "theta_E2": 180.0,
         "m1": 1.0, "m2": 1.5}
_INIT = {"theta_D": 100.0, "n": 3.0, "gamma": 0.007, "theta_E1": 50.0, "theta_E2": 150.0,
         "m1": 1.0, "m2": 2.0}
_FIXED = {k: k == "n" for k in _INIT}


def _data():
    T = np.linspace(2.0, 300.0, 150)
    return T, specific_heat_full(T, **_TRUE)


def _full(**kw):
    T, cp = _data()
    out = fit_full_range(T, cp, init=dict(_INIT), fixed=dict(_FIXED), **kw)
    assert out["ok"], out["reason"]
    return out, T


def _flags(out):
    return np.asarray(out["in_fit_window"], bool)


# ---------------------------------------------------------------- Debye-Einstein

def test_full_curve_runs_from_zero_to_the_highest_data_temperature():
    out, T = _full(fit_min_k=20.0, fit_max_k=150.0)
    g = np.asarray(out["t_grid"]); y = np.asarray(out["cp_fit"])
    assert g[0] == 0.0 and y[0] == 0.0           # the model raises at T<=0: Cp(0)=0 is written
    assert g[-1] == pytest.approx(T.max())
    assert g.size == y.size and np.all(np.isfinite(y))


def test_the_fit_itself_still_uses_only_the_window():
    out, T = _full(fit_min_k=20.0, fit_max_k=150.0)
    inside = T[(T >= 20.0) & (T <= 150.0)]
    assert out["n_points"] == inside.size
    assert out["fit_range"] == [pytest.approx(inside.min()), pytest.approx(inside.max())]


def test_in_fit_window_flips_exactly_at_the_fitted_bounds():
    out, _ = _full(fit_min_k=20.0, fit_max_k=150.0)
    g = np.asarray(out["t_grid"]); f = _flags(out)
    lo, hi = out["fit_range"]
    assert f.size == g.size
    assert lo in g and hi in g                   # the bounds are grid points, not bracketed
    assert np.array_equal(f, (g >= lo) & (g <= hi))
    assert not f[0] and not f[-1] and f.any()


def test_full_grid_is_strictly_ascending_with_no_duplicate():
    """With no window the fit-range top IS the curve top; inserting it again would give two
    CSV rows for one temperature."""
    for kw in ({}, {"fit_min_k": 20.0, "fit_max_k": 150.0}, {"curve_max_k": 150.0}):
        g = np.asarray(_full(**kw)[0]["t_grid"])
        assert np.all(np.diff(g) > 0), kw


def test_curve_limit_can_exceed_the_data():
    out, _ = _full(fit_min_k=20.0, fit_max_k=150.0, curve_max_k=400.0)
    assert out["t_grid"][-1] == pytest.approx(400.0)


def test_curve_limit_never_cuts_into_the_fitted_window():
    out, _ = _full(fit_min_k=20.0, fit_max_k=150.0, curve_max_k=100.0)
    assert out["t_grid"][-1] == pytest.approx(out["fit_range"][1])
    assert _flags(out)[-1]


def test_widening_the_curve_does_not_move_a_fitted_number():
    a, _ = _full(fit_min_k=20.0, fit_max_k=150.0)
    b, _ = _full(fit_min_k=20.0, fit_max_k=150.0, curve_max_k=500.0)
    assert a["params"] == b["params"] and a["r2"] == b["r2"]


def test_the_curve_equals_the_model_wherever_it_is_drawn():
    out, _ = _full(fit_min_k=20.0, fit_max_k=150.0)
    g = np.asarray(out["t_grid"])[1:]
    assert np.allclose(np.asarray(out["cp_fit"])[1:], specific_heat_full(g, **out["params"]))


# ---------------------------------------------------------------- low-T models

def _lowt(lo=3.0, hi=8.0):
    T = np.linspace(2.0, 20.0, 73)
    cp = 0.005 * T + 2.0e-4 * T ** 3
    m = (T >= lo) & (T <= hi)
    return fit_lowt_models(T[m], cp[m], n_atoms=2.0), T[m]


def test_lowt_curves_reach_the_gamma_intercept_and_stop_at_the_window_top():
    fs, Tw = _lowt()
    ok = [f for f in fs["fits"] if f["ok"]]
    assert len(ok) == 4
    for f in ok:
        x = np.asarray(f["t2_grid"]); y = np.asarray(f["cp_over_t_fit"])
        assert x[0] == 0.0 and y[0] == pytest.approx(f["params"]["gamma"])
        assert x[-1] == pytest.approx(Tw.max() ** 2)     # never above the window (owner call)
        assert np.all(np.diff(x) > 0) and np.all(np.isfinite(y))


def test_lowt_fit_range_and_flags_name_the_fitted_part():
    fs, Tw = _lowt()
    for f in (f for f in fs["fits"] if f["ok"]):
        assert f["fit_range"] == [pytest.approx(Tw.min()), pytest.approx(Tw.max())]
        x = np.asarray(f["t2_grid"]); flag = np.asarray(f["in_fit_window"], bool)
        edge = float((Tw ** 2).min())
        assert edge in x                                  # flips AT the bound, same float
        assert np.array_equal(flag, x >= edge)
        assert not flag[0] and flag[-1]


def test_a_failed_lowt_fit_carries_empty_curve_lists():
    fs = fit_lowt_models(np.array([2.0, 3.0, 4.0]), np.array([0.01, 0.02, 0.03]))
    for f in fs["fits"]:
        if not f["ok"]:
            assert f["t2_grid"] == [] and f["cp_over_t_fit"] == [] and f["in_fit_window"] == []


# ---------------------------------------------------------------- analyzer + export

class _Hdr:
    title = "synthetic"; app_version = None; n_atoms = 3.0


def _analyze(**hc):
    T, cp = _data()
    df = pd.DataFrame({"Sample Temp (Kelvin)": T, "Samp HC (mJ/mole-K)": cp * 1e3,
                       "Field (Oe)": np.zeros_like(T)})
    cfg = RunConfig()
    for k, v in hc.items():
        setattr(cfg.heatcapacity, k, v)
    return HCAnalyzer().analyze(SimpleNamespace(df=df, header=_Hdr(), path=None), cfg)


def test_analyzer_passes_the_curve_limit_through():
    d = _analyze(full_fit_min_k=20.0, full_fit_max_k=150.0, full_curve_max_k=350.0).data
    assert d["full_fit"]["t_grid"][0] == 0.0
    assert d["full_fit"]["t_grid"][-1] == pytest.approx(350.0)


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), 0.0, -5.0])
def test_an_unusable_curve_limit_is_ignored_with_a_warning(bad):
    """`inf` parses as a float in the GUI box and would make the grid NaN."""
    r = _analyze(full_curve_max_k=bad)
    g = r.data["full_fit"]["t_grid"]
    assert g[-1] == pytest.approx(300.0) and np.all(np.isfinite(g))
    assert any("full_curve_max_k" in w for w in r.warnings)


def test_model_curves_csv_spans_the_drawn_curve_and_marks_the_fitted_rows(tmp_path):
    r = _analyze(full_fit_min_k=20.0, full_fit_max_k=150.0, lowt_fit_min_k=3.0,
                 lowt_fit_max_k=20.0)
    out = export_result(r, tmp_path / "hc")
    with open(out["model_curves"], newline="") as f:
        rd = csv.reader(f)
        assert next(rd) == ["model", "x", "y", "in_fit_window"]
        rows = list(rd)
    de = [(float(x), int(w)) for m, x, y, w in rows if m == "full:debye_einstein"]
    assert de[0] == (0.0, 0) and de[-1][0] == pytest.approx(300.0) and de[-1][1] == 0
    lo, hi = r.data["full_fit"]["fit_range"]
    assert [x for x, w in de if w] == [x for x, _ in de if lo <= x <= hi]
    for key in ("debye_t3", "debye_t3_t5"):
        lt = [(float(x), int(w)) for m, x, y, w in rows if m == f"lowt:{key}"]
        assert lt[0] == (0.0, 0) and lt[-1][1] == 1
        assert min(x for x, w in lt if w) == pytest.approx(r.data["temperature"][0] ** 2)


# ---------------------------------------------------------------- figures: axis stays on data

def _narrow_example():
    import pathlib
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.registry import build_default_registry
    cfg = RunConfig.load()
    cfg.heatcapacity.full_fit_min_k = 20.0
    cfg.heatcapacity.full_fit_max_k = 60.0
    ex = pathlib.Path(__file__).parents[2] / "examples" / "heat_capacity.dat"
    return analyze_file(load_dat(str(ex)), cfg, build_default_registry())


@pytest.mark.parametrize("kind", ["cp_vs_t", "hc_full_cp_t"])
def test_an_overshooting_curve_does_not_stretch_the_axis(kind):
    """A 20-60 K fit carried to 300 K reaches ~147 J/mol·K against ~76 measured. The axis
    stays framed on the data; the curve runs off the top of the panel (owner call)."""
    import matplotlib
    matplotlib.use("Agg")
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    r = _narrow_example()
    ymax_data = max(r.data["full_cp"])
    assert max(r.data["full_fit"]["cp_fit"]) > 1.5 * ymax_data     # the premise: it overshoots
    ax = render_kind([r], kind, PlotSpec(), GlobalStyle()).axes[0]
    lo, hi = ax.get_ylim()
    assert ymax_data <= hi < 1.15 * ymax_data
    fit = [ln for ln in ax.lines if ln.get_gid() == "fit"]
    assert fit and max(fit[0].get_xdata()) == pytest.approx(max(r.data["full_temperature"]))


@pytest.mark.parametrize("kind", ["cp_vs_t", "hc_full_cp_t"])
def test_a_user_y_limit_still_wins(kind):
    import matplotlib
    matplotlib.use("Agg")
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    ax = render_kind([_narrow_example()], kind, PlotSpec(ymin=0.0, ymax=200.0),
                     GlobalStyle()).axes[0]
    assert ax.get_ylim() == pytest.approx((0.0, 200.0))
