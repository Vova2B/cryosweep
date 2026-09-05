"""Fit curves extrapolate to their 0-intercept — the DATA stays where it was measured.

Owner request 2026-09-05: "fits are not extrapolated to 0k on figures ... extrapolating to
0 for the fitting curve(s) not data." The intercept at 0 is the parameter the figure
already claims in text (gamma on cp_over_t, theta via the Curie-Weiss line on inverse_chi,
rho0 on the resistivity kinds) — extending the fitted curve makes the claim visible.

The extrapolated segment is a statement about behaviour OUTSIDE the fitted window, so it
must never read as fit: dotted, thinner, half-alpha, gid="fit-extrap" (the fitted portion
keeps gid="fit" and its exact range). Kinds where 0 K is not on the abscissa
(resistivity_arrhenius plots against 1000/T) get NO extrapolation.

Also here: the fit-window shade becomes opt-in (PlotSpec.fit_window_shade, default OFF) —
owner: "it can be useful, but switched off by default".
"""
import numpy as np
import pathlib
import pytest

import matplotlib
matplotlib.use("Agg")

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle

EX = pathlib.Path(__file__).parents[2] / "examples"
_REG = build_default_registry()
_CACHE = {}


def _res(name):
    if name not in _CACHE:
        _CACHE[name] = analyze_file(load_dat(str(EX / name)), RunConfig.load(), _REG)
    return _CACHE[name]


def _fig(name, kind, **spec_kw):
    return render_kind([_res(name)], kind, PlotSpec(**spec_kw), GlobalStyle())


def _lines(ax, gid):
    return [ln for ln in ax.lines if ln.get_gid() == gid]


# ---------------- cp_over_t: the gamma intercept becomes visible ----------------

def test_cp_over_t_fit_extrapolates_to_zero_and_hits_gamma():
    fig = _fig("heat_capacity.dat", "cp_over_t")
    ax = fig.axes[0]
    ext = _lines(ax, "fit-extrap")
    assert ext, "no extrapolated segment drawn"
    d = _res("heat_capacity.dat").data
    fits = {f["key"]: f for f in d["lowt_fits"] if f.get("ok")}
    assert len(ext) == len(fits)                 # one continuation per drawn fit line
    ln = ext[0]
    x = np.asarray(ln.get_xdata(), float)
    assert x.min() == 0.0                        # reaches the T^2 = 0 axis
    # the intercept at T^2 = 0 IS gamma — the number the annotation prints
    gamma = fits["debye_t3"]["params"]["gamma"]
    deb = [l for l in ext if abs(float(np.asarray(l.get_ydata(), float)[np.argmin(np.asarray(l.get_xdata(), float))]) - gamma) < 1e-9]
    assert deb, "no extrapolated line lands on gamma at T^2=0"


def test_cp_over_t_fitted_portion_is_unchanged():
    fig = _fig("heat_capacity.dat", "cp_over_t")
    ax = fig.axes[0]
    d = _res("heat_capacity.dat").data
    lo = min(x for f in d["lowt_fits"] if f.get("ok") for x in f["t2_grid"])
    for ln in _lines(ax, "fit"):
        assert float(np.asarray(ln.get_xdata(), float).min()) == pytest.approx(lo)


def test_extrapolation_is_visually_distinct_from_the_fit():
    fig = _fig("heat_capacity.dat", "cp_over_t")
    ax = fig.axes[0]
    fits, exts = _lines(ax, "fit"), _lines(ax, "fit-extrap")
    assert fits and exts
    for ln in exts:
        assert ln.get_alpha() is not None and ln.get_alpha() < 1.0
        assert ln.get_linewidth() < min(f.get_linewidth() for f in fits)
        assert ln.get_label().startswith("_")    # never a legend entry


# ---------------- inverse_chi: the Curie-Weiss theta crossing ----------------

def test_inverse_chi_cw_extends_through_the_theta_crossing():
    fig = _fig("magnetization_vsm.dat", "inverse_chi")
    ax = fig.axes[0]
    ext = _lines(ax, "fit-extrap")
    assert ext
    d = _res("magnetization_vsm.dat").data
    p = d["fit"]["params"]
    ln = ext[0]
    x = np.asarray(ln.get_xdata(), float); y = np.asarray(ln.get_ydata(), float)
    assert x.min() == pytest.approx(min(0.0, p["theta"]))    # reaches theta (< 0 here)
    i0 = int(np.argmin(np.abs(x)))                           # the T = 0 sample
    assert y[i0] == pytest.approx(-p["theta"] / p["C"], rel=0.05)
    # the crossing point itself: 1/chi = 0 at T = theta
    assert y[np.argmin(x)] == pytest.approx(0.0, abs=1e-9)


def test_inverse_chi_modified_cw_extension_stays_finite_above_its_pole():
    d = _res("magnetization_vsm.dat").data
    pm = (d.get("fit_modified") or {}).get("params") or {}
    if "theta" not in pm:
        pytest.skip("no modified CW fit on this file")
    fig = _fig("magnetization_vsm.dat", "inverse_chi")
    ax = fig.axes[0]
    ext = _lines(ax, "fit-extrap")
    assert len(ext) >= 2                     # CW and modified CW both continue
    for ln in ext:
        y = np.asarray(ln.get_ydata(), float)
        assert np.isfinite(y).all()


# ---------------- resistivity: rho0 ----------------

def test_rho_t2_extrapolates_to_rho0():
    r = _res("resistivity_superconductor.dat")
    fig = render_kind([r], "resistivity_rho_t2", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    ext = _lines(ax, "fit-extrap")
    assert ext
    b = next(b for b in r.data["bridges"] if b.get("rho_t2_linear"))
    rho0 = b["rho_t2_linear"]["params"]["rho0"]
    hit = [ln for ln in ext
           if float(np.asarray(ln.get_xdata(), float).min()) == 0.0
           and abs(float(np.asarray(ln.get_ydata(), float)[np.argmin(np.asarray(ln.get_xdata(), float))]) - rho0) < abs(rho0) * 1e-6 + 1e-12]
    assert hit, "no extrapolated line lands on rho0 at T^2=0"


def test_rho_t_headline_extrapolation_reaches_zero():
    fig = _fig("resistivity_superconductor.dat", "resistivity_rho_t")
    ax = fig.axes[0]
    ext = _lines(ax, "fit-extrap")
    assert ext
    assert min(float(np.asarray(ln.get_xdata(), float).min()) for ln in ext) == 0.0


# ---------------- exclusions and guards ----------------

def test_arrhenius_kind_gets_no_extrapolation():
    # x = 1000/T: T = 0 sits at x = infinity — "extrapolate to 0 K" is meaningless there
    fig = _fig("resistivity_semiconductor.dat", "resistivity_arrhenius")
    ax = fig.axes[0]
    assert not _lines(ax, "fit-extrap")


def test_extrap_guard_drops_blowup_points():
    from cryosweep_core.plotting.render import _extrap_plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    x = np.linspace(0, 1, 5)
    y = np.array([1e9, 2.0, 1.5, 1.2, 1.0])     # pole-like first point
    ln = _extrap_plot(ax, x, y, GlobalStyle(), "C0", y_ref=np.array([1.0, 2.0]))
    kept = np.asarray(ln.get_ydata(), float)
    assert np.nanmax(np.abs(kept[np.isfinite(kept)])) <= 3 * 2.0
    plt.close(fig)


def test_extrapolation_does_not_blow_up_the_y_axis():
    # the y-span with extrapolation must stay comparable to the data's own span
    fig = _fig("heat_capacity.dat", "cp_over_t")
    ax = fig.axes[0]
    data = [np.asarray(ln.get_ydata(), float) for ln in ax.lines
            if ln.get_gid() not in ("refline", "fit", "fit-extrap")]
    dmin = min(a.min() for a in data); dmax = max(a.max() for a in data)
    span = dmax - dmin
    lo, hi = ax.get_ylim()
    assert (hi - lo) < 2.0 * span


# ---------------- the fit-window shade is opt-in, default OFF ----------------

def _spans(ax):
    # axvspan draws a Rectangle patch (gid="refline" per _hc_fit_window_shade)
    return [p for p in ax.patches if p.get_gid() == "refline"]

@pytest.mark.parametrize("name,kind", [
    ("heat_capacity.dat", "cp_over_t"),
    ("heat_capacity.dat", "hc_c_over_t_linear"),
    ("resistivity_superconductor.dat", "resistivity_rho_t"),
])
def test_fit_window_shade_defaults_off_and_is_recoverable(name, kind):
    ax = render_kind([_res(name)], kind, PlotSpec(), GlobalStyle()).axes[0]
    assert not _spans(ax), "shade drawn although fit_window_shade defaults OFF"
    ax = render_kind([_res(name)], kind, PlotSpec(fit_window_shade=True), GlobalStyle()).axes[0]
    assert _spans(ax), "opt-in shade did not come back"
