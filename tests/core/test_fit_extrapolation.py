"""Fit curves extrapolate to their 0-intercept — the DATA stays where it was measured.

Owner request 2026-09-05: "fits are not extrapolated to 0k on figures ... extrapolating to
0 for the fitting curve(s) not data." The intercept at 0 is the parameter the figure
already claims in text (gamma on cp_over_t, theta via the Curie-Weiss line on inverse_chi,
rho0 on the resistivity kinds) — extending the fitted curve makes the claim visible.

Where a separate continuation is drawn (resistivity) it must never read as fit: dotted,
thinner, half-alpha, gid="fit-extrap". Heat capacity and Curie-Weiss no longer draw one —
their curves are built over the whole drawn span (see test_hc_curve_span.py and
test_cw_fit_window.py), so the fit line itself reaches 0 as one line. Kinds where 0 K is not on the abscissa (resistivity_arrhenius
plots against 1000/T) get NO extrapolation.

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

def test_cp_over_t_fit_line_reaches_zero_and_hits_gamma():
    """The heat-capacity curves carry their own span since the curve-span change: the fit
    line itself runs to T^2 = 0, as one line, with no separate dotted continuation."""
    fig = _fig("heat_capacity.dat", "cp_over_t")
    ax = fig.axes[0]
    d = _res("heat_capacity.dat").data
    fits = {f["key"]: f for f in d["lowt_fits"] if f.get("ok")}
    lines = _lines(ax, "fit")
    assert len(lines) == len(fits)
    assert not _lines(ax, "fit-extrap")          # one continuous line, not fit + extension
    gamma = fits["debye_t3"]["params"]["gamma"]
    hits = []
    for ln in lines:
        x = np.asarray(ln.get_xdata(), float); y = np.asarray(ln.get_ydata(), float)
        assert x.min() == 0.0                    # reaches the T^2 = 0 axis
        hits.append(abs(float(y[np.argmin(x)]) - gamma) < 1e-9)
    assert any(hits), "no fit line lands on gamma at T^2=0"


def test_cp_over_t_line_stops_at_the_top_of_the_fit_window():
    fig = _fig("heat_capacity.dat", "cp_over_t")
    ax = fig.axes[0]
    d = _res("heat_capacity.dat").data
    top = max(d["t_squared"])
    for ln in _lines(ax, "fit"):
        assert float(np.asarray(ln.get_xdata(), float).max()) == pytest.approx(top)


def test_cp_vs_t_debye_einstein_line_spans_zero_to_the_highest_data_temperature():
    fig = _fig("heat_capacity.dat", "cp_vs_t")
    ax = fig.axes[0]
    d = _res("heat_capacity.dat").data
    (ln,) = _lines(ax, "fit")
    x = np.asarray(ln.get_xdata(), float)
    assert x.min() == 0.0 and x.max() == pytest.approx(max(d["full_temperature"]))


def test_extrapolation_is_visually_distinct_from_the_fit():
    """Still true wherever a separate continuation is drawn (resistivity)."""
    fig = _fig("resistivity_superconductor.dat", "resistivity_rho_t")
    ax = fig.axes[0]
    fits, exts = _lines(ax, "fit"), _lines(ax, "fit-extrap")
    assert fits and exts
    for ln in exts:
        assert ln.get_alpha() is not None and ln.get_alpha() < 1.0
        assert ln.get_linewidth() < min(f.get_linewidth() for f in fits)
        assert ln.get_label().startswith("_")    # never a legend entry


# ---------------- inverse_chi: the Curie-Weiss theta crossing ----------------

def test_inverse_chi_cw_line_itself_reaches_the_theta_crossing():
    """The Curie-Weiss curve is stored by the analyzer from 1/chi = 0 at theta to the top of
    the data, so the fit line itself reaches theta — one line, no dotted continuation."""
    fig = _fig("magnetization_vsm.dat", "inverse_chi")
    ax = fig.axes[0]
    assert not _lines(ax, "fit-extrap")
    p = _res("magnetization_vsm.dat").data["fit"]["params"]
    ln = next(ln for ln in _lines(ax, "fit") if ln.get_label() == "Curie-Weiss fit")
    x = np.asarray(ln.get_xdata(), float); y = np.asarray(ln.get_ydata(), float)
    assert x.min() == pytest.approx(p["theta"])               # reaches theta (< 0 here)
    i0 = int(np.argmin(np.abs(x)))                            # the sample nearest T = 0
    assert y[i0] == pytest.approx((x[i0] - p["theta"]) / p["C"])
    assert y[np.argmin(x)] == 0.0                             # 1/chi = 0 at T = theta


def test_inverse_chi_modified_cw_line_stays_finite():
    d = _res("magnetization_vsm.dat").data
    pm = (d.get("fit_modified") or {}).get("params") or {}
    if "theta" not in pm:
        pytest.skip("no modified CW fit on this file")
    fig = _fig("magnetization_vsm.dat", "inverse_chi")
    ax = fig.axes[0]
    fits = _lines(ax, "fit")
    assert len(fits) == 2                    # CW and modified CW, each one line
    for ln in fits:
        assert np.isfinite(np.asarray(ln.get_ydata(), float)).all()


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


def _guard(x, y, ref=(1.0, 2.0)):
    from cryosweep_core.plotting.render import _extrap_plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ln = _extrap_plot(ax, np.asarray(x, float), np.asarray(y, float), GlobalStyle(), "C0",
                      y_ref=np.asarray(ref, float))
    out = (np.asarray(ln.get_xdata(), float), np.asarray(ln.get_ydata(), float), ln)
    plt.close(fig)
    return out


def test_extrap_guard_cuts_a_blowup_at_the_cap():
    """cap = 3 x max|fitted segment| = 6. The line is cut where it CROSSES the cap, not at the
    last sample that happened to survive — so it visibly runs off instead of stopping short."""
    x, y, ln = _guard(np.linspace(0, 1, 5), [1e9, 2.0, 1.5, 1.2, 1.0])
    assert np.abs(y).max() == 6.0                     # clamped exactly, no float overshoot
    assert 0.0 < x[0] < 0.25                          # the crossing lies between the samples
    assert list(y[1:]) == [2.0, 1.5, 1.2, 1.0]        # everything below the cap is untouched
    assert ln._extrap_cut == 1


def test_extrap_guard_keeps_only_the_run_attached_to_the_fit():
    """Samples beyond a blow-up are on the far side of a pole. Joining them to the near side
    draws a straight line through the pole."""
    x, y, ln = _guard(np.linspace(0, 1, 6), [1.0, 1.1, 50.0, 1.4, 1.2, 1.0])
    assert x[-1] == 1.0 and x[0] > 0.4                # nothing from before the blow-up
    assert np.abs(y).max() == 6.0 and ln._extrap_cut == 3


def test_extrap_guard_does_not_interpolate_across_a_non_finite_sample():
    x, y, ln = _guard(np.linspace(0, 1, 5), [np.inf, np.nan, 1.5, 1.2, 1.0])
    assert list(y) == [1.5, 1.2, 1.0] and np.all(np.isfinite(x))
    assert ln._extrap_cut == 2


def test_extrap_guard_handles_a_negative_blowup():
    x, y, ln = _guard(np.linspace(0, 1, 4), [-1e6, 1.0, 1.0, 1.0])
    assert y[0] == -6.0 and ln._extrap_cut == 1


def test_extrap_guard_leaves_a_well_behaved_line_alone():
    x, y, ln = _guard(np.linspace(0, 1, 5), [1.8, 1.6, 1.4, 1.2, 1.0])
    assert list(y) == [1.8, 1.6, 1.4, 1.2, 1.0] and ln._extrap_cut == 0


def test_extrap_guard_draws_nothing_when_the_sample_next_to_the_fit_is_beyond_the_cap():
    x, y, ln = _guard(np.linspace(0, 1, 4), [1.0, 1.0, 1.0, 99.0])
    assert x.size == 0 and ln._extrap_cut == 4


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
