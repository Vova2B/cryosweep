"""KNOWN-ISSUES item 1 — the low-T inset is placed by measurement, not a fixed corner.

The inset is placed FIRST (in the renderer body) and the legend, drawn later, treats its
bbox as a hard obstacle — acyclic by construction; pinned here by the no-overlap and
determinism tests. When no corner is clear, the inset is dropped with an on-figure note
rather than hiding a third of the primary curve behind a supplement.

Counting assertions are necessary, not sufficient — the reproducer, the counter-example,
hc_full_cp_t and tto_kappa_t are also rendered at two font sizes and inspected by eye.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pytest

from cryosweep_core.plotting.render import (_inset_spot, render_kind,
                                            _axes_points_in_host_frac)
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
_REG = build_default_registry()
_RESULTS = {}


def _result(fname):
    if fname not in _RESULTS:
        _RESULTS[fname] = analyze_file(load_dat(str(EXAMPLES / fname)), RunConfig.load(), _REG)
    return _RESULTS[fname]


def _inset_of(fig):
    return next((a for a in fig.axes if a.get_label() == "inset"), None)


def _frac_box(ax, bb):
    inv = ax.transAxes.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _points_under(fig, ax, bb):
    """Host-data points under bbox. The inset is hidden while counting: its own magnified
    points live inside its bbox by definition and are not 'hidden data'."""
    fig.canvas.draw()
    x0, y0, x1, y1 = _frac_box(ax, bb)
    iax = _inset_of(fig)
    if iax is not None:
        iax.set_visible(False)
    try:
        pts = _axes_points_in_host_frac(ax)
    finally:
        if iax is not None:
            iax.set_visible(True)
    if not len(pts):
        return 0, 0
    n = int(((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
             (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum())
    return n, len(pts)


# ---------------- the chooser ----------------

def _bottom_data_fig():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.linspace(0, 1, 200)
    ax.plot(x, 0.03 + 0.02 * np.sin(25 * x), ls="none", marker="o", ms=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    return fig, ax


def test_corner_prefers_lower_right_when_clear():
    # data hugs the bottom edge BELOW the padded corner box -> lower right is clear and,
    # as the shipped journal default, must win the tie against the equally-clear uppers
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.linspace(0, 1, 200)
    ax.plot(x, 0.5 + 0.02 * np.sin(25 * x), ls="none", marker="o", ms=2)  # mid-band data
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    assert _inset_spot(ax) == "lower right"
    plt.close(fig)


def test_corner_moves_off_data():
    # data band runs through the lower half -> both lower corners occupied -> upper right
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.linspace(0, 1, 300)
    ax.plot(x, 0.25 + 0.05 * np.sin(9 * x), ls="none", marker="o", ms=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    assert _inset_spot(ax) == "upper right"
    plt.close(fig)


def test_corner_vetoes_text():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.linspace(0, 1, 300)
    ax.plot(x, 0.25 + 0.05 * np.sin(9 * x), ls="none", marker="o", ms=2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.97, 0.97, "annotation\nblock", transform=ax.transAxes, ha="right", va="top")
    fig.canvas.draw()
    assert _inset_spot(ax) == "upper left"          # upper right now holds text
    plt.close(fig)


def test_corner_none_when_everything_covered():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    xx, yy = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    ax.plot(xx.ravel(), yy.ravel(), ls="none", marker=".", ms=1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    assert _inset_spot(ax) is None
    plt.close(fig)


# ---------------- the reproducer (item 1) ----------------

@pytest.mark.parametrize("font_pt", [9.0, 14.0])
def test_item1_inset_no_longer_hides_the_curve(font_pt):
    fig = render_kind([_result("resistivity_superconductor.dat")], "resistivity_rho_t",
                      PlotSpec(), GlobalStyle(font_pt=font_pt))
    ax = fig.axes[0]
    iax = _inset_of(fig)
    assert iax is not None, "the inset must still exist on this figure"
    fig.canvas.draw()
    n, total = _points_under(fig, ax, iax.get_window_extent())
    # was 110 of 314 points (35%); clear-corner standard is <=2% or <3 points
    assert n < 3 or n / total <= 0.02, f"inset hides {n}/{total} points"
    plt.close(fig)


def test_item1_inset_avoids_the_annotation():
    fig = render_kind([_result("resistivity_superconductor.dat")], "resistivity_rho_t",
                      PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    iax = _inset_of(fig)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    ib = iax.get_window_extent(rend)
    for t in ax.texts:
        assert not ib.overlaps(t.get_window_extent(rend)), t.get_text()
    plt.close(fig)


def test_inset_first_legend_avoids_it_and_placement_is_deterministic():
    def _render():
        fig = render_kind([_result("resistivity_superconductor.dat")], "resistivity_rho_t",
                          PlotSpec(), GlobalStyle())
        fig.canvas.draw()
        return fig
    fig1, fig2 = _render(), _render()
    for fig in (fig1, fig2):
        ax = fig.axes[0]
        iax = _inset_of(fig)
        leg = ax.get_legend()
        assert iax is not None and leg is not None
        assert not leg.get_window_extent().overlaps(iax.get_window_extent())
    b1 = _frac_box(fig1.axes[0], _inset_of(fig1).get_window_extent())
    b2 = _frac_box(fig2.axes[0], _inset_of(fig2).get_window_extent())
    assert b1 == pytest.approx(b2)
    plt.close(fig1); plt.close(fig2)


# ---------------- the counter-example must not break ----------------

def test_counter_example_keeps_its_inset_in_the_clear_lower_right():
    fig = render_kind([_result("hall_temperature_dependence.dat")], "resistivity_rho_t",
                      PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    iax = _inset_of(fig)
    assert iax is not None
    fig.canvas.draw()
    x0, y0, x1, y1 = _frac_box(ax, iax.get_window_extent())
    assert x1 > 0.5 and y0 < 0.5, "inset should stay in the lower-right region"
    n, total = _points_under(fig, ax, iax.get_window_extent())
    assert n < 3 or n / total <= 0.02
    plt.close(fig)


# ---------------- same defect class on the other two builders ----------------

@pytest.mark.parametrize("fname,kind", [("heat_capacity.dat", "hc_full_cp_t"),
                                        ("thermal_transport.dat", "tto_kappa_t")])
def test_other_inset_builders_leave_data_visible(fname, kind):
    fig = render_kind([_result(fname)], kind, PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    iax = _inset_of(fig)
    assert iax is not None
    n, total = _points_under(fig, ax, iax.get_window_extent())
    assert n < 3 or n / total <= 0.02, f"{kind}: inset hides {n}/{total} points"
    plt.close(fig)


# ---------------- the drop-with-note fallback ----------------

def test_no_clear_corner_drops_inset_with_note():
    # synthetic resistivity result whose rho(T) fills the whole panel: data at every corner
    from types import SimpleNamespace
    T = np.linspace(2, 300, 400)
    rho = 100.0 + 60.0 * np.sin(T / 18.0)             # oscillates across the full y-range
    d = {"probe": "resistivity",
         "bridges": [{"bridge": 1, "rho_source": "instrument_column",
                      "rho_t_curves": [{"temperature": T.tolist(), "rho": (rho * 1e-6).tolist(),
                                        "field_oe": 0.0, "direction": "warming"}],
                      "rho_h_curves": [], "power_law": None}]}
    res = SimpleNamespace(data=d, status="ok", diagnostics=[], warnings=[])
    fig = render_kind([res], "resistivity_rho_t", PlotSpec(), GlobalStyle())
    assert _inset_of(fig) is None, "no clear corner -> the inset must be dropped"
    notes = [t for ax in fig.axes for t in ax.texts if t.get_gid() == "inset_note"]
    assert len(notes) == 1, "dropping must be said on the figure, not silent"
    assert "inset" in notes[0].get_text()
    plt.close(fig)


# ---------------- the least-bad fallback tier ----------------

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def _fixture_result(fname):
    return analyze_file(load_dat(str(FIXTURES / fname)),
                        RunConfig.load(probe_override="resistivity"), _REG)


def test_fallback_keeps_act_synth_grazing_midsection():
    # act_synth has NO clear spot anywhere (measured), but its best corner grazes only 5.25%
    # of the points, all midsection, no endpoint hidden -> the least-bad fallback keeps the
    # inset at the shipped lower right instead of dropping a useful zoom.
    fig = render_kind([_fixture_result("act_synth.dat")], "resistivity_rho_t",
                      PlotSpec(), GlobalStyle())
    iax = _inset_of(fig)
    assert iax is not None
    x0, y0, x1, y1 = _frac_box(fig.axes[0], iax.get_window_extent())
    assert x1 > 0.5 and y0 < 0.5                      # lower right, as shipped
    plt.close(fig)


def test_endpoint_veto_drops_rho_sc_synth_with_note():
    # rho_sc_synth's best corner covers 9.7% — under the 10% cap — but the Ch2 curve ENDS
    # inside that box: hiding a terminal point recreates the "curve stops at the inset"
    # illusion of the original defect, so the endpoint veto drops the inset, with the note.
    fig = render_kind([_fixture_result("rho_sc_synth.dat")], "resistivity_rho_t",
                      PlotSpec(), GlobalStyle())
    assert _inset_of(fig) is None
    notes = [t for ax in fig.axes for t in ax.texts if t.get_gid() == "inset_note"]
    assert len(notes) == 1
    plt.close(fig)
