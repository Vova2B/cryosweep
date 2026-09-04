"""Legend auto-placement from data occupancy + explicit manual positions.

KNOWN-ISSUES 4/5/11/12. The chooser scores the nine matplotlib inside positions against the
figure's actual ink — plotted points, text annotations, insets — and relocates outside only
when nothing inside is clear. Explicit positions pass through verbatim.

The counting assertions here are necessary but NOT sufficient — the visual gate (rendering
the reproducers at two font sizes and looking) is part of this change's verification.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cryosweep_core.plotting.render import (_occupancy_legend_loc, _draw_legend, render_kind)
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

import pathlib
EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"

_REG = build_default_registry()
_RESULTS = {}


def _result(fname):
    if fname not in _RESULTS:
        _RESULTS[fname] = analyze_file(load_dat(str(EXAMPLES / fname)), RunConfig.load(), _REG)
    return _RESULTS[fname]


def _fig_with_bottom_data():
    """Data hugging the bottom edge: every upper position is clear."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    x = np.linspace(0, 1, 300)
    ax.plot(x, 0.02 + 0.02 * np.sin(20 * x), label="curve")
    ax.set_ylim(0, 1); ax.set_xlim(0, 1)
    fig.canvas.draw()
    return fig, ax


def _prop():
    return {"size": 8}


# ---------------- the chooser ----------------

def test_chooser_picks_clear_corner():
    fig, ax = _fig_with_bottom_data()
    loc, clear = _occupancy_legend_loc(ax, None, _prop(), GlobalStyle())
    assert clear is True
    assert loc == "upper right"                    # first clear candidate in preference order
    plt.close(fig)


def test_chooser_sees_text_artists():
    fig, ax = _fig_with_bottom_data()
    ax.text(0.98, 0.98, "a big annotation\nthree lines\nof text", transform=ax.transAxes,
            ha="right", va="top")
    fig.canvas.draw()
    loc, clear = _occupancy_legend_loc(ax, None, _prop(), GlobalStyle())
    assert clear is True
    assert loc != "upper right"                    # occupied by text now
    plt.close(fig)


def test_chooser_sees_inset_axes():
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    fig, ax = _fig_with_bottom_data()
    iax = inset_axes(ax, width="45%", height="45%", loc="upper right")
    iax.set_label("inset")
    fig.canvas.draw()
    loc, clear = _occupancy_legend_loc(ax, None, _prop(), GlobalStyle())
    assert clear is True
    assert loc != "upper right"
    plt.close(fig)


def test_chooser_ignores_twin_axes_as_obstacle():
    # a twin axis shares the host frame; it must not veto every candidate
    fig, ax = _fig_with_bottom_data()
    tax = ax.twinx()
    tax.plot([0, 1], [0.01, 0.02])
    fig.canvas.draw()
    loc, clear = _occupancy_legend_loc(ax, None, _prop(), GlobalStyle())
    assert clear is True
    plt.close(fig)


def test_chooser_declines_when_everything_covered():
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    xx, yy = np.meshgrid(np.linspace(0, 1, 40), np.linspace(0, 1, 40))
    ax.plot(xx.ravel(), yy.ravel(), ls="none", marker=".", label="grid")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    _, clear = _occupancy_legend_loc(ax, None, _prop(), GlobalStyle())
    assert clear is False                          # caller relocates outside
    plt.close(fig)


def test_font_size_changes_the_decision_at_render_time():
    """The owner's screen-size point: the same data can hold a small legend but not a big
    one. Placement is a render-time decision — the font is a style input to the render."""
    fig, ax = _fig_with_bottom_data()
    _, clear_small = _occupancy_legend_loc(ax, None, {"size": 7}, GlobalStyle())
    _, clear_huge = _occupancy_legend_loc(ax, None, {"size": 80}, GlobalStyle())
    assert clear_small is True
    assert clear_huge is False                     # a canvas-sized legend fits nowhere inside
    plt.close(fig)


# ---------------- explicit manual positions ----------------

@pytest.mark.parametrize("loc", ["upper left", "lower center", "center right", "center"])
def test_explicit_position_passes_through_verbatim(loc):
    fig, ax = _fig_with_bottom_data()
    _draw_legend(ax, _prop(), GlobalStyle(), PlotSpec(legend_loc=loc))
    leg = ax.get_legend()
    assert leg is not None
    import matplotlib.legend as mlegend
    assert leg._get_loc() == mlegend.Legend.codes[loc]
    plt.close(fig)


def test_spec_accepts_nine_positions_and_rejects_junk():
    PlotSpec(legend_loc="upper left")
    GlobalStyle(legend_loc="lower center")
    with pytest.raises(Exception):
        PlotSpec(legend_loc="somewhere nice")


# ---------------- the KNOWN-ISSUES reproducers ----------------

def _legend_data_cover_frac(fig, ax):
    """Fraction of the figure's plotted points under the legend bbox (host-frac)."""
    from cryosweep_core.plotting.render import _axes_points_in_host_frac
    leg = ax.get_legend()
    if leg is None:
        return 0.0
    fig.canvas.draw()
    bb = leg.get_window_extent()
    inv = ax.transAxes.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    pts = _axes_points_in_host_frac(ax)
    if not len(pts):
        return 0.0
    inside = ((pts[:, 0] >= min(x0, x1)) & (pts[:, 0] <= max(x0, x1)) &
              (pts[:, 1] >= min(y0, y1)) & (pts[:, 1] <= max(y0, y1))).sum()
    return inside / len(pts)


@pytest.mark.parametrize("font_pt", [9.0, 14.0])
def test_item4_inverse_chi_legend_not_on_the_data(font_pt):
    res = _result("magnetization_vsm_multifield.dat")
    fig = render_kind([res], "inverse_chi", PlotSpec(), GlobalStyle(font_pt=font_pt))
    ax = fig.axes[0]
    assert _legend_data_cover_frac(fig, ax) <= 0.02
    plt.close(fig)


def test_item5_tto_small_legend_stays_inside_when_room_exists():
    # thermal_transport.dat: two entries, upper-right quadrant empty -> inside, no canvas grow
    res = _result("thermal_transport.dat")
    fig = render_kind([res], "tto_kappa_t", PlotSpec(), GlobalStyle())
    assert not getattr(fig, "_cryosweep_legend_grown", False)
    assert fig.axes[0].get_legend() is not None
    plt.close(fig)


def test_item5_explicit_outside_still_relocates():
    res = _result("thermal_transport.dat")
    fig = render_kind([res], "tto_kappa_t", PlotSpec(legend_loc="outside"), GlobalStyle())
    assert getattr(fig, "_cryosweep_legend_grown", False)
    plt.close(fig)


@pytest.mark.parametrize("font_pt", [9.0, 14.0])
def test_item11_hc_legend_clear_of_dp_text_and_inset(font_pt):
    res = _result("heat_capacity.dat")
    fig = render_kind([res], "hc_full_cp_t", PlotSpec(), GlobalStyle(font_pt=font_pt))
    ax = fig.axes[0]
    leg = ax.get_legend()
    assert leg is not None
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lb = leg.get_window_extent(rend)
    for t in ax.texts:                             # the Dulong-Petit label
        assert not lb.overlaps(t.get_window_extent(rend)), t.get_text()
    for a in fig.axes:
        if a.get_label() == "inset":
            assert not lb.overlaps(a.get_window_extent(rend))
    plt.close(fig)


def test_item12_unresolved_magnetic_entropy_absent_from_figure_and_legend():
    res = _result("heat_capacity.dat")               # magnetic entropy NOT meaningfully resolved
    fig = render_kind([res], "hc_entropy_vs_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    labels = [ln.get_label() for ln in ax.get_lines()]
    assert "S magnetic" not in labels               # no invisible flat-zero curve
    leg = ax.get_legend()
    if leg is not None:
        assert all(t.get_text() != "S magnetic" for t in leg.get_texts())
    plt.close(fig)
