"""Reference-line labels are placed by measurement (KNOWN-ISSUES: new item).

Every reference-line label used to be pinned at a hardcoded fraction along its line
(x = 0.02 for horizontal, y = 0.98/0.02 for vertical) — the same fixed-position defect
class as the legend and the low-T inset before it. The fix slides each label ALONG ITS
OWN LINE to the clearest stretch: the line is the constraint, the position along it is
the free parameter, and the current position is always the first candidate so a label
that is already clear does not move (and no golden image moves with it).

Placement order is inset -> reference-line labels -> legend (labels settle inside
_finish after the robust view and before the legend, which treats text as an obstacle);
the one late-created label (tto_lorenz_t's WF label, drawn after _finish because its
in-view guard needs the settled ylim) instead treats the already-drawn legend as an
obstacle. Both directions leave nothing sitting on anything.

Counting is necessary, not sufficient — the three reproducers are also rendered at two
canvas sizes and inspected by eye.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pytest

from cryosweep_core.plotting.render import (render_kind, _place_refline_labels,
                                            _axes_points_in_host_frac)
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle, ReferenceLine
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


def _frac_box(ax, bb):
    inv = ax.transAxes.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def _covered(ax, t):
    """Points of ax's data under text artist t, with the low-T inset's own points ignored."""
    fig = ax.get_figure()
    fig.canvas.draw()
    x0, y0, x1, y1 = _frac_box(ax, t.get_window_extent(fig.canvas.get_renderer()))
    insets = [a for a in fig.axes if a.get_label() == "inset"]
    for a in insets:
        a.set_visible(False)
    try:
        pts = _axes_points_in_host_frac(ax)
    finally:
        for a in insets:
            a.set_visible(True)
    if not len(pts):
        return 0, 0
    n = int(((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
             (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum())
    return n, len(pts)


def _clear(n, total):
    return n < 3 or (total and n / total <= 0.02)


def _labels(fig, gid_prefix="refline-label"):
    return [t for ax in fig.axes for t in ax.texts
            if (t.get_gid() or "").startswith(gid_prefix)]


# ---------------- the placer, on synthetic axes ----------------

def _h_line_fig(data_x_frac):
    """A horizontal refline at y=0.5 with a dense data blob sitting ON the line over the
    given x-fraction band; the label starts at the shipped x=0.02."""
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    lo, hi = data_x_frac
    x = np.linspace(lo, hi, 250)
    ax.plot(x, 0.5 + 0.012 * np.sin(60 * x), ls="none", marker="o", ms=2)
    ax.axhline(0.5, ls="--", color="0.4", gid="refline")
    t = ax.text(0.02, 0.5, "the label", transform=ax.get_yaxis_transform(),
                va="bottom", ha="left", fontsize="small", gid="refline-label:h")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    return fig, ax, t


def test_h_label_slides_off_data():
    fig, ax, t = _h_line_fig((0.0, 0.55))          # left half occupied -> must slide right
    _place_refline_labels(ax, GlobalStyle())
    n, total = _covered(ax, t)
    assert _clear(n, total), f"label still covers {n}/{total}"
    assert t.get_position()[0] > 0.5               # it moved to the clear right stretch
    plt.close(fig)


def test_h_label_stays_put_when_clear():
    fig, ax, t = _h_line_fig((0.45, 1.0))          # left stretch clear -> exact no-op
    _place_refline_labels(ax, GlobalStyle())
    assert t.get_position() == (0.02, 0.5)
    plt.close(fig)


def test_v_label_slides_up_and_no_clear_stretch_takes_least_covered():
    # vertical line, rotated label (the Tc idiom); data blob on the lower half of the line
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    y = np.linspace(0.0, 0.55, 250)
    ax.plot(0.5 + 0.012 * np.sin(60 * y), y, ls="none", marker="o", ms=2)
    ax.axvline(0.5, ls="--", color="0.4", gid="refline")
    t = ax.text(0.5, 0.02, " $T_c$ = 8 K", transform=ax.get_xaxis_transform(),
                rotation=90, va="bottom", ha="left", fontsize=8, gid="refline-label:v")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.canvas.draw()
    _place_refline_labels(ax, GlobalStyle())
    n, total = _covered(ax, t)
    assert _clear(n, total), f"label still covers {n}/{total}"
    assert t.get_position()[1] > 0.5               # slid up past the blob
    plt.close(fig)


def test_generic_reference_line_spec_label_is_placed():
    # the user-configured path: PlotSpec.reference_lines through _draw_reference_lines
    d = {"probe": "resistivity",
         "bridges": [{"channel": 1, "rho_t_curves": [
             {"held_field_oe": 0.0, "direction": 0, "n_points": 250,
              "temperature": np.linspace(2, 150, 250).tolist(),
              "rho": (5e-5 + 1e-7 * np.linspace(2, 150, 250)).tolist()}],
             "rho_h_curves": []}], "capabilities": []}
    import types
    res = types.SimpleNamespace(data=d)
    # a horizontal line right through the rising curve, labeled: at x=0.02 the label sits
    # on the curve; the clear stretch is wherever the curve is far from the line's y
    rl = ReferenceLine(axis="h", value=6.0e-3, label="reference")   # mOhm-cm axis units
    fig = render_kind(res, "resistivity_rho_t", PlotSpec(reference_lines=[rl]))
    labs = [t for ax in fig.axes for t in ax.texts if t.get_text() == "reference"]
    assert len(labs) == 1
    t = labs[0]
    assert (t.get_gid() or "").startswith("refline-label")
    n, total = _covered(fig.axes[0], t)
    assert _clear(n, total), f"spec refline label covers {n}/{total}"
    plt.close(fig)


# ---------------- the three reproducers ----------------

@pytest.mark.parametrize("fname,kind,text_frag", [
    ("heat_capacity_multifield.dat", "hc_full_cp_t", "Dulong"),
    ("resistivity_superconductor.dat", "resistivity_rho_t", "$T_c$"),
    ("thermal_transport.dat", "tto_lorenz_t", "Wiedemann"),
])
def test_reproducer_label_no_longer_covers_data(fname, kind, text_frag):
    fig = render_kind([_result(fname)], kind, PlotSpec(), GlobalStyle())
    hits = [(ax, t) for ax in fig.axes for t in ax.texts if text_frag in t.get_text()]
    assert hits, f"reproducer label {text_frag!r} missing"
    for ax, t in hits:
        n, total = _covered(ax, t)
        assert _clear(n, total), f"{fname} {text_frag!r} covers {n}/{total}"
    plt.close(fig)


def test_order_pinned_inset_labels_legend_no_overlap_and_deterministic():
    def _render():
        fig = render_kind([_result("resistivity_superconductor.dat")], "resistivity_rho_t",
                          PlotSpec(), GlobalStyle())
        fig.canvas.draw()
        return fig
    fig1, fig2 = _render(), _render()
    for fig in (fig1, fig2):
        ax = fig.axes[0]
        rend = fig.canvas.get_renderer()
        iax = next(a for a in fig.axes if a.get_label() == "inset")
        leg = ax.get_legend()
        labs = _labels(fig)
        assert labs, "the Tc label must exist"
        for t in labs:
            tb = t.get_window_extent(rend)
            assert not tb.overlaps(iax.get_window_extent(rend)), "label on inset"
            assert not leg.get_window_extent(rend).overlaps(tb), "legend on label"
    p1 = sorted(t.get_position() for t in _labels(fig1))
    p2 = sorted(t.get_position() for t in _labels(fig2))
    assert p1 == pytest.approx(p2)
    plt.close(fig1); plt.close(fig2)


def test_late_wf_label_avoids_the_existing_legend():
    # tto_lorenz_t draws its label AFTER _finish (in-view guard needs the settled ylim);
    # the placer must then treat the already-drawn legend as an obstacle.
    fig = render_kind([_result("thermal_transport.dat")], "tto_lorenz_t",
                      PlotSpec(), GlobalStyle())
    fig.canvas.draw()
    ax = fig.axes[0]
    rend = fig.canvas.get_renderer()
    labs = _labels(fig)
    assert labs
    leg = ax.get_legend()
    if leg is not None:
        for t in labs:
            assert not leg.get_window_extent(rend).overlaps(t.get_window_extent(rend))
    plt.close(fig)
