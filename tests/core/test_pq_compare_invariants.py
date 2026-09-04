"""I1 occlusion + I2 degenerate axis. Both must be phantom-immune.

I1 deliberately compares the legend BOX against the inset BOX. A box-vs-text form is
INERT on the real figures -- the inset's tick labels never intersect the legend, so the
motivating defect (hc_full_cp_t, legend 28% covered at 12pt / 62% at 16pt) goes
undetected. That mistake was caught in adversarial plan review; these tests pin it.
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

ENTRY = {"id": "synthetic", "v2_kind": "x"}


def test_degenerate_axis_flagged():
    """Reproduces the real regime: R_H constant to ~3e-15 relative.

    The limits are set explicitly rather than autoscaled from data on purpose.
    matplotlib's `nonsingular` expands a range only when it is degenerate below its own
    ~1e-15 relative tolerance; the two real entries sit at 3.31e-15, just ABOVE that, so
    they survive autoscaling untouched. Feeding constant data here would be expanded to
    a healthy ~10% span and would test nothing.
    """
    import pq_compare

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [-3.0e-8, -3.0e-8, -3.0e-8])
    ax.set_ylim(-3.0e-8, -3.0e-8 + 1e-22)  # rel-span ~3.3e-15, as measured on real files
    fig.canvas.draw()
    lo, hi = ax.get_ylim()
    rel = (hi - lo) / max(abs(lo), abs(hi))
    assert rel < 1e-12, f"fixture no longer degenerate (rel={rel:.2e})"
    fails = pq_compare._check_fig(ENTRY, fig, "ok")
    assert any("DEGENERATE-AXIS" in f for f in fails), fails
    plt.close(fig)


def test_healthy_axes_clean():
    import pq_compare

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1.0, 5.0, 9.0])
    fig.canvas.draw()
    fails = pq_compare._check_fig(ENTRY, fig, "ok")
    assert fails == [], fails
    plt.close(fig)


def test_twin_overlap_not_reported():
    import pq_compare

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1.0, 5.0, 9.0], label="a")
    tw = ax.twinx()
    tw.plot([1, 2, 3], [2.0, 4.0, 8.0], label="b")
    ax.legend(loc="center")
    fig.canvas.draw()
    fails = pq_compare._check_fig(ENTRY, fig, "ok")
    assert not any("OCCLUSION" in f for f in fails), fails
    plt.close(fig)


def test_legend_over_inset_flagged():
    """The motivating defect: legend BOX covered by the inset BOX."""
    import pq_compare
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1.0, 5.0, 9.0], label="series")
    iax = inset_axes(ax, width="42%", height="40%", loc="lower right", borderpad=1.4)
    iax.set_label("inset")  # how _check_fig identifies an inset
    ax.legend(loc="lower right")  # deliberately on top of the inset
    fig.canvas.draw()
    fails = pq_compare._check_fig(ENTRY, fig, "ok")
    assert any("OCCLUSION" in f for f in fails), fails
    plt.close(fig)


def test_legend_clear_of_inset_is_clean():
    """Guard against the inverse: a legend well away from the inset must NOT fire."""
    import pq_compare
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1.0, 5.0, 9.0], label="series")
    iax = inset_axes(ax, width="42%", height="40%", loc="lower right", borderpad=1.4)
    iax.set_label("inset")
    ax.legend(loc="upper left")
    fig.canvas.draw()
    assert not [f for f in pq_compare._check_fig(ENTRY, fig, "ok") if "OCCLUSION" in f]
    plt.close(fig)
