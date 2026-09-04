"""_drawn_texts must return only artists matplotlib actually draws.

Zipping get_ticklocs() against get_ticklabels() misaligns with stale tick artists:
it reports far-off-canvas phantoms AND drops genuinely-drawn labels. Harvest from
the tick objects instead. Measured before this helper existed: an unfiltered
off-figure check flagged 38-39 of 41 entries, every single flag a phantom.
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


def test_no_phantom_ticks_outside_view():
    import pq_compare

    fig, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1, 2])
    ax.set_xlim(0, 1)  # ticks at 1.25..2.0 become stale, never-drawn artists
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    F = fig.bbox
    for t in pq_compare._drawn_texts(fig):
        bb = t.get_window_extent(renderer=r)
        assert bb.x1 >= F.x0 - 1 and bb.x0 <= F.x1 + 1, f"phantom harvested: {t.get_text()!r}"
    plt.close(fig)


def test_drawn_labels_are_kept():
    """The filter must not be so aggressive that it drops genuinely drawn labels."""
    import pq_compare

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1.0, 5.0, 9.0])
    fig.canvas.draw()
    texts = {t.get_text() for t in pq_compare._drawn_texts(fig)}
    drawn = {
        lab.get_text()
        for axis in (ax.xaxis, ax.yaxis)
        for tick in axis.get_major_ticks()
        for lab in (tick.label1, tick.label2)
        if lab is not None and lab.get_visible() and (lab.get_text() or "").strip()
        and min(axis.get_view_interval()) - 1e-9 <= tick.get_loc() <= max(axis.get_view_interval()) + 1e-9
    }
    assert drawn and drawn <= texts, f"dropped drawn labels: {drawn - texts}"
    plt.close(fig)


def test_twin_axes_detected():
    import pq_compare

    fig, ax = plt.subplots()
    tw = ax.twinx()
    assert pq_compare._is_twin(ax, tw)
    assert pq_compare._is_twin(tw, ax)
    fig2, ax2 = plt.subplots()
    assert not pq_compare._is_twin(ax, ax2)
    plt.close(fig)
    plt.close(fig2)


def test_sharex_stacked_panels_are_not_twins():
    """False-green guard: sharex-stacked panels are 'joined' on x exactly like a twin,
    but occupy different bboxes. Treating them as twins would disable occlusion
    checking across stacked headline panels (tto_summary_t and friends)."""
    import pq_compare

    fig, (a1, a2) = plt.subplots(2, sharex=True)
    assert a1.get_shared_x_axes().joined(a1, a2)  # the trap
    assert not pq_compare._is_twin(a1, a2)  # must still not be a twin
    plt.close(fig)
