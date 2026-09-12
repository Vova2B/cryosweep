"""The top-headroom guarantee, in the units the problem is actually in.

KNOWN-ISSUES #6 established that matplotlib's 5% y-margin is measured to the data
COORDINATE, so the marker glyph drawn at that coordinate eats the margin and the topmost
point touches or crosses the axes frame. `_ensure_top_headroom` fixed that -- but it was
wired into two panels (the TTO kappa panel and the ACMS chi panels) and expressed as a fixed
FRACTION of the data span, while a marker is a PHYSICAL size. A fraction that clears a 7 pt
glyph on the shipped 90x70 mm canvas does not clear it on a smaller one, and the GUI draws
its grid panels small. Measured before the fix, on the Hall R_H(T) panel at 7 pt markers:

    canvas      headroom      marker radius      gap
    90 x 70 mm   24.8 px         14.6 px      +10.2 px   ok
    60 x 45 mm   12.2 px         14.6 px       -2.4 px   marker crosses the frame
    40 x 30 mm    4.1 px         14.6 px      -10.5 px   marker crosses the frame

So these assertions are in PIXELS against the glyph, not in fractions of the span.
"""
import numpy as np
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.spec import GlobalStyle

_CACHE = {}


def _gap_px(fig, ax, marker_pt):
    """Pixels between the top of the topmost drawn glyph and the axes frame."""
    fig.draw_without_rendering()
    bb = ax.get_window_extent(fig.canvas.get_renderer())
    lo, hi = ax.get_ylim()
    ys = [np.asarray(l.get_ydata(), float) for l in ax.lines if l.get_gid() is None]
    ys = np.concatenate(ys) if ys else np.array([])
    ys = ys[np.isfinite(ys)]
    if ys.size == 0 or not np.isfinite(hi - lo) or hi <= lo:
        return None
    headroom_px = bb.height * (hi - float(ys.max())) / (hi - lo)
    return headroom_px - (marker_pt / 72.0 * fig.dpi / 2.0)


@pytest.fixture
def tdep(hall_real_path):
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file gitignored/absent")
    if "r" not in _CACHE:                    # the analysis is the slow part; the renders are not
        _CACHE["r"] = HallTempDepAnalyzer().analyze(load_dat(str(hall_real_path)), RunConfig(
            hall={"hall_channel": 1, "thickness_mm": 0.07, "longitudinal_channel": 2}))
    return _CACHE["r"]


@pytest.mark.parametrize("w_mm,h_mm", [(60, 45), (90, 70), (140, 110), (220, 170)])
def test_the_topmost_marker_clears_the_frame_at_every_canvas_size(tdep, w_mm, h_mm):
    """The guarantee is that the glyph is fully drawn, and a glyph has a physical size. A
    headroom expressed as a share of the data span shrinks with the canvas while the glyph
    does not, so this is asserted in pixels at four sizes including two the GUI actually
    uses for its grid panels.

    The floor is ~50x38 mm, measured: there the axes box is 182 px tall against a 14.6 px
    glyph radius and the gap lands at -0.03 px; below it (40x30 -> 90 px, 35x25 -> 31 px)
    matplotlib itself reports "axes sizes collapsed to zero" and no headroom rule can place
    a physical glyph inside a box a few glyphs tall. That is a canvas too small for its
    decorations, not a headroom defect, so it is not asserted here."""
    ms = 7.0
    fig = render_kind(tdep, "hall_tdep_RH_T", None, GlobalStyle(width_mm=w_mm, height_mm=h_mm,
                                                               marker_size=ms))
    gap = _gap_px(fig, fig.axes[0], ms)
    assert gap is not None and gap >= 0, f"topmost marker crosses the frame by {-gap:.1f} px"


def test_a_robust_view_exclusion_is_still_left_alone(tdep):
    """The headroom rule must not re-open a view the robust rule deliberately narrowed. On
    this panel the robust view cuts a far outlier (series max 1.088e-3 against a view top of
    ~4.0e-4), and re-opening it to fit that point would flatten every other point onto the
    axis. Headroom is a trim, not a robust-view override -- so here the top point is EXPECTED
    to sit outside the frame, and that is a separate question from the glyph clipping above."""
    fig = render_kind(tdep, "hall_tdep_mobility_T", None, GlobalStyle(marker_size=7.0))
    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    ys = np.concatenate([np.asarray(l.get_ydata(), float) for l in ax.lines
                         if l.get_gid() is None])
    ys = ys[np.isfinite(ys)]
    assert float(ys.max()) > hi, "the robust view should still be excluding the far outlier"
    assert hi < 1e-3, "the view must not have been re-opened to fit it"
