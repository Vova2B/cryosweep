"""KNOWN-ISSUES #13 — the vsm_mh low-field zoom panel must rescale its y-axis.

It inherited the full-range autoscale (0-0.55 µ_B on the multifield example) while its own
window holds only 0-0.06, so the zoom panel was ~80% empty — the opposite of a zoom.
"""
import pathlib
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.plotting.render import render_kind

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _fig():
    res = analyze_file(load_dat(str(ROOT / "examples" / "magnetization_vsm_multifield.dat")),
                       RunConfig.load(), build_default_registry())
    return render_kind(res, "vsm_mh")


def _window_data(ax):
    xs, ys = [], []
    lo, hi = ax.get_xlim()
    for ln in ax.lines:
        if ln.get_gid() is not None:
            continue
        x = np.asarray(ln.get_xdata(), float); y = np.asarray(ln.get_ydata(), float)
        m = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
        xs.append(x[m]); ys.append(y[m])
    return np.concatenate(xs), np.concatenate(ys)


def test_zoom_panel_y_view_tracks_its_own_window():
    fig = _fig()
    zoom = fig.axes[1]
    assert zoom.get_title() == "low field"          # premise: panel identity
    _, y = _window_data(zoom)
    assert y.size                                   # premise: the window holds data
    dspan = float(y.max() - y.min())
    ylo, yhi = zoom.get_ylim()
    # the view is a padded fit to the WINDOW's data, not the full-range inherit:
    # the window span fills at least half the view (was ~11% before the fix)
    assert dspan / (yhi - ylo) >= 0.5, (dspan, ylo, yhi)
    assert yhi >= float(y.max()) and ylo <= float(y.min())     # nothing clipped


def test_main_panel_view_is_untouched():
    fig = _fig()
    main, zoom = fig.axes[0], fig.axes[1]
    _, y = _window_data(main)
    # main still frames the full loop
    assert main.get_ylim()[1] >= float(y.max())
    assert main.get_ylim()[1] > zoom.get_ylim()[1]  # the zoom really is zoomed in y now
