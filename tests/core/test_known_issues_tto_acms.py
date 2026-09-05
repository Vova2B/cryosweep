"""KNOWN-ISSUES #6 / #8 / #16 — TTO and ACMS display defects, pinned against shipped examples.

#6  No top headroom: the κ peak (thermal_transport.dat, stacked panel a) and the χ′ high-T
    plateau (ac_susceptibility.dat, top panel) touch the axes frame — the 5% matplotlib
    margin is measured to the data coordinate and the 7 pt marker glyph eats most of it.
#8  The zero-field TTO curve omits its field: legend read `cooling` beside
    `90000 Oe, cooling`, leaving the reader to guess the held field of the first curve.
#16 tto_lorenz_t on a linear axis cannot show the thing it exists to show: L/L₀ diverges at
    low T, the curve clips at the top, and the Wiedemann-Franz line at 1 flattens onto the
    bottom axis.
"""
import pathlib
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.plotting.render import render_kind

ROOT = pathlib.Path(__file__).resolve().parents[2]
_REG = build_default_registry()
_CACHE = {}


def _analyzed(path):
    if path not in _CACHE:
        _CACHE[path] = analyze_file(load_dat(str(ROOT / path)), RunConfig.load(), _REG)
    return _CACHE[path]


def _headroom_frac(ax):
    ys = np.concatenate([np.asarray(l.get_ydata(), float) for l in ax.lines
                         if l.get_gid() is None]) if ax.lines else np.array([])
    ys = ys[np.isfinite(ys)]
    lo, hi = ax.get_ylim()
    return (hi - float(ys.max())) / (hi - lo)


# ---------------- #6: top headroom ---------------------------------------------------------

def test_tto_summary_kappa_panel_has_top_headroom():
    fig = render_kind(_analyzed("examples/thermal_transport.dat"), "tto_summary_t")
    assert _headroom_frac(fig.axes[0]) >= 0.075     # was 0.0455: the marker glyph ate it


def test_acms_chi_panels_have_top_headroom_and_no_clipping():
    fig = render_kind(_analyzed("examples/ac_susceptibility.dat"), "acms_chi_t")
    for ax in fig.axes[:2]:
        h = _headroom_frac(ax)
        assert h >= 0.075, h              # χ′ plateau was at 0.044; χ″ peak at -0.005 (clipped)


# ---------------- #8: every curve names its field on a multi-field file --------------------

def test_zero_field_curve_names_its_field_when_other_fields_present():
    from cryosweep_core.plotting.catalog import series_tto_kappa_t
    labels = [s.label for s in series_tto_kappa_t(_analyzed("examples/thermal_transport.dat"))]
    assert "0 Oe, cooling" in labels                # |H| < 50 Oe is the zero-field convention
    assert "90000 Oe, cooling" in labels
    assert "cooling" not in labels                  # no field-less orphan next to a named one


def test_single_field_file_keeps_direction_only_labels():
    from cryosweep_core.plotting.catalog import series_tto_kappa_t
    res = _analyzed("tests/core/fixtures/tto_powerlaw_synth.dat")
    labels = [s.label for s in series_tto_kappa_t(res)]
    assert labels and all("Oe" not in l for l in labels)


# ---------------- #16: lorenz panel shows the reference it exists for ----------------------

def test_lorenz_defaults_to_log_y_with_reference_in_view():
    fig = render_kind(_analyzed("examples/thermal_transport.dat"), "tto_lorenz_t")
    ax = fig.axes[0]
    assert ax.get_yscale() == "log"
    lo, hi = ax.get_ylim()
    assert lo < 1.0 < hi                            # the L/L₀ = 1 line is IN the view
    # no data clipping: every finite point of every curve is inside the view
    ys = np.concatenate([np.asarray(l.get_ydata(), float) for l in ax.lines
                         if l.get_gid() is None])
    ys = ys[np.isfinite(ys)]
    assert float(ys.max()) <= hi and float(ys.min()) >= lo
    # and the line is labelled
    assert any("Wiedemann" in t.get_text() for t in ax.texts)


def test_lorenz_explicit_linear_yscale_still_honoured():
    from cryosweep_core.plotting.spec import PlotSpec
    fig = render_kind(_analyzed("examples/thermal_transport.dat"), "tto_lorenz_t",
                      PlotSpec(yscale="linear"))
    assert fig.axes[0].get_yscale() == "linear"
