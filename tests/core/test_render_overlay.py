import pathlib, dataclasses
import matplotlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.catalog import OverlayFile, overlay_series, BUILTIN_PLOTKINDS
from cryosweep_core.plotting.render import render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}

def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())

def _ov():
    return [OverlayFile(0, "A", None), OverlayFile(1, "B", None)]

def test_overlay_two_files_two_distinct_coloured_lines():
    r = _vsm()
    fig = render_kind([r, r], "inverse_chi", PlotSpec(), GlobalStyle(), overlay=_ov())
    lines = [ln for ln in fig.axes[0].lines if ln.get_marker() != "None"]   # data series (no fit in overlay)
    assert len(lines) == 2
    c0 = matplotlib.colors.to_hex(lines[0].get_color()); c1 = matplotlib.colors.to_hex(lines[1].get_color())
    assert c0 != c1                                                          # colour-by-file
    labels = {ln.get_label() for ln in lines}
    assert labels == {"A · 1/χ", "B · 1/χ"}                                  # filename-tagged

def test_overlay_fit_line_skipped():
    r = _vsm()
    fig = render_kind([r, r], "inverse_chi", PlotSpec(fit_line=True), GlobalStyle(), overlay=_ov())
    assert not any(ln.get_label() == "Curie-Weiss fit" for ln in fig.axes[0].lines)  # no fit in overlay

def test_overlay_per_file_colour_override():
    r = _vsm()
    ov = [OverlayFile(0, "A", "#ff0000"), OverlayFile(1, "B", "#00ff00")]
    fig = render_kind([r, r], "inverse_chi", PlotSpec(), GlobalStyle(), overlay=ov)
    lines = [ln for ln in fig.axes[0].lines if ln.get_marker() != "None"]
    cols = {matplotlib.colors.to_hex(ln.get_color()) for ln in lines}
    assert cols == {"#ff0000", "#00ff00"}

def test_overlay_curve_selection_file_qualified():
    r = _vsm()
    eff = [s.key for s in overlay_series(KINDS["inverse_chi"], [r, r], _ov())]   # ["0::curve","1::curve"]
    fig = render_kind([r, r], "inverse_chi", PlotSpec(curves=[eff[0]]), GlobalStyle(), overlay=_ov())
    assert len([ln for ln in fig.axes[0].lines if ln.get_marker() != "None"]) == 1   # only file A

def test_overlay_none_is_byte_identical():
    # the A/B path: two results, colour-by-plotted-index, untagged labels
    r = _vsm()
    fig = render_kind([r, r], "vsm_moment_t", PlotSpec(fit_line=False), GlobalStyle())  # overlay defaults None
    lines = fig.axes[0].lines
    assert len(lines) == 2 and lines[0].get_label() == "Moment"   # untagged (A behavior)
