import pathlib, dataclasses
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.render import render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"

def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())

def test_labels_stay_inside_after_small_wide_resize():
    # mimics the fixed embedded-canvas case: GUI renders at screen_dpi=100,
    # so a 704x280 px widget -> 7.04x2.80 in figure (vs the broken 2.35x0.93 in at dpi=300)
    fig = render_kind(_vsm(), "inverse_chi", PlotSpec(), GlobalStyle(dpi=100))
    fig.set_size_inches(7.04, 2.80)
    fig.canvas.draw()
    ax = fig.axes[0]; fw, fh = fig.canvas.get_width_height()
    for art in (ax.xaxis.label, ax.yaxis.label):
        bb = art.get_window_extent()
        assert bb.x0 >= 0 and bb.y0 >= 0 and bb.x1 <= fw and bb.y1 <= fh, \
            f"label clipped: bbox=({bb.x0:.0f},{bb.y0:.0f},{bb.x1:.0f},{bb.y1:.0f}) fig=({fw}x{fh})"

def test_constrained_layout_engine_active():
    fig = render_kind(_vsm(), "inverse_chi", PlotSpec(), GlobalStyle())
    assert fig.get_layout_engine() is not None   # constrained layout engine set
