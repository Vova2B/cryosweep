import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.render import render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"
def _res():
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_legend_frameless_default():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle()).axes[0]
    leg = ax.get_legend()
    assert leg is not None and leg.get_frame_on() is False

def test_legend_off():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle(legend_on=False)).axes[0]
    assert ax.get_legend() is None

def test_legend_off_per_plot():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(legend_on=False), GlobalStyle()).axes[0]
    assert ax.get_legend() is None

def test_legend_forced_outside():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle(legend_loc="outside")).axes[0]
    leg = ax.get_legend()
    anchor = leg.get_bbox_to_anchor()
    frac_x = ax.transAxes.inverted().transform((anchor.x1, anchor.y1))[0]
    assert frac_x > 1.0
