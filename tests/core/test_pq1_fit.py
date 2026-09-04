import pathlib
import matplotlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.render import render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"

def _vsm():
    return analyze_file(load_dat(str(FIX / "vsm_synth.dat")), RunConfig.load(), build_default_registry())

def test_fit_line_tagged_gid():
    ax = render_kind(_vsm(), "inverse_chi", PlotSpec(fit_line=True), GlobalStyle()).axes[0]
    fits = [l for l in ax.lines if l.get_gid() == "fit"]
    assert len(fits) >= 1

def test_fit_color_override():
    ax = render_kind(_vsm(), "inverse_chi", PlotSpec(fit_line=True),
                     GlobalStyle(fit_color="red")).axes[0]
    fits = [l for l in ax.lines if l.get_gid() == "fit"]
    assert matplotlib.colors.to_hex(fits[0].get_color()) == "#ff0000"

def test_fit_linestyle_default_solid():
    ax = render_kind(_vsm(), "inverse_chi", PlotSpec(fit_line=True), GlobalStyle()).axes[0]
    fits = [l for l in ax.lines if l.get_gid() == "fit"]
    assert fits[0].get_linestyle() == "-"
