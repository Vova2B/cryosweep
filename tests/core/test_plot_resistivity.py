from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
from cryosweep_core.plotting.render import render_for
from cryosweep_core.plotting.spec import PlotSpec

def test_render_for_resistivity_returns_figure(res_path):
    res = ResistivityAnalyzer().analyze(load_dat(res_path), RunConfig())
    fig = render_for(res, PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Temperature (K)"
    assert "cm" in ax.get_ylabel()           # rho axis label mentions Ohm*cm
    assert len(ax.lines) >= 1                 # at least the main rho(T) ramp
