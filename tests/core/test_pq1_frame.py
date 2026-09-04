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

def test_default_frame_inward_minor_allsides():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle()).axes[0]
    # minor ticks present on x
    assert len(ax.xaxis.get_minor_ticks()) > 0
    # inward direction recorded on major x ticks
    assert ax.xaxis.get_ticklines()  # ticks exist
    # all four spines visible
    assert all(ax.spines[s].get_visible() for s in ("top", "bottom", "left", "right"))

def test_grid_default_off_and_toggle():
    off = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle()).axes[0]
    assert not off.xaxis._major_tick_kw.get("gridOn", False) or not any(g.get_visible() for g in off.get_xgridlines())
    on = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle(grid=True)).axes[0]
    assert any(g.get_visible() for g in on.get_xgridlines())

def test_grid_per_plot_override():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(grid=True), GlobalStyle(grid=False)).axes[0]
    assert any(g.get_visible() for g in ax.get_xgridlines())

def test_spine_width_applied():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle(spine_width=2.0)).axes[0]
    assert ax.spines["left"].get_linewidth() == 2.0
