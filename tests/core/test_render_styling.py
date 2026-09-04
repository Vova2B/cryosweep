import pathlib, dataclasses
import matplotlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.render import render_kind, _cmap_colors

FIX = pathlib.Path(__file__).parent / "fixtures"

def _res():           # 2 bridges x 1 rho_t ramp -> 2 data series default-on
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_cmap_colors_n_and_reverse():
    a = _cmap_colors("viridis", 3, False)
    b = _cmap_colors("viridis", 3, True)
    assert a is not None and len(a) == 3 and a[0] != a[2]
    assert b[0] == a[2] or b != a          # reversed differs
    assert _cmap_colors("bogus_name", 3, False) is None     # unknown -> None, no raise

def test_colormap_assigns_distinct_colors():
    fig = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle(colormap="viridis"))
    cols = [ln.get_color() for ln in fig.axes[0].lines[:2]]
    assert matplotlib.colors.to_hex(cols[0]) != matplotlib.colors.to_hex(cols[1])

def test_marker_edge_applied():
    fig = render_kind(_res(), "resistivity_rho_t", PlotSpec(),
                      GlobalStyle(edge_color="black", edge_width=1.5))
    ln = fig.axes[0].lines[0]
    assert matplotlib.colors.to_hex(ln.get_markeredgecolor()) == "#000000"
    assert ln.get_markeredgewidth() == 1.5

def test_color_wins_for_single_series_even_with_colormap():
    fig = render_kind(_res(), "resistivity_rho_t",
                      PlotSpec(curves=[s.key for s in build_default_registry()
                               .plot_kinds_for("resistivity")[0].series(_res())][:1]),
                      GlobalStyle(color="#ff0000", colormap="viridis"))
    assert matplotlib.colors.to_hex(fig.axes[0].lines[0].get_color()) == "#ff0000"

def test_default_legend_size_is_font_pt_minus_one():
    fig = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle(font_pt=10))
    leg = fig.axes[0].get_legend()
    assert leg is not None
    assert all(t.get_fontsize() == 9.0 for t in leg.get_texts())   # font_pt - 1 (prop= equivalence)

def test_per_element_sizes_applied():
    fig = render_kind(_res(), "resistivity_rho_t", PlotSpec(),
                      GlobalStyle(label_size=15, tick_size=7, legend_size=6))
    ax = fig.axes[0]
    assert ax.xaxis.label.get_fontsize() == 15
    assert ax.get_legend().get_texts()[0].get_fontsize() == 6
    assert ax.get_xticklabels()[0].get_fontsize() == 7
