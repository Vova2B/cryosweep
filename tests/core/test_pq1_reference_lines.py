import pathlib
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle, ReferenceLine
from cryosweep_core.plotting.render import render_kind, _apply_robust_view

FIX = pathlib.Path(__file__).parent / "fixtures"
def _res():
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_reference_lines_drawn_with_label():
    spec = PlotSpec(reference_lines=[
        ReferenceLine(axis="h", value=0.0, label="ρ=0"),
        ReferenceLine(axis="v", value=50.0, label="T_c"),
    ])
    ax = render_kind(_res(), "resistivity_rho_t", spec, GlobalStyle()).axes[0]
    texts = [t.get_text() for t in ax.texts]
    assert "ρ=0" in texts and "T_c" in texts
    # reference lines are not added to the legend
    labels = ax.get_legend_handles_labels()[1] if ax.get_legend() else []
    assert "ρ=0" not in labels and "T_c" not in labels

def test_no_reference_lines_by_default():
    # annotation=False isolates reference-line behaviour from the Task 6 rho0/n/RRR box.
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(annotation=False), GlobalStyle()).axes[0]
    assert ax.texts == [] or all("=" not in t.get_text() for t in ax.texts)

def test_robust_view_noop_on_refline_only_axes():
    """Regression: all-refline axes should not crash _apply_robust_view;
    robust_view enabled (default), but no data lines to scale."""
    fig = Figure(); FigureCanvasAgg(fig); ax = fig.add_subplot(111)
    ax.axvline(0.0, gid="refline")      # only a refline, no data lines
    # robust_view defaults True in GlobalStyle; explicitly pass to exercise the path
    _apply_robust_view(ax, PlotSpec(), GlobalStyle(robust_view=True))
    # test passes iff no ValueError is raised
