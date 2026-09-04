import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec
from cryosweep_core.plotting.render import _apply_frame

def _ax(xdata):
    fig = Figure(); FigureCanvasAgg(fig); ax = fig.add_subplot(111)
    ax.plot(xdata, [1, 2, 3])
    return fig, ax

def test_thousands_groups_large_linear_axis():
    fig, ax = _ax([0, 30000, 60000])
    _apply_frame(ax, GlobalStyle(thousands_sep=True), PlotSpec())
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert any("," in s for s in labels)   # e.g. 60,000

def test_thousands_noop_small_axis():
    fig, ax = _ax([0.0, 0.5, 1.0])
    _apply_frame(ax, GlobalStyle(thousands_sep=True), PlotSpec())
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert not any("," in s for s in labels)

def test_thousands_off_by_default():
    fig, ax = _ax([0, 30000, 60000])
    _apply_frame(ax, GlobalStyle(), PlotSpec())   # thousands_sep default False
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
    assert not any("," in s for s in labels)
