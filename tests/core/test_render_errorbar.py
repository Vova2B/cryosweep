"""Task 6: Series.yerr / open_mask + render errorbar branch tests."""
import numpy as np
import pytest
from cryosweep_core.plotting.catalog import Series


def test_series_accepts_yerr_and_open_mask():
    s = Series(key="g", label="γ", x=[0.0, 1.0], y=[0.5, 0.6],
               yerr=[0.01, 0.02], open_mask=[False, True])
    assert s.yerr == [0.01, 0.02]
    assert s.open_mask == [False, True]


def test_existing_series_default_none():
    s = Series(key="c", label="Cp", x=[1.0], y=[2.0])
    assert s.yerr is None and s.open_mask is None


# ---- render path tests ----

def _make_result_with_series(s):
    """Minimal fake result whose series function returns [s]."""
    from types import SimpleNamespace
    from cryosweep_core.plotting.catalog import PlotKind
    kind = PlotKind(key="cp_over_t", label="test", probe="heatcapacity",
                    series=lambda r, field_unit="Oe": [s])
    result = SimpleNamespace(data={"probe": "heatcapacity"})
    return result, kind


def test_errorbar_render_no_crash():
    """Series with yerr renders without error (exercises _errorbar_series)."""
    from cryosweep_core.plotting.render import _plot_data
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = Series(key="g", label="γ", x=[1.0, 2.0, 3.0], y=[0.5, 0.6, 0.7],
               yerr=[0.01, 0.02, 0.01], open_mask=[False, False, True])
    result, kind = _make_result_with_series(s)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    spec = PlotSpec(); style = GlobalStyle()

    # should not raise
    _plot_data(ax, [result], kind, spec, style)
    # check errorbars were drawn (ax.containers holds ErrorbarContainer objects)
    assert len(ax.containers) >= 1


def test_errorbar_render_hollow_second_call():
    """open_mask=True points get a second errorbar call (hollow). Check 2 containers."""
    from cryosweep_core.plotting.render import _plot_data
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = Series(key="g", label="γ", x=[1.0, 2.0], y=[0.5, 0.6],
               yerr=[0.01, 0.02], open_mask=[False, True])
    result, kind = _make_result_with_series(s)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    spec = PlotSpec(); style = GlobalStyle()

    _plot_data(ax, [result], kind, spec, style)
    # one container for solid, one for hollow
    assert len(ax.containers) == 2


def test_no_yerr_uses_ax_plot_path():
    """Series with yerr=None still uses ax.plot (one line, no containers)."""
    from cryosweep_core.plotting.render import _plot_data
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = Series(key="c", label="Cp", x=[1.0, 2.0], y=[2.0, 3.0])
    result, kind = _make_result_with_series(s)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    spec = PlotSpec(); style = GlobalStyle()

    _plot_data(ax, [result], kind, spec, style)
    assert len(ax.containers) == 0   # no errorbars
    assert len(ax.lines) == 1        # one ax.plot line


def test_all_masked_no_solid_container():
    """All points open_mask=True: only the hollow container, no solid one."""
    from cryosweep_core.plotting.render import _plot_data
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = Series(key="g", label="γ", x=[1.0, 2.0], y=[0.5, 0.6],
               yerr=[0.01, 0.02], open_mask=[True, True])
    result, kind = _make_result_with_series(s)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    spec = PlotSpec(); style = GlobalStyle()

    _plot_data(ax, [result], kind, spec, style)
    # only hollow call was made (no solid points)
    assert len(ax.containers) == 1


def test_all_solid_no_hollow_container():
    """All points open_mask=False: only the solid container."""
    from cryosweep_core.plotting.render import _plot_data
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = Series(key="g", label="γ", x=[1.0, 2.0], y=[0.5, 0.6],
               yerr=[0.01, 0.02], open_mask=[False, False])
    result, kind = _make_result_with_series(s)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    spec = PlotSpec(); style = GlobalStyle()

    _plot_data(ax, [result], kind, spec, style)
    assert len(ax.containers) == 1


def test_all_masked_series_has_legend_entry():
    """All points open_mask=True: hollow-only call must carry the label (one legend handle)."""
    from cryosweep_core.plotting.render import _plot_data
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = Series(key="g", label="γ_all_masked", x=[1.0, 2.0], y=[0.5, 0.6],
               yerr=[0.01, 0.02], open_mask=[True, True])
    result, kind = _make_result_with_series(s)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    spec = PlotSpec(); style = GlobalStyle()

    _plot_data(ax, [result], kind, spec, style)
    # exactly one container (the hollow call)
    assert len(ax.containers) == 1
    # that container must carry the label (not "_nolegend_")
    handles, labels = ax.get_legend_handles_labels()
    assert any("γ_all_masked" in lbl for lbl in labels), (
        f"expected 'γ_all_masked' in legend labels, got {labels}"
    )


def test_mixed_mask_no_duplicate_legend():
    """Mixed solid+hollow: only one legend entry (solid gets the label, hollow does not)."""
    from cryosweep_core.plotting.render import _plot_data
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    s = Series(key="g", label="γ_mixed", x=[1.0, 2.0, 3.0], y=[0.5, 0.6, 0.7],
               yerr=[0.01, 0.02, 0.01], open_mask=[False, True, False])
    result, kind = _make_result_with_series(s)

    fig = Figure()
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    spec = PlotSpec(); style = GlobalStyle()

    _plot_data(ax, [result], kind, spec, style)
    handles, labels = ax.get_legend_handles_labels()
    count = sum(1 for lbl in labels if "γ_mixed" in lbl)
    assert count == 1, f"expected exactly 1 legend entry for 'γ_mixed', got {count}"
