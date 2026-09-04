import pathlib, dataclasses
import numpy as np
import matplotlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.render import render_kind, render_for, default_kind_for

FIX = pathlib.Path(__file__).parent / "fixtures"

def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())

def test_render_for_vsm_is_inverse_chi_default():
    assert default_kind_for("vsm") == "inverse_chi"
    fig = render_for(_vsm(), PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Temperature (K)"
    assert len(ax.lines) >= 1                       # data (+ fit)

def test_axis_limits_and_scale_override():
    fig = render_kind(_vsm(), "inverse_chi", PlotSpec(xmin=10, xmax=200, yscale="log"))
    ax = fig.axes[0]
    assert ax.get_xlim() == (10.0, 200.0)
    assert ax.get_yscale() == "log"

def test_empty_curve_selection_raises():
    import pytest
    with pytest.raises(ValueError):
        render_kind(_vsm(), "inverse_chi", PlotSpec(curves=[]))

def test_multi_result_doubles_data_lines():
    r = _vsm()
    fig = render_kind([r, r], "vsm_moment_t", PlotSpec(fit_line=False))
    assert len([ln for ln in fig.axes[0].lines]) == 2     # one per result, no fit on moment

def test_style_color_applies_only_when_single_series():
    fig = render_kind(_vsm(), "vsm_moment_t", PlotSpec(fit_line=False), GlobalStyle(color="#ff0000"))
    ln = fig.axes[0].lines[0]
    assert matplotlib.colors.to_hex(ln.get_color()) == "#ff0000"

def test_unknown_kind_raises_keyerror():
    import pytest
    with pytest.raises(KeyError):
        render_kind(_vsm(), "does_not_exist", PlotSpec())

def test_render_for_rejects_list():
    import pytest
    with pytest.raises(TypeError):
        render_for([_vsm()], PlotSpec())
