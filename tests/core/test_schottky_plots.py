import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import (
    series_hc_delta_vs_field, series_hc_f_vs_field, series_hc_schottky_multifield)
from cryosweep_core.plotting.render import (
    render_hc_delta_vs_field, render_hc_f_vs_field,
    render_hc_alphaN_vs_field, render_hc_schottky_multifield)

FIX = pathlib.Path(__file__).parent / "fixtures"


def _res(**hc):
    cfg = RunConfig.model_validate({"heatcapacity": {**hc}})
    return analyze_file(load_dat(str(FIX / "hc_schottky_synth.dat")), cfg, build_default_registry())


def test_series_empty_when_off():
    res = _res()                                     # schottky off
    assert series_hc_delta_vs_field(res) == []
    assert series_hc_schottky_multifield(res) == []


def test_delta_series_has_points_and_open_mask():
    res = _res(schottky_enabled=True)
    s = series_hc_delta_vs_field(res)
    assert s and len(s[0].x) >= 1
    assert hasattr(s[0], "open_mask")                # hollow markers for undetermined Δ


def test_multifield_series_one_per_ok_group():
    res = _res(schottky_enabled=True)
    s = series_hc_schottky_multifield(res)
    assert len(s) >= 1


def test_f_series_gated():
    res = _res()
    assert series_hc_f_vs_field(res) == []


def test_f_series_has_points_when_enabled():
    res = _res(schottky_enabled=True)
    s = series_hc_f_vs_field(res)
    assert s and len(s[0].x) >= 1


def test_render_delta_returns_figure():
    res = _res(schottky_enabled=True, schottky_delta_h_model="zeeman")
    fig = render_hc_delta_vs_field([res])
    ax = fig.axes[0]
    assert "Oe" in ax.get_xlabel() or "Field" in ax.get_xlabel()
    assert "Δ" in ax.get_ylabel()


def test_render_f_returns_figure():
    res = _res(schottky_enabled=True)
    fig = render_hc_f_vs_field([res])
    assert fig.axes[0].get_ylabel() is not None


def test_alphaN_series_empty_without_nuclear():
    """alphaN is only populated from nuclear Schottky fits; fixture disables include_nuclear."""
    from cryosweep_core.plotting.catalog import series_hc_alphaN_vs_field
    res = _res(schottky_enabled=True)
    s = series_hc_alphaN_vs_field(res)
    assert s == []    # no nuclear data -> empty, no crash


def test_render_schottky_multifield_returns_figure():
    res = _res(schottky_enabled=True)
    fig = render_hc_schottky_multifield([res])
    assert fig.axes[0].get_xlabel() == "T (K)"
