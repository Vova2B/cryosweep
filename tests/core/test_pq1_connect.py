import pathlib
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.render import render_kind, NON_DATA_GIDS

FIX = pathlib.Path(__file__).parent / "fixtures"

def _res():
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def _hall_res():   # hall_synth.dat carries field loops -> backs resistivity_mr (see test_render_resistivity.py)
    return analyze_file(load_dat(str(FIX / "hall_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_connect_on_for_rho_t_default():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle()).axes[0]
    data_lines = [l for l in ax.lines if l.get_gid() not in NON_DATA_GIDS]
    assert data_lines and all(l.get_linestyle() == "-" for l in data_lines)
    # x is sorted ascending on each connected series
    for l in data_lines:
        x = np.asarray(l.get_xdata(), float)
        assert np.all(np.diff(x) >= 0)

def test_connect_off_toggle():
    ax = render_kind(_res(), "resistivity_rho_t", PlotSpec(), GlobalStyle(connect_lines=False)).axes[0]
    data_lines = [l for l in ax.lines if l.get_gid() not in NON_DATA_GIDS]
    assert all(l.get_linestyle() == "None" for l in data_lines)

def test_field_loop_kind_stays_markers_only():
    # resistivity_mr is NOT in the allowlist -> markers-only even with connect default on.
    # Use hall_synth.dat: it carries the field loops resistivity_mr needs (act_synth is T-ramp only).
    ax = render_kind(_hall_res(), "resistivity_mr", PlotSpec(), GlobalStyle()).axes[0]
    data_lines = [l for l in ax.lines if l.get_gid() not in NON_DATA_GIDS]
    assert data_lines and all(l.get_linestyle() == "None" for l in data_lines)

def test_no_new_artist_count_unchanged():
    n_off = len([l for l in render_kind(_res(), "resistivity_rho_t", PlotSpec(),
                 GlobalStyle(connect_lines=False)).axes[0].lines])
    n_on = len([l for l in render_kind(_res(), "resistivity_rho_t", PlotSpec(),
                 GlobalStyle(connect_lines=True)).axes[0].lines])
    assert n_on == n_off
