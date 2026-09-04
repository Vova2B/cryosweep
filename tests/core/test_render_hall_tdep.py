"""Smoke tests for hall_tdep renderers (Task 9)."""
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from cryosweep_core.plotting.render import render_kind, default_kind_for


def _res(hall_tdep_synth_path):
    return HallTempDepAnalyzer().analyze(
        load_dat(hall_tdep_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05, "longitudinal_channel": 2}),
    )


def test_default_kind_for_hall_tdep():
    assert default_kind_for("hall_tdep") == "hall_tdep_RH_T"


def test_render_hall_tdep_kinds(hall_tdep_synth_path):
    res = _res(hall_tdep_synth_path)
    for kind in [
        "hall_tdep_RH_T",
        "hall_tdep_n_T",
        "hall_tdep_mobility_T",
        "hall_tdep_asym_vs_B",
        "hall_tdep_stages",
    ]:
        fig = render_kind(res, kind)
        assert fig is not None, f"render_kind returned None for {kind}"
        assert len(fig.axes) >= 1, f"no axes for {kind}"
        assert len(fig.axes[0].lines) >= 1, f"no lines plotted for {kind}"
