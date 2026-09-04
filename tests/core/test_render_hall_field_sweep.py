# tests/core/test_render_hall_field_sweep.py
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.plotting.render import render_kind   # dispatch: render_kind(results, kind_key, spec=None, style=None, overlay=None)


def _res(hall_synth_path):
    rt = load_dat(hall_synth_path)
    return HallAnalyzer().analyze(rt, RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))


@pytest.mark.parametrize("kind", ["hall_rxy_vs_B", "hall_asym_vs_B", "hall_raw_vs_asym"])
def test_render_kind_produces_nonempty_axes(hall_synth_path, kind):
    fig = render_kind(_res(hall_synth_path), kind)
    ax = fig.axes[0]
    assert len(ax.lines) > 0                              # markers drawn


def test_asym_render_adds_fit_lines(hall_synth_path):
    # 3 asym marker series + 3 fit lines = 6 line artists
    fig = render_kind(_res(hall_synth_path), "hall_asym_vs_B")
    assert len(fig.axes[0].lines) == 6


def test_asym_render_fit_line_off_when_disabled(hall_synth_path):
    from cryosweep_core.plotting.spec import PlotSpec
    fig = render_kind(_res(hall_synth_path), "hall_asym_vs_B", spec=PlotSpec(fit_line=False))
    assert len(fig.axes[0].lines) == 3                   # markers only, no fit lines
