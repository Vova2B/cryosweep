import pathlib, pytest
from tests.core.conftest import real_data, require_manifest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
from cryosweep_core.plotting.render import render_for, render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}

def _hall():
    return analyze_file(load_dat(str(FIX / "hall_synth.dat")),
                        RunConfig.load(probe_override="hall", hall={"hall_channel": 1, "thickness_mm": 0.5}),
                        build_default_registry())

def test_render_for_hall_is_rh_t():
    fig = render_for(_hall(), PlotSpec())
    ax = fig.axes[0]
    assert "R_H" in ax.get_ylabel() and ax.get_xlabel() == "Temperature (K)"
    assert len(ax.lines) >= 1

def test_mobility_kind_gated_when_absent():
    res = _hall()                                      # no longitudinal data -> no mobility
    if KINDS["hall_mobility_t"].series(res) == []:
        with pytest.raises(ValueError):
            render_kind(res, "hall_mobility_t", PlotSpec())
    else:
        assert len(render_kind(res, "hall_mobility_t", PlotSpec()).axes[0].lines) >= 1


def test_degenerate_rh_axis_is_padded():
    """R_H constant to ~3e-15 relative leaves the axis with no scale; AutoLocator +
    ScalarFormatter then emit a 17-significant-digit tick label ('-3.0000000000000004').
    Measured on both affected entries at rel-span 3.31e-15."""
    require_manifest()
    import pathlib, sys
    TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import matplotlib.pyplot as plt
    import pq_compare

    for eid in ("hall_tdep_summary", "hall_tdep_rh_n_twin"):
        entry = [e for e in pq_compare._load_manifest() if e.get("id") == eid][0]
        fig, status = pq_compare._render_v2(entry)
        assert fig is not None, status
        fig.canvas.draw()
        fails = [f for f in pq_compare._check_fig(entry, fig, status) if "DEGENERATE" in f]
        plt.close(fig)
        assert not fails, (eid, fails)


def test_padding_is_noop_on_healthy_axis():
    """The pad must not touch an axis that already has a real span."""
    if real_data("res") is None:
        pytest.skip("local-only measurement file for key 'res' is not available")
    import pathlib, sys
    TOOLS = pathlib.Path(__file__).resolve().parents[2] / "tools"
    if str(TOOLS) not in sys.path:
        sys.path.insert(0, str(TOOLS))
    import matplotlib.pyplot as plt
    import pq_compare

    entry = [e for e in pq_compare._load_manifest() if e.get("id") == "hall_rh_t"][0]
    fig, status = pq_compare._render_v2(entry)
    assert fig is not None, status
    fig.canvas.draw()
    fails = [f for f in pq_compare._check_fig(entry, fig, status) if "DEGENERATE" in f]
    plt.close(fig)
    assert not fails, fails
