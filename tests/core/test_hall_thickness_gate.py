"""R_H = slope x thickness. Without a thickness the product measures a slope, not a Hall
coefficient — a missing USER INPUT, which cryosweep's contract says must be `status:
"gated"` with a gate[] entry naming the flag (cryosweep/CLAUDE.md, "Missing inputs gate;
they do not guess"). It returned low_confidence with an EMPTY gate[] and no warning, so
the only trace was a capability `reason` string nothing surfaces.
"""
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer


def _no_thickness(analyzer, path):
    return analyzer.analyze(load_dat(path), RunConfig(hall={"hall_channel": 1}))


def _assert_gated_with_remedy(res):
    assert res.status == "gated", res.status
    needs = [g.need for g in res.gate]
    assert "thickness_mm" in needs, needs
    g = next(g for g in res.gate if g.need == "thickness_mm")
    assert g.remedy.get("flag") == "--thickness"
    assert "--thickness" in g.remedy.get("example", "")


def _assert_slope_only_work_survives(res):
    """Gating means 'tell the user what to supply', not 'discard the work'."""
    assert res.data.get("points"), "slope-only points must survive the gate"


def test_field_sweep_gates_on_missing_thickness(hall_synth_path):
    res = _no_thickness(HallAnalyzer(), hall_synth_path)
    _assert_gated_with_remedy(res)
    _assert_slope_only_work_survives(res)


def test_temp_dep_gates_on_missing_thickness(hall_tdep_synth_path):
    res = _no_thickness(HallTempDepAnalyzer(), hall_tdep_synth_path)
    _assert_gated_with_remedy(res)
    _assert_slope_only_work_survives(res)
