import types
import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer, _duplicate_setpoint_diagnostics


def _analyze(res_path):
    # Grouping oracles cover BOTH channels; disable Hall-channel routing (the QD example's
    # Ch2 is Hall-wired and would otherwise be excluded at defaults per the 2026-07-02 spec).
    cfg = RunConfig.load(resistivity={"exclude_hall_channel": False})
    return ResistivityAnalyzer().analyze(load_dat(res_path), cfg)


def test_field_ramps_collapse_to_one_loop_per_setpoint(res_path):
    res = _analyze(res_path)
    by_ch = {b["channel"]: b for b in res.data["bridges"]}
    assert len(by_ch[1]["rho_h_curves"]) == 9
    assert len(by_ch[2]["rho_h_curves"]) == 9
    for b in res.data["bridges"]:
        for c in b["rho_h_curves"]:
            assert c["direction"] == 0
    keys = sorted(round(c["held_temp_k"]) for c in by_ch[1]["rho_h_curves"])
    assert keys == [2, 5, 10, 15, 20, 50, 100, 200, 300]


def test_grouped_300k_loop_carries_widest_ramp_mr(res_path):
    res = _analyze(res_path)
    by_ch = {b["channel"]: b for b in res.data["bridges"]}

    def loop300(b):
        return next(c for c in b["rho_h_curves"]
                    if round(c["held_temp_k"]) == 300 and c["direction"] == 0)
    c1, c2 = loop300(by_ch[1]), loop300(by_ch[2])
    assert c1["mr_percent_at_max_field"] == pytest.approx(3.51, abs=0.3)
    assert c2["mr_percent_at_max_field"] == pytest.approx(29.46, abs=0.5)
    assert min(c2["field"]) < -1000 and max(c2["field"]) > 1000


def test_single_segment_group_matches_per_ramp_build():
    import types
    import numpy as np
    from cryosweep_core.analyzers.resistivity import _build_h_curve_grouped
    # a one-segment group: grouping is a no-op; the combined curve must equal the lone ramp
    H = np.linspace(-1000.0, 1000.0, 41)
    rho = 1.0e-6 * (1.0 + (H / 1000.0) ** 2)   # rho0=1e-6, rho(1000)=2e-6 -> MR=100%
    seg = types.SimpleNamespace(idx=np.arange(H.size), direction=-1,
                                setpoint={"temperature": 5.0})
    c = _build_h_curve_grouped(H, rho, [seg], 5.0, None)
    assert c.direction == 0 and c.held_temp_k == 5.0
    assert c.n_points == 41
    assert c.mr_percent_at_max_field == pytest.approx(100.0, rel=1e-3)
    assert c.rho_zero_field == pytest.approx(1.0e-6, rel=1e-6)


def _fseg(temp):
    return types.SimpleNamespace(setpoint={"temperature": temp},
                                 idx=np.array([0, 1, 2]), direction=-1)


def test_real_file_has_no_spurious_setpoint_warnings(res_path):
    res = _analyze(res_path)
    dups = [d for d in res.diagnostics if d.kind == "duplicate_setpoints"]
    assert dups == []


def test_unstable_hold_fires_on_wide_within_group_spread():
    cfg = RunConfig()
    # 199.6 and 200.4 both round to integer key 200 (>= threshold); raw span 0.8 K > 0.5 K.
    # (199.0/201.0 would land in *separate* integer bins, so they cannot test within-group spread;
    #  0.8 K within one bin is the design's validated "wrong-merge / drift" case.)
    segs = [_fseg(199.6), _fseg(200.4)]
    diags = _duplicate_setpoint_diagnostics(segs, cfg)
    kinds = [d for d in diags if "unstable" in d.message]
    assert kinds and kinds[0].kind == "duplicate_setpoints" and kinds[0].severity == "warning"
    assert kinds[0].data["spread"] == pytest.approx(0.8)


def test_near_duplicate_fires_on_boundary_straddle():
    cfg = RunConfig()
    segs = [_fseg(9.7), _fseg(10.1)]      # 9.7 -> key 9.5, 10.1 -> key 10.0; raw gap 0.4 < 0.5
    diags = _duplicate_setpoint_diagnostics(segs, cfg)
    near = [d for d in diags if "near-duplicate" in d.message]
    assert near and near[0].data["setpoints"] == [9.5, 10.0]


def test_diagnostics_silent_on_well_separated_setpoints():
    cfg = RunConfig()
    segs = [_fseg(2.0), _fseg(5.0), _fseg(10.0)]   # gaps 3 / 5 K
    assert _duplicate_setpoint_diagnostics(segs, cfg) == []
