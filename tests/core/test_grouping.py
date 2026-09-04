import math
import types
import numpy as np
import pytest
from cryosweep_core.grouping import setpoint_key, group_segments_by_setpoint


def _seg(temp, idx=(0, 1, 2)):
    # lightweight Segment stand-in: grouping reads .setpoint and .idx only
    return types.SimpleNamespace(setpoint={"temperature": temp},
                                 idx=np.asarray(idx, int), direction=-1)


def test_setpoint_key_high_T_round_to_nearest_merges_drift():
    assert setpoint_key(199.9) == 200.0
    assert setpoint_key(200.0) == 200.0
    assert setpoint_key(199.994) == 200.0


def test_setpoint_key_is_round_half_up_not_bankers():
    assert setpoint_key(11.5) == 12.0
    assert setpoint_key(12.5) == 13.0
    assert setpoint_key(199.5) == 200.0
    assert setpoint_key(200.5) == 201.0


def test_setpoint_key_low_T_nearest_half_integer():
    assert setpoint_key(2.0) == 2.0
    assert setpoint_key(5.0) == 5.0
    assert setpoint_key(4.5) == 4.5
    assert setpoint_key(9.6) == 9.5
    assert setpoint_key(9.997) == 10.0
    assert setpoint_key(2.0) != setpoint_key(5.0)


def test_setpoint_key_threshold_boundary():
    assert setpoint_key(10.0) == 10.0
    assert setpoint_key(10.4) == 10.0
    assert setpoint_key(9.99) == 10.0


def test_setpoint_key_non_finite_returns_nan():
    assert math.isnan(setpoint_key(float("nan")))
    assert math.isnan(setpoint_key(float("inf")))


def test_setpoint_key_cross_threshold_values_merge_at_boundary():
    # values just below threshold that round up to the integer boundary must merge
    # with values at/above it (this is the intended high-T-drift merge behavior)
    assert setpoint_key(9.997) == setpoint_key(10.0) == 10.0


def test_group_segments_sorts_and_merges_near_duplicates():
    segs = [_seg(200.0), _seg(5.0), _seg(199.9)]
    groups = group_segments_by_setpoint(segs, "temperature", 10.0)
    keys = [k for k, _ in groups]
    assert keys == [5.0, 200.0]
    g200 = dict(groups)[200.0]
    assert len(g200) == 2


def test_group_segments_skips_missing_or_nonfinite_setpoint():
    s_ok = _seg(5.0)
    s_none = types.SimpleNamespace(setpoint={}, idx=np.array([0]), direction=1)
    s_nan = _seg(float("nan"))
    groups = group_segments_by_setpoint([s_ok, s_none, s_nan], "temperature", 10.0)
    assert [k for k, _ in groups] == [5.0]
    assert len(dict(groups)[5.0]) == 1


def test_quality_cfg_has_dqb_defaults_and_old_json_validates():
    from cryosweep_core.config import RunConfig, QualityCfg
    q = QualityCfg()
    assert q.setpoint_threshold_k == 10.0
    assert q.setpoint_unstable_k == 0.5
    assert q.setpoint_near_dup_k == 0.5
    # additive: a config written before DQ-B (no setpoint_* keys) still validates
    cfg = RunConfig.load(quality={"exclude_outliers": True, "outlier_k": 8.0})
    assert cfg.quality.setpoint_threshold_k == 10.0       # default filled in
