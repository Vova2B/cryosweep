# tests/core/test_series_hall_field_sweep.py
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.plotting.catalog import (
    series_hall_rxy_vs_B, series_hall_asym_vs_B, series_hall_raw_vs_asym)


def _res(hall_synth_path):
    rt = load_dat(hall_synth_path)
    return HallAnalyzer().analyze(rt, RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))


def test_series_rxy_vs_b_one_per_temperature(hall_synth_path):
    s = series_hall_rxy_vs_B(_res(hall_synth_path))
    assert len(s) == 3                                   # one Series per held T
    groups = {ser.group for ser in s}
    assert groups == {"10.0K", "100.0K", "300.0K"}
    for ser in s:
        assert ser.default_on and len(ser.x) == len(ser.y) > 0
        assert min(ser.x) < 0 < max(ser.x)              # signed field axis


def test_series_asym_vs_b_positive_axis(hall_synth_path):
    s = series_hall_asym_vs_B(_res(hall_synth_path))
    assert len(s) == 3
    for ser in s:
        assert ser.key.startswith("asym:")
        assert len(ser.x) == len(ser.y) >= 2
        assert min(ser.x) > 0                            # |B| only


def test_series_raw_vs_asym_triplet_per_temperature(hall_synth_path):
    s = series_hall_raw_vs_asym(_res(hall_synth_path))
    # +branch, -branch (reflected), asym -> 3 series per T, all positive x
    assert len(s) == 9
    keys = {ser.key.split(":")[0] for ser in s}
    assert keys == {"rawpos", "rawneg", "asym"}
    for ser in s:
        assert all(x > 0 for x in ser.x)
