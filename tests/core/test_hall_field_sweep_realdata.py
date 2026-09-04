# tests/core/test_hall_field_sweep_realdata.py
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.plotting.catalog import (
    series_hall_rxy_vs_B, series_hall_asym_vs_B, series_hall_raw_vs_asym)


def test_real_field_sweep_series_nonempty(hall_real_path):
    if hall_real_path is None:
        pytest.skip("real Hall measurement file not present (gitignored)")
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file not present (gitignored)")
    rt = load_dat(hall_real_path)
    res = HallAnalyzer().analyze(rt, RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))
    assert res.data.get("points")
    assert series_hall_rxy_vs_B(res)
    assert series_hall_asym_vs_B(res)
    assert series_hall_raw_vs_asym(res)
