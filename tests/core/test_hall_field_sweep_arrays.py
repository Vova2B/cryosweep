# tests/core/test_hall_field_sweep_arrays.py
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer


def test_field_sweep_arrays_persisted_and_consistent(hall_synth_path):
    rt = load_dat(hall_synth_path)
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5})
    res = HallAnalyzer().analyze(rt, cfg)
    assert res.status == "ok"
    pts = res.data["points"]
    assert len(pts) == 3
    for p in pts:
        # raw arrays present, finite-masked, length == n_points, span both field signs
        assert len(p["field_raw_T"]) == p["n_points"]
        assert len(p["R_xy_raw"]) == p["n_points"]
        assert min(p["field_raw_T"]) < 0 < max(p["field_raw_T"])
        assert all(np.isfinite(p["field_raw_T"])) and all(np.isfinite(p["R_xy_raw"]))
        # antisym arrays present and equal-length and positive-only |B|
        assert len(p["field_asym_T"]) == len(p["R_asym"]) >= 2
        assert min(p["field_asym_T"]) > 0
        # intercept persisted whenever Stage B ran
        assert p["antisymmetrized"] is True
        assert p["asym_intercept_ohm"] is not None
        # CONSISTENCY: re-fitting the persisted asym arrays reproduces the stored slope,
        # and slope*thickness*sign == R_H (the scalar path the analyzer already reports)
        slope_refit = float(np.polyfit(p["field_asym_T"], p["R_asym"], 1)[0])
        assert slope_refit == p["slope_ohm_per_T"] or abs(
            slope_refit - p["slope_ohm_per_T"]) <= 1e-9 * abs(p["slope_ohm_per_T"])
        assert abs(p["slope_ohm_per_T"] * (0.5e-3) * 1 - p["R_H"]) <= 1e-18

    # the grounded slope oracle (thickness-independent), T=10 K point
    p10 = next(p for p in pts if abs(p["temperature"] - 10.0) < 0.5)
    assert abs(p10["slope_ohm_per_T"] - (-5.000e-4)) <= 1e-6
    assert p10["r2"] >= 0.9999


def test_field_rxx_populated_with_longitudinal_channel(hall_synth_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.hall import HallAnalyzer
    res = HallAnalyzer().analyze(
        load_dat(hall_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5, "longitudinal_channel": 2}))
    pts = res.data["points"]
    assert pts
    for p in pts:
        assert p["field_rxx_T"], "longitudinal sweep must be populated"
        assert len(p["field_rxx_T"]) == len(p["R_xx_raw"])
        # same signed-B ordering as the transverse raw sweep (ch2 is finite everywhere here)
        assert p["field_rxx_T"] == p["field_raw_T"]


def test_field_rxx_empty_without_longitudinal_channel(hall_synth_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.hall import HallAnalyzer
    res = HallAnalyzer().analyze(
        load_dat(hall_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))
    for p in res.data["points"]:
        assert p["field_rxx_T"] == []
        assert p["R_xx_raw"] == []
