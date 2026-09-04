import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer, E_CHG

def _analyze(path, **hall):
    rt = load_dat(path)
    cfg = RunConfig.load(hall=hall)
    return HallAnalyzer().analyze(rt, cfg)

def test_error_without_hall_channel(hall_synth_path):
    res = _analyze(hall_synth_path)                       # no hall_channel
    assert res.status == "error"
    assert any("hall_channel" in e for e in res.errors)

def test_synthetic_recovers_R_H_n_mobility(hall_synth_path):
    res = _analyze(hall_synth_path, hall_channel=1, thickness_mm=0.1, longitudinal_channel=2)
    assert res.status == "ok"
    pts = res.data["points"]
    assert len(pts) == 3                                  # 3 held temperatures
    for p in pts:
        assert p["antisymmetrized"] is True
        assert p["R_H"] == pytest.approx(-5.0e-8, rel=2e-3)        # Stage B trusted value
        assert p["carrier_type"] == "electrons"
        assert p["carrier_n"] == pytest.approx(1.0/(E_CHG*5.0e-8), rel=2e-3)
        assert p["mobility"] == pytest.approx(0.05, rel=5e-3)
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["hall_coefficient"]["applicable"] is True
    assert caps["carrier_concentration"]["applicable"] is True
    assert caps["mobility"]["applicable"] is True

def test_no_thickness_is_low_confidence_slope_only(hall_synth_path):
    res = _analyze(hall_synth_path, hall_channel=1)       # no thickness
    assert res.status == "low_confidence"
    p = res.data["points"][0]
    assert p["slope_ohm_per_T"] is not None               # slope still computed
    assert p["R_H"] is None                                # but not R_H without thickness
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["hall_coefficient"]["applicable"] is False

def test_no_longitudinal_means_no_mobility(hall_synth_path):
    res = _analyze(hall_synth_path, hall_channel=1, thickness_mm=0.1)   # no longitudinal
    assert res.status == "ok"
    assert res.data["points"][0]["mobility"] is None
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["mobility"]["applicable"] is False
    assert "longitudinal" in caps["mobility"]["reason"]

def test_dispatch_probe_override_routes_hall(hall_synth_path):
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.analyzers.dispatch import analyze_file
    rt = load_dat(hall_synth_path)
    cfg = RunConfig.load(probe_override="hall", hall={"hall_channel": 1, "thickness_mm": 0.1})
    res = analyze_file(rt, cfg, build_default_registry())
    assert res.data["probe"] == "hall"
    # Synthetic file has known thickness=0.1 mm and clean fits — must be ok, not low_confidence
    assert res.status == "ok"

def test_two_file_longitudinal_mobility(hall_synth_path, hall_long_synth_path):
    # Hall channel from hall_synth (Bridge 1); rho_xx from a SEPARATE longitudinal file (Bridge 2).
    # hall_long_synth uses RHO_XX_LONG=2e-6 (distinct from hall_synth Bridge 2 RHO_XX=1e-6),
    # so expected mobility = |R_H| / rho_xx_long = 5e-8 / 2e-6 = 0.025.
    # If code fell back to same-file rho_xx=1e-6 it would yield 0.05 and this assertion fails.
    res = _analyze(hall_synth_path, hall_channel=1, thickness_mm=0.1,
                   longitudinal_channel=2, longitudinal_file=str(hall_long_synth_path))
    assert res.status == "ok"
    assert res.data["longitudinal_source"].startswith("file:")
    for p in res.data["points"]:
        assert p["mobility"] == pytest.approx(0.025, rel=5e-3)

def test_real_hall_file_analyzes(hall_real_path):
    import pathlib
    if not pathlib.Path(hall_real_path).exists():
        pytest.skip("real Hall measurement file not present (gitignored)")
    res = _analyze(hall_real_path, hall_channel=1, thickness_mm=0.1, longitudinal_channel=2)
    assert res.status in ("ok", "low_confidence")
    assert res.data["probe"] == "hall"
    pts = res.data["points"]
    assert len(pts) >= 5                              # several held temperatures
    # antisymmetrization must engage (real signal ~1% of even admixture)
    assert any(p["antisymmetrized"] for p in pts)
    # R_H finite and carrier metrics populated where antisymmetrized
    anti = [p for p in pts if p["antisymmetrized"] and p["R_H"] is not None]
    assert anti
    for p in anti:
        assert np.isfinite(p["R_H"]) and p["carrier_n"] is not None
    # Stage B must MATERIALLY differ from raw Stage A (admixture removed), not just float noise
    # Measured on the real Hall file: max rel_diff ~ 0.16 (16%) at 200 K; >1% threshold is safe and meaningful.
    rel_diffs = [abs(p["R_H"] - p["R_H_raw"]) / abs(p["R_H"])
                 for p in anti if p["R_H_raw"] is not None and p["R_H"]]
    assert rel_diffs and max(rel_diffs) > 0.01   # >1% change on at least one held-T loop
