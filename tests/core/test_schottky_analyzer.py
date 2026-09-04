# tests/core/test_schottky_analyzer.py
import pathlib, pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

FIX = pathlib.Path(__file__).parent / "fixtures"

def _res(cfg):
    return analyze_file(load_dat(str(FIX / "hc_schottky_synth.dat")), cfg, build_default_registry())

def _cfg(**hc):
    return RunConfig.model_validate({"heatcapacity": {**hc}})

def test_off_by_default_no_schottky_key():
    res = _res(RunConfig.load())
    assert res.data["schottky_enabled"] is False
    for g in res.data["field_groups"]:
        assert "schottky" not in g                       # nothing computed when off

def test_enabled_fits_delta_per_field_and_recovers_higher_field():
    res = _res(_cfg(schottky_enabled=True))
    fg = res.data["field_groups"]
    determined = [(g["field_oe"], g["schottky"]) for g in fg
                  if g.get("status") == "ok" and g["schottky"]["delta_determined"]]
    assert determined, "expected at least one determined Δ at higher field"
    # Δ rises with field (Zeeman): highest determined field has the largest Δ
    fields = [f for f, _ in determined]; deltas = [s["params"]["Delta"] for _, s in determined]
    assert deltas[fields.index(max(fields))] == max(deltas)

def test_lowest_field_flagged_kramers():
    res = _res(_cfg(schottky_enabled=True))
    prim = res.data["field_groups"][0]
    assert any("kramers" in w.lower() for w in prim["schottky"]["warnings"])

def test_overlay_recovers_g_when_zeeman():
    res = _res(_cfg(schottky_enabled=True, schottky_delta_h_model="zeeman"))
    ov = res.data["schottky_overlay"]
    assert ov and ov["ok"] and ov["model"] == "zeeman"
    assert ov["g_factor"] == pytest.approx(2.0, rel=0.15)

def test_overlay_none_by_default():
    res = _res(_cfg(schottky_enabled=True))
    assert res.data["schottky_overlay"] is None or res.data["schottky_overlay"]["model"] == "none"
