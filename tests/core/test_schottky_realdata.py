from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry


def _res(path):
    cfg = RunConfig.model_validate({"heatcapacity": {
        "schottky_enabled": True, "schottky_delta_h_model": "zeeman"}})
    return analyze_file(load_dat(str(path)), cfg, build_default_registry())


def test_realdata_determinism_gate_not_overclaiming(hc_path):
    res = _res(hc_path)
    fg = res.data["field_groups"]
    zero_field = [g for g in fg if g.get("field_oe", 1e9) < 50]
    for g in zero_field:
        if g.get("status") == "ok" and "schottky" in g:
            assert g["schottky"]["delta_determined"] is False

    determined = [g for g in fg if g.get("status") == "ok"
                  and g.get("schottky", {}).get("delta_determined")]
    assert len(determined) <= 2

    ov = res.data.get("schottky_overlay")
    assert ov is None or not ov.get("ok")
