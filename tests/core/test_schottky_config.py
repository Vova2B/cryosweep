from cryosweep_core.config import RunConfig


def test_schottky_defaults_off():
    cfg = RunConfig.load().heatcapacity
    assert cfg.schottky_enabled is False
    assert cfg.schottky_r == 1.0
    assert cfg.schottky_fit_max_k == 15.0
    assert cfg.schottky_nuclear_max_tmin_k == 2.5
    assert cfg.schottky_delta_h_model == "none"


def test_schottky_roundtrip():
    cfg = RunConfig.model_validate({"heatcapacity": {"schottky_enabled": True,
                                                     "schottky_delta_h_model": "zeeman"}})
    assert cfg.heatcapacity.schottky_enabled is True
    assert cfg.heatcapacity.schottky_delta_h_model == "zeeman"
