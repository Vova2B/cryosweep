from cryosweep_core.config import RunConfig, SampleGeometry

def test_geometry_default_is_incomplete():
    assert SampleGeometry().complete() is False

def test_geometry_complete_requires_all_positive():
    assert SampleGeometry(width_mm=1.0, thickness_mm=0.5, length_mm=2.0).complete() is True
    assert SampleGeometry(width_mm=1.0, thickness_mm=0.0, length_mm=2.0).complete() is False
    assert SampleGeometry(width_mm=1.0, thickness_mm=0.5).complete() is False

def test_runconfig_carries_geometry():
    cfg = RunConfig.load(geometry={"width_mm": 1.0, "thickness_mm": 0.5, "length_mm": 2.0})
    assert cfg.geometry.complete() is True

def test_hall_config_defaults_and_load():
    from cryosweep_core.config import RunConfig, HallCfg
    assert HallCfg().hall_channel is None
    cfg = RunConfig.load(hall={"hall_channel": 1, "thickness_mm": 0.1,
                               "longitudinal_channel": 2, "geometry_sign": -1})
    assert cfg.hall.hall_channel == 1
    assert cfg.hall.thickness_mm == 0.1
    assert cfg.hall.longitudinal_channel == 2
    assert cfg.hall.geometry_sign == -1
    assert cfg.model_dump(mode="json")["hall"]["hall_channel"] == 1   # round-trips into provenance
