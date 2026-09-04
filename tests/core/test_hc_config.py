# tests/core/test_hc_config.py
from cryosweep_core.config import RunConfig


def test_heatcapacity_defaults():
    cfg = RunConfig()
    hc = cfg.heatcapacity
    assert hc.full_init["theta_E1"] == 50.0 and hc.full_init["theta_E2"] == 150.0
    assert hc.full_init["m1"] == 1.0 and hc.full_init["m2"] == 2.0
    assert hc.full_fixed["n"] is True
    assert hc.full_max_t_min_k == 50.0 and hc.full_min_points == 15
    assert cfg.hc_parsimony_r2 == 0.99    # unchanged canonical field


def test_heatcapacity_override_load():
    cfg = RunConfig.load(heatcapacity={"full_min_points": 8})
    assert cfg.heatcapacity.full_min_points == 8


def test_multifield_config_defaults():
    from cryosweep_core.config import RunConfig
    hc = RunConfig.load().heatcapacity
    assert hc.field_bin_koe == 1.0
    assert hc.min_lowt_per_field == 5
    assert hc.identifiability_rel_sigma == 1.0
    assert hc.bound_rail_frac == 0.01
    assert hc.corr_warn == 0.99
