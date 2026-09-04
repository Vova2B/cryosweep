import json
from cryosweep_core.config import StabilityCfg, RunConfig

def test_defaults_and_load():
    cfg = RunConfig.load()
    assert cfg.unit_system == "CGS"
    assert cfg.confidence_min == 0.5
    assert cfg.stability.span_drift_ratio_min == 5.0
    assert cfg.stability.monotone_fraction_min == 0.8
    assert cfg.stability.min_segment_len == 8

def test_overrides_merge():
    cfg = RunConfig.load(unit_system="SI")
    assert cfg.unit_system == "SI"

def test_json_schema_is_valid():
    schema = RunConfig.model_json_schema()
    # round-trips as JSON and pins the detection knobs
    s = json.dumps(schema, sort_keys=True)
    assert "span_drift_ratio_min" in s
    assert "confidence_min" in s


def test_quality_subconfig_additive_and_loads_via_overrides():
    from cryosweep_core.config import RunConfig, QualityCfg
    c0 = RunConfig()
    assert c0.quality.exclude_outliers is False and c0.quality.outlier_k == 8.0
    c = RunConfig.load(quality={"exclude_outliers": True, "outlier_k": 6.0})
    assert c.quality.exclude_outliers is True and c.quality.outlier_k == 6.0
    assert RunConfig(**{"unit_system": "SI"}).quality.exclude_outliers is False
