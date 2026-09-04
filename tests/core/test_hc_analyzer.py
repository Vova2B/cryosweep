import numpy as np, pathlib
from cryosweep_core.analyzers.hc import HCAnalyzer, HCData
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig

def test_hc_analyze_returns_ok_with_gamma_beta_theta(hc_path):
    res = HCAnalyzer().analyze(load_dat(hc_path), RunConfig.load())
    assert res.status in ("ok", "low_confidence")
    p = res.data["fit"]["params"]
    assert p["gamma"] == p["gamma"]            # not NaN
    assert p["beta"] > 0
    assert p["theta_D"] > 0
    assert res.provenance.sha256

def test_hcdata_schema():
    props = HCData.model_json_schema()["properties"]
    for f in ("temperature", "cp", "cp_over_t", "t_squared", "fit"):
        assert f in props
