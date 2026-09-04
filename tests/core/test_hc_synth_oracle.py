import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

def test_hc_synth_detects_and_recovers_known_values(hc_synth_path):
    res = analyze_file(load_dat(str(hc_synth_path)), RunConfig.load(), build_default_registry())
    assert res.data["probe"] == "heatcapacity"
    assert res.status == "ok"
    p = res.data["fit"]["params"]
    assert p["gamma"] == pytest.approx(0.010, rel=1e-3)
    assert p["beta"] == pytest.approx(5.0e-4, rel=1e-3)
    assert p["theta_D"] == pytest.approx(226.777, rel=1e-3)
