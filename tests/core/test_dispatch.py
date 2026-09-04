from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
import pathlib
VSM = pathlib.Path("tests/core/fixtures/vsm_synth.dat")

def test_dispatch_routes_by_probe(hc_path):
    reg = build_default_registry()
    r_hc = analyze_file(load_dat(hc_path), RunConfig.load(), reg)
    assert r_hc.data["probe"] == "heatcapacity"
    r_vsm = analyze_file(load_dat(VSM), RunConfig.load(), reg)
    assert r_vsm.data["probe"] == "vsm"

def test_dispatch_unknown_probe_errors(tmp_path):
    # a file no analyzer claims -> error Result, not crash
    p = tmp_path / "x.dat"
    p.write_text("[Header]\nBYAPP,Unknown,1,1\n[Data]\nA,B\n1,2\n")
    r = analyze_file(load_dat(p), RunConfig.load(), build_default_registry())
    assert r.status == "error"
