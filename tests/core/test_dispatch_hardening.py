import itertools
import json
import pathlib
import subprocess
import sys
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"

def test_unknown_probe_override_errors_not_misroutes():
    rt = load_dat(FIX / "vsm_synth.dat")
    res = analyze_file(rt, RunConfig.load(probe_override="nope"), build_default_registry())
    assert res.status == "error"
    assert "nope" in (res.errors[0] if res.errors else "")
    assert res.data.get("probe") != "vsm"          # must NOT be silently routed to VSM

class _BoomAnalyzer:
    probe = "boom"
    def analyze(self, rt, cfg):
        raise RuntimeError("kaboom")

class _BoomRegistry:
    def get_analyzer(self, key):
        return _BoomAnalyzer() if key == "boom" else None

def test_analyzer_exception_becomes_error_result():
    rt = load_dat(FIX / "vsm_synth.dat")
    res = analyze_file(rt, RunConfig.load(probe_override="boom"), _BoomRegistry())
    assert res.status == "error"
    assert "boom" in res.errors[0] and "kaboom" in res.errors[0]


_FIXTURES = ["vsm_synth.dat", "hall_synth.dat", "hall_long_synth.dat"]
_PROBES = ["vsm", "heatcapacity", "resistivity", "hall"]

@pytest.mark.parametrize("fixture,probe", list(itertools.product(_FIXTURES, _PROBES)))
def test_no_analyzer_raises_on_any_fixture_x_probe(fixture, probe):
    rt = load_dat(FIX / fixture)
    cfg = RunConfig.load(probe_override=probe)        # hall w/o channel -> error Result, still no raise
    res = analyze_file(rt, cfg, build_default_registry())   # must not raise
    assert res.status in ("ok", "gated", "low_confidence", "error")
    assert isinstance(res.data, dict)


_REPO = pathlib.Path(__file__).resolve().parents[2]

def test_cli_mismatched_probe_is_graceful_error():
    # hall_synth.dat has no 'moment' column; forcing --probe vsm must give a clean error envelope + exit 2,
    # never a traceback / empty stdout.
    p = subprocess.run([sys.executable, "-m", "cryosweep_cli", "analyze",
                        "tests/core/fixtures/hall_synth.dat", "--probe", "vsm"],
                       cwd=_REPO, capture_output=True, text=True)
    assert p.returncode == 2, p.stderr
    out = json.loads(p.stdout)
    assert out["status"] == "error"
