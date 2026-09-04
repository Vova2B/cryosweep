import subprocess, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = str(ROOT / "tests/core/fixtures/vsm_synth.dat")


def _run(*a):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *a], capture_output=True, text=True, cwd=ROOT)


def test_analyze_output_is_byte_deterministic():
    a = _run("analyze", FIX).stdout
    b = _run("analyze", FIX).stdout
    assert a == b and a.strip()                       # identical bytes, non-empty


def test_schema_output_deterministic():
    assert _run("schema", "result").stdout == _run("schema", "result").stdout


def test_hc_entropy_arrays_byte_identical(hc_path):
    """A second analyze() on the same input yields byte-identical entropy arrays (JSON repr)."""
    import json
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.analyzers.hc import HCAnalyzer
    from cryosweep_core.config import RunConfig
    hc = hc_path
    keys = ("entropy_temperature", "entropy_total", "entropy_magnetic",
            "entropy_rln_suggestion", "entropy_lattice_source", "entropy_reason")
    a = HCAnalyzer().analyze(load_dat(hc), RunConfig.load()).data
    b = HCAnalyzer().analyze(load_dat(hc), RunConfig.load()).data
    for k in keys:
        assert json.dumps(a[k]) == json.dumps(b[k]), f"{k} not deterministic"
    for ga, gb in zip(a["field_groups"], b["field_groups"]):
        assert json.dumps(ga["entropy"]) == json.dumps(gb["entropy"])
