import json, subprocess, sys, pathlib
FIX = str(pathlib.Path(__file__).parent / "fixtures" / "vsm_synth.dat")
ROOT = pathlib.Path(__file__).resolve().parents[2]

def _run(*args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args], capture_output=True, text=True, cwd=ROOT)

def test_cli_analyze_emits_result_json():
    r = _run("analyze", FIX)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["status"] == "ok"
    assert abs(payload["data"]["fit"]["params"]["C"] - 0.5) < 0.02

def test_cli_analyze_si_mu_eff():
    # Bug 1: SI unit system must give the same physical mu_eff (~1.999), not 564.
    r = _run("analyze", FIX, "--unit-system", "SI")
    assert r.returncode == 0, r.stderr
    fit = json.loads(r.stdout)["data"]["fit"]
    assert abs(fit["params"]["mu_eff"] - 1.999) < 0.02
    assert fit["units"]["C"] == "m^3*K/mol"


def test_cli_detect():
    r = _run("detect", FIX)
    assert r.returncode == 0
    assert json.loads(r.stdout)["data"]["probe"] == "vsm"

def test_cli_export_writes_files(tmp_path):
    r = _run("export", FIX, "--out", str(tmp_path / "out"))
    assert r.returncode == 0
    assert (tmp_path / "out.points.csv").exists()
