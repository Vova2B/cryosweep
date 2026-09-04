import json, subprocess, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]

def _run(args):
    p = subprocess.run([sys.executable, "-m", "cryosweep_cli", *args], cwd=REPO,
                       capture_output=True, text=True)
    return p

def test_cli_analyze_resistivity_ok(res_path):
    p = _run(["analyze", str(res_path)])
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)["data"]
    assert data["probe"] == "resistivity"
    assert sorted(b["channel"] for b in data["bridges"]) == [1]   # Ch2 routed out (Hall-wired)
    assert data["excluded_hall_channel"] == 2
    assert data["rho_source"] == "instrument_column"

def test_cli_analyze_resistivity_geometry_flags(res_path):
    p = _run(["analyze", str(res_path), "--width-mm", "1.0", "--thickness-mm", "0.1", "--length-mm", "2.0"])
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)["data"]
    assert data["rho_source"] == "geometry"

def test_cli_schema_resistivity():
    p = _run(["schema", "analyze:resistivity"])
    assert p.returncode == 0, p.stderr
    sch = json.loads(p.stdout)
    assert "bridges" in sch["properties"]

def test_cli_export_resistivity_writes_files(tmp_path, res_path):
    out = tmp_path / "rescli"
    p = _run(["export", str(res_path), "--out", str(out)])
    assert p.returncode == 0, p.stderr
    exported = json.loads(p.stdout)["data"]["exported"]
    assert pathlib.Path(exported["curves"]).exists()
    assert pathlib.Path(exported["derived"]).exists()
