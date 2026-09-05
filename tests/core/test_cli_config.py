"""--config FILE: a RunConfig JSON reaches the CLI (2026-09-05).

Before this flag the CLI could set only unit_system / geometry / hall / probe_override
(__main__.py override block), so every other RunConfig knob — the whole heatcapacity
surface (Schottky, transition search, fit windows, entropy source) and
quality.exclude_outliers — was reachable from the GUI but NOT headless.
`cryosweep schema config` already publishes the file's schema.

Precedence: explicit CLI flags override the file, merged PER KEY inside the nested
geometry/hall sub-configs (a config file's hall.temp_interval survives --hall-channel).
"""
import json, pathlib, subprocess, sys

FIX = pathlib.Path(__file__).parent / "fixtures"
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run(args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args],
                          capture_output=True, text=True, cwd=ROOT)


def test_config_unlocks_schottky(tmp_path):
    cf = tmp_path / "cfg.json"
    cf.write_text(json.dumps({"heatcapacity": {"schottky_enabled": True}}))
    base = _run(["analyze", str(FIX / "hc_schottky_synth.dat")])
    with_cfg = _run(["analyze", str(FIX / "hc_schottky_synth.dat"), "--config", str(cf)])
    assert base.returncode in (0, 11), base.stderr
    assert with_cfg.returncode in (0, 11), with_cfg.stderr
    d0 = json.loads(base.stdout)["data"]
    d1 = json.loads(with_cfg.stdout)["data"]
    assert d0["schottky_enabled"] is False
    assert d1["schottky_enabled"] is True
    # the behaviour differs, not just the echo: the Schottky fit actually ran per field group
    assert any("schottky" in g for g in d1["field_groups"])
    assert not any("schottky" in g for g in d0["field_groups"])


def test_config_sets_quality_exclude_outliers(tmp_path):
    cf = tmp_path / "cfg.json"
    cf.write_text(json.dumps({"quality": {"exclude_outliers": True}}))
    r = _run(["analyze", str(FIX / "act_synth.dat"), "--config", str(cf)])
    assert r.returncode in (0, 11), r.stderr
    prov_cfg = json.loads(r.stdout)["provenance"]["config"]
    assert prov_cfg["quality"]["exclude_outliers"] is True


def test_config_flag_precedence_per_key(tmp_path):
    cf = tmp_path / "cfg.json"
    cf.write_text(json.dumps({"unit_system": "SI", "hall": {"temp_interval": 2.5}}))
    r = _run(["analyze", str(FIX / "act_synth.dat"), "--config", str(cf),
              "--unit-system", "CGS", "--hall-channel", "1"])
    assert r.returncode in (0, 11), r.stderr
    prov_cfg = json.loads(r.stdout)["provenance"]["config"]
    assert prov_cfg["unit_system"] == "CGS"            # explicit flag beats the file
    assert prov_cfg["hall"]["temp_interval"] == 2.5    # file key survives the hall merge
    assert prov_cfg["hall"]["hall_channel"] == 1       # flag key merged in


def test_config_file_wins_over_flag_default(tmp_path):
    cf = tmp_path / "cfg.json"
    cf.write_text(json.dumps({"unit_system": "SI"}))
    r = _run(["analyze", str(FIX / "act_synth.dat"), "--config", str(cf)])
    assert r.returncode in (0, 11), r.stderr
    assert json.loads(r.stdout)["provenance"]["config"]["unit_system"] == "SI"


def test_config_unknown_key_warns(tmp_path):
    cf = tmp_path / "cfg.json"
    cf.write_text(json.dumps({"heatcapacity": {"schotky_enabled": True}}))   # typo
    r = _run(["analyze", str(FIX / "act_synth.dat"), "--config", str(cf)])
    assert r.returncode in (0, 11), r.stderr
    env = json.loads(r.stdout)
    assert any("--config" in w and "heatcapacity.schotky_enabled" in w for w in env["warnings"])
    assert "heatcapacity.schotky_enabled" in r.stderr


def test_config_bad_file_error_envelope(tmp_path):
    r = _run(["analyze", str(FIX / "act_synth.dat"), "--config", str(tmp_path / "nope.json")])
    assert r.returncode == 2
    assert json.loads(r.stdout)["status"] == "error"
