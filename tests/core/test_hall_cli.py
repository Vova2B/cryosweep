import json, subprocess, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
SYNTH = "tests/core/fixtures/hall_synth.dat"

def _run(args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args], cwd=REPO, capture_output=True, text=True)

def test_cli_hall_subcommand():
    p = _run(["hall", SYNTH, "--hall-channel", "1", "--thickness", "0.1",
              "--thickness-unit", "mm", "--long-channel", "2"])
    assert p.returncode == 0, p.stderr
    d = json.loads(p.stdout)["data"]
    assert d["probe"] == "hall"
    assert d["points"][0]["mobility"] is not None

def test_cli_analyze_probe_override():
    p = _run(["analyze", SYNTH, "--probe", "hall", "--hall-channel", "1", "--thickness", "0.1"])
    assert p.returncode == 0, p.stderr
    assert json.loads(p.stdout)["data"]["probe"] == "hall"

def test_cli_hall_missing_channel_gates():
    # Repinned 2026-09-05 (was exit 2 / status error): missing --hall-channel now follows
    # the gate discipline every other missing input uses — exit 10 with a remedy an agent
    # can act on, instead of a hard error with an empty gate[].
    p = _run(["hall", SYNTH])                          # no --hall-channel
    assert p.returncode == 10
    env = json.loads(p.stdout)
    assert env["status"] == "gated"
    g = next(g for g in env["gate"] if g["need"] == "hall_channel")
    assert g["remedy"]["flag"] == "--hall-channel"


def test_cli_hall_tdep_missing_channel_gates():
    p = _run(["hall-tdep", SYNTH])                     # no --hall-channel
    assert p.returncode == 10
    env = json.loads(p.stdout)
    assert env["status"] == "gated"
    assert any(g["need"] == "hall_channel" for g in env["gate"])

def test_cli_hall_thickness_unit_um():
    p = _run(["hall", SYNTH, "--hall-channel", "1", "--thickness", "100", "--thickness-unit", "um"])
    # 100 um == 0.1 mm -> same R_H as the 0.1 mm case
    assert p.returncode == 0, p.stderr
    rh = json.loads(p.stdout)["data"]["points"][0]["R_H"]
    assert abs(rh - (-5.0e-8)) / 5.0e-8 < 0.01
