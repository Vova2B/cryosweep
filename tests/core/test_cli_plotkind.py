import json, pathlib, subprocess, sys
FIX = pathlib.Path(__file__).parent / "fixtures"
ROOT = pathlib.Path(__file__).resolve().parents[2]      # repo root (matches test_cli.py)

def _run(args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args],
                          capture_output=True, text=True, cwd=ROOT)

def test_plot_kind_renders_named_kind(tmp_path):
    out = tmp_path / "p"
    r = _run(["plot", str(FIX / "vsm_synth.dat"), "--probe", "vsm",
              "--molar-mass", "200", "--mass-mg", "5", "--plot-kind", "vsm_moment_t", "--out", str(out)])
    assert r.returncode == 0, r.stderr
    env = json.loads(r.stdout)
    assert env["data"]["plot"].endswith(".png")
    assert (tmp_path / "p.png").exists()

def test_unbacked_plot_kind_emits_envelope_no_png(tmp_path):
    out = tmp_path / "q"
    r = _run(["plot", str(FIX / "vsm_synth.dat"), "--probe", "vsm",
              "--molar-mass", "200", "--mass-mg", "5", "--plot-kind", "resistivity_mr", "--out", str(out)])
    env = json.loads(r.stdout)
    assert "plot" not in env["data"] or env["data"].get("plot") is None
    assert not (tmp_path / "q.png").exists()              # no PNG, no traceback
    assert r.returncode in (0, 2)
