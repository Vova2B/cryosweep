import json, pathlib, subprocess, sys
from cryosweep_core.plotting.spec import GlobalStyle, PlotLayout, PlotEntry, PlotSpec
FIX = pathlib.Path(__file__).parent / "fixtures"
ROOT = pathlib.Path(__file__).resolve().parents[2]

def _run(args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args], capture_output=True, text=True, cwd=ROOT)

def test_style_file_renders_ok(tmp_path):
    sf = tmp_path / "style.json"; sf.write_text(GlobalStyle(marker="s", colormap="viridis").model_dump_json())
    out = tmp_path / "p"
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--style-file", str(sf), "--out", str(out)])
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "p.png").exists()

def test_layout_file_spec_applied(tmp_path):
    lay = PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t", spec=PlotSpec(yscale="linear"))])
    lf = tmp_path / "lay.json"; lf.write_text(lay.model_dump_json())
    out = tmp_path / "q"
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--plot-kind", "resistivity_rho_t", "--layout-file", str(lf), "--out", str(out)])
    assert r.returncode == 0 and (tmp_path / "q.png").exists()

def test_wrong_probe_layout_file_warns(tmp_path):
    lay = PlotLayout(plots=[PlotEntry(kind="inverse_chi")])   # vsm layout vs resistivity file
    lf = tmp_path / "v.json"; lf.write_text(lay.model_dump_json())
    out = tmp_path / "w"
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--layout-file", str(lf), "--out", str(out)])
    env = json.loads(r.stdout)
    assert any("no plots for probe" in w for w in env.get("warnings", []))

def test_bad_style_file_path_error_envelope(tmp_path):
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--style-file", str(tmp_path / "nope.json"), "--out", str(tmp_path / "x")])
    env = json.loads(r.stdout)
    assert env["status"] == "error" and r.returncode == 2
