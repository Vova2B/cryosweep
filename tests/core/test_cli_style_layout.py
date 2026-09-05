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


# ---- unknown keys in --layout-file / --style-file must warn, never pass silently ----
# (2026-09-05: both traps below previously gave exit 0, no warning, and a figure without
# the requested feature — an agent then reports a figure property that is not there.)

def test_layout_file_unknown_key_at_entry_level_warns(tmp_path):
    # the plausible wrong shape: the spec boolean placed on the ENTRY, not under "spec"
    lf = tmp_path / "lay.json"
    lf.write_text(json.dumps({"plots": [{"kind": "resistivity_rho_t", "error_band": True}]}))
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--plot-kind", "resistivity_rho_t", "--layout-file", str(lf),
              "--out", str(tmp_path / "p")])
    assert r.returncode == 0, r.stderr                      # warn, not error (presets from
    env = json.loads(r.stdout)                              # other versions must keep loading)
    assert any("--layout-file" in w and "plots[0].error_band" in w for w in env["warnings"])


def test_layout_file_wrong_top_level_shape_warns(tmp_path):
    lf = tmp_path / "lay.json"
    lf.write_text(json.dumps({"resistivity_rho_t": {"error_band": True}}))
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--layout-file", str(lf), "--out", str(tmp_path / "p")])
    assert r.returncode == 0, r.stderr
    env = json.loads(r.stdout)
    assert any("--layout-file" in w and "resistivity_rho_t" in w for w in env["warnings"])


def test_style_file_unknown_key_warns(tmp_path):
    sf = tmp_path / "style.json"
    sf.write_text(json.dumps({"errorband": True, "grid": True}))   # typo'd key + one valid
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--style-file", str(sf), "--out", str(tmp_path / "p")])
    assert r.returncode == 0, r.stderr
    env = json.loads(r.stdout)
    assert any("--style-file" in w and "errorband" in w for w in env["warnings"])


def test_valid_layout_and_style_files_do_not_warn(tmp_path):
    sf = tmp_path / "style.json"; sf.write_text(GlobalStyle(marker="s").model_dump_json())
    lay = PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t", spec=PlotSpec(yscale="linear"))])
    lf = tmp_path / "lay.json"; lf.write_text(lay.model_dump_json())
    r = _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
              "--plot-kind", "resistivity_rho_t", "--layout-file", str(lf),
              "--style-file", str(sf), "--out", str(tmp_path / "p")])
    assert r.returncode == 0, r.stderr
    env = json.loads(r.stdout)
    assert not any("unknown key" in w for w in env["warnings"])


def test_unknown_keys_helper_recurses():
    from cryosweep_core.schema import unknown_keys
    from cryosweep_core.plotting.spec import GlobalStyle as GS, PlotLayout as PL
    assert unknown_keys(GS, {"grid": True}) == []
    assert unknown_keys(GS, {"errorband": True}) == ["errorband"]
    got = unknown_keys(PL, {"plots": [{"kind": "x", "error_band": True,
                                       "spec": {"error_band": True, "shady": 1}}],
                            "bogus": 0})
    assert set(got) == {"plots[0].error_band", "plots[0].spec.shady", "bogus"}
    assert unknown_keys(GS, "not a dict") == []
