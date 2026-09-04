"""PQ-1 2b CLI: cryosweep plot --format/--all/--tight/--dpi (spec §3)."""
import json, pathlib, subprocess, sys

from PIL import Image

FIX = pathlib.Path(__file__).parent / "fixtures"
ROOT = pathlib.Path(__file__).resolve().parents[2]

def _run(args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args],
                          capture_output=True, text=True, cwd=ROOT)

def _plot(extra, out):
    return _run(["plot", str(FIX / "act_synth.dat"), "--probe", "resistivity",
                 "--out", str(out), *extra])


def test_format_multi_writes_one_file_per_format(tmp_path):
    out = tmp_path / "p"
    r = _plot(["--format", "png,pdf,svg", "--plot-kind", "resistivity_rho_t"], out)
    assert r.returncode == 0, r.stderr
    for ext in ("png", "pdf", "svg"):
        assert (tmp_path / f"p.{ext}").exists(), ext
    env = json.loads(r.stdout)
    assert len(env["data"]["plots"]) == 3


def test_default_format_is_png_and_dpi_now_honored(tmp_path):
    out = tmp_path / "p"
    r = _plot(["--plot-kind", "resistivity_rho_t"], out)     # default style: 90x70mm @300
    assert r.returncode == 0, r.stderr
    with Image.open(tmp_path / "p.png") as im:
        assert im.size == (round(90 / 25.4 * 300), round(70 / 25.4 * 300))


def test_dpi_flag_overrides(tmp_path):
    out = tmp_path / "p"
    r = _plot(["--plot-kind", "resistivity_rho_t", "--dpi", "150"], out)
    assert r.returncode == 0, r.stderr
    with Image.open(tmp_path / "p.png") as im:
        assert im.size == (round(90 / 25.4 * 150), round(70 / 25.4 * 150))


def test_all_exports_every_kind_with_prefix_naming(tmp_path):
    out = tmp_path / "batch" / "sample"
    r = _plot(["--all"], out)
    assert r.returncode == 0, r.stderr
    env = json.loads(r.stdout)
    names = sorted(pathlib.Path(p).name for p in env["data"]["plots"])
    assert "sample_resistivity_rho_t.png" in names
    assert len(names) >= 2                       # act_synth backs multiple kinds
    for p in env["data"]["plots"]:
        assert pathlib.Path(p).exists()


def test_all_conflicts_with_plot_kind(tmp_path):
    r = _plot(["--all", "--plot-kind", "resistivity_rho_t"], tmp_path / "x")
    assert r.returncode != 0
    assert "plot-kind" in r.stderr


def test_tight_runs_and_changes_png_dims(tmp_path):
    r1 = _plot(["--plot-kind", "resistivity_rho_t"], tmp_path / "a")
    r2 = _plot(["--plot-kind", "resistivity_rho_t", "--tight"], tmp_path / "b")
    assert r1.returncode == 0 and r2.returncode == 0, r2.stderr
    with Image.open(tmp_path / "a.png") as a, Image.open(tmp_path / "b.png") as b:
        assert a.size != b.size
