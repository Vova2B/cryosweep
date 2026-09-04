"""CLI file/path handling: ~ expansion, --out parent creation, non-QD-file messages.

The `~` cases drive expansion at the two user-input boundaries (argv in
cryosweep_cli.__main__, pipeline step files in cryosweep_core.pipeline) via a
subprocess with HOME pointed at tmp_path — the shell never sees the path, which
is exactly the unmasked (pipeline/JSON) failure mode.
"""
import json, os, pathlib, shutil, subprocess, sys

FIX = pathlib.Path(__file__).parent / "fixtures" / "vsm_synth.dat"
ROOT = pathlib.Path(__file__).resolve().parents[2]


def _run_home(home, *args):
    env = {**os.environ, "HOME": str(home)}
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args],
                          capture_output=True, text=True, cwd=ROOT, env=env)


def _fixture_under(home):
    (home / "data").mkdir(exist_ok=True)
    shutil.copy(FIX, home / "data" / "vsm_synth.dat")


# --- Defect 1: a leading ~ is never expanded ---
# Before the fix: exit 2, errors[0] = "[Errno 2] No such file or directory: '~/data/...'"

def test_tilde_expands_for_positional_file(tmp_path):
    _fixture_under(tmp_path)
    r = _run_home(tmp_path, "analyze", "~/data/vsm_synth.dat")
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["status"] == "ok"


def test_tilde_expands_for_out(tmp_path):
    _fixture_under(tmp_path)
    r = _run_home(tmp_path, "export", str(FIX), "--out", "~/outdir/out")
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "outdir" / "out.points.csv").exists()


def test_tilde_expands_for_style_file(tmp_path):
    (tmp_path / "style.json").write_text("{}")
    r = _run_home(tmp_path, "plot", str(FIX), "--style-file", "~/style.json",
                  "--out", str(tmp_path / "fig"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "fig.png").exists()


def test_tilde_expands_in_pipeline_step_file(tmp_path):
    # The real damage: a pipeline JSON path is a plain string, no shell involved.
    _fixture_under(tmp_path)
    pipe = tmp_path / "pipe.json"
    pipe.write_text(json.dumps(
        {"steps": [{"command": "analyze", "file": "~/data/vsm_synth.dat"}]}))
    r = _run_home(tmp_path, "run", str(pipe))
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["results"][0]["status"] == "ok", out
    # the step echo keeps the path exactly as the user wrote it
    assert out["results"][0].get("file", "~/data/vsm_synth.dat") == "~/data/vsm_synth.dat"


# --- Defect 2: --out into a directory that does not exist ---
# Before the fix: exit 2, "[Errno 2] No such file or directory: '/.../no_such_dir/x...'"

def test_export_out_creates_missing_parent_dir(tmp_path):
    r = _run_home(tmp_path, "export", str(FIX), "--out", str(tmp_path / "no_such_dir" / "x"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "no_such_dir" / "x.points.csv").exists()


def test_plot_out_creates_missing_parent_dir(tmp_path):
    r = _run_home(tmp_path, "plot", str(FIX), "--out", str(tmp_path / "missing2" / "fig"))
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "missing2" / "fig.png").exists()


# --- Defect 3: raw pandas internals leak for non-QD files ---
# Before the fix: errors[0] = "Error tokenizing data. C error: Expected 1 fields
# in line 3, saw 2" / "No columns to parse from file" — nothing actionable.

def test_non_qd_file_says_so(tmp_path):
    bad = tmp_path / "notes.md"
    bad.write_text("alpha\nbeta\ngamma, delta\n")   # ragged: ParserError in pandas
    r = _run_home(tmp_path, "analyze", str(bad))
    assert r.returncode == 2
    env = json.loads(r.stdout)
    assert env["status"] == "error"
    assert "Quantum Design" in env["errors"][0]
    # underlying pandas detail preserved in the envelope for debugging
    assert "tokenizing" in env["errors"][0]


def test_empty_file_says_empty(tmp_path):
    empty = tmp_path / "empty.dat"
    empty.write_bytes(b"")
    r = _run_home(tmp_path, "analyze", str(empty))
    assert r.returncode == 2
    env = json.loads(r.stdout)
    assert env["status"] == "error"
    assert "empty" in env["errors"][0]
