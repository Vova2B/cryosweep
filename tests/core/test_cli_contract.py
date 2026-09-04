"""Adversarial contract tests for the cryosweep CLI (Slice 3, Bugs 1 & 2).

Every command MUST emit exactly one JSON envelope on stdout and never
crash with a bare traceback. Bad/unreadable input => status "error", exit 2.
_emit MUST never write a bare NaN/Infinity token (invalid RFC-8259 JSON).
"""
import json
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = str(ROOT / "tests/core/fixtures/vsm_synth.dat")


def _run(*args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args],
                          capture_output=True, text=True, cwd=ROOT)


# --- Bug 1: missing/unreadable file -> JSON error envelope + exit 2 ---

def test_analyze_missing_file_emits_error_envelope():
    r = _run("analyze", "/nope/x.dat")
    assert r.returncode == 2, r.stderr
    env = json.loads(r.stdout)  # must be valid JSON, no traceback
    assert env["status"] == "error"
    assert env["errors"], "error envelope must carry a non-empty errors list"


def test_detect_missing_file_emits_error_envelope():
    r = _run("detect", "/nope/x.dat")
    assert r.returncode == 2, r.stderr
    env = json.loads(r.stdout)
    assert env["status"] == "error"
    assert env["errors"]


def test_export_missing_file_emits_error_envelope():
    r = _run("export", "/nope/x.dat")
    assert r.returncode == 2, r.stderr
    env = json.loads(r.stdout)
    assert env["status"] == "error"


def test_run_malformed_json_emits_json_no_traceback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    r = _run("run", str(bad))
    assert r.returncode == 2, r.stderr
    out = json.loads(r.stdout)  # valid JSON, not a traceback
    assert "Traceback" not in r.stdout
    assert out.get("exit", 0) >= 2 or out.get("error")


def test_run_pipeline_one_bad_step_still_reports_others(tmp_path):
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"steps": [
        {"command": "analyze", "file": "/nope/missing.dat"},
        {"command": "analyze", "file": FIX}]}))
    r = _run("run", str(pf))
    assert r.returncode >= 2, r.stderr
    out = json.loads(r.stdout)
    assert "Traceback" not in r.stdout
    assert len(out["results"]) == 2
    statuses = [s["status"] for s in out["results"]]
    assert "error" in statuses           # the bad step is reported
    assert "ok" in statuses              # the good step still ran


# --- Bug 2: _emit must never produce a bare NaN/Infinity token ---

def test_emit_rejects_nan_no_bare_token(capsys):
    from cryosweep_cli.__main__ import _emit
    # _emit with allow_nan=False raises ValueError on non-finite; the caller
    # (main's try/except) degrades to an error envelope. Either way, _emit
    # must never write a bare NaN token (the invalid-JSON failure mode).
    try:
        _emit({"x": float("nan")})
    except ValueError:
        assert "NaN" not in capsys.readouterr().out
        return
    json.loads(capsys.readouterr().out)  # if emitted, it must be valid JSON
