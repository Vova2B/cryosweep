"""A pipeline's exit code must report its WORST step, and the codes are not a severity scale.

`EXIT_CODES` is {ok: 0, gated: 10, low_confidence: 11, error: 2} -- chosen so the soft,
recoverable outcomes sit in a high band away from the shell's conventional hard-failure 2.
That makes `max()` over the raw codes wrong: it ranks a step that HARD-FAILED (2) below one
that merely needs a flag (10). The consumer this matters to is the agent the shipped skill
targets: told "gated", it re-runs with --molar-mass and never learns a file was missing.

Severity order is error > gated > low_confidence > ok.
"""
import json
import pathlib
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.pipeline import run_pipeline

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


def _pipeline(tmp_path, *steps):
    p = tmp_path / "p.json"
    p.write_text(json.dumps({"steps": [{"command": "analyze", "file": str(f)} for f in steps]}))
    return run_pipeline(str(p), RunConfig())


def test_a_hard_error_outranks_a_gated_step_whatever_the_order(tmp_path):
    """The regression: max() over raw codes returned 10 (gated) and hid the missing file."""
    missing = EXAMPLES / "does_not_exist.dat"
    gated = EXAMPLES / "magnetization_mpms.dat"
    for steps in ((missing, gated), (gated, missing)):
        out = _pipeline(tmp_path, *steps)
        assert {r["status"] for r in out["results"]} == {"error", "gated"}
        assert out["exit"] == 2, f"error must win over gated, got {out['exit']} for {steps}"


def test_a_hard_error_outranks_low_confidence(tmp_path):
    out = _pipeline(tmp_path, EXAMPLES / "does_not_exist.dat",
                    EXAMPLES / "magnetization_vsm_multifield.dat")
    assert {r["status"] for r in out["results"]} == {"error", "low_confidence"}
    assert out["exit"] == 2


def test_soft_statuses_still_rank_above_ok_and_against_each_other(tmp_path):
    """Guard the other direction: fixing the inversion must not flatten the soft band."""
    out = _pipeline(tmp_path, EXAMPLES / "magnetization_vsm.dat",
                    EXAMPLES / "magnetization_mpms.dat")
    assert out["exit"] == 10, "gated must outrank ok"
    out = _pipeline(tmp_path, EXAMPLES / "magnetization_vsm.dat",
                    EXAMPLES / "magnetization_vsm_multifield.dat")
    assert out["exit"] == 11, "low_confidence must outrank ok"


def test_an_all_ok_pipeline_still_exits_zero(tmp_path):
    out = _pipeline(tmp_path, EXAMPLES / "magnetization_vsm.dat", EXAMPLES / "heat_capacity.dat")
    assert [r["status"] for r in out["results"]] == ["ok", "ok"]
    assert out["exit"] == 0


def test_a_malformed_pipeline_file_is_still_exit_2(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert run_pipeline(str(bad), RunConfig())["exit"] == 2
