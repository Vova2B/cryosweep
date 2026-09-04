import json, pathlib

import pytest

from cryosweep_core.pipeline import run_pipeline, PipelineCfg
from cryosweep_core.config import RunConfig
FIX = str(pathlib.Path("tests/core/fixtures/vsm_synth.dat").resolve())


def test_pipeline_runs_steps(tmp_path):
    pf = tmp_path / "p.json"
    pf.write_text(json.dumps({"steps": [
        {"command": "detect", "file": FIX},
        {"command": "analyze", "file": FIX}]}))
    out = run_pipeline(str(pf), RunConfig.load())
    assert len(out["results"]) == 2
    assert out["results"][1]["status"] == "ok"
    assert out["exit"] == 0


def test_pipeline_validates():
    with pytest.raises(Exception):
        PipelineCfg(steps=[{"command": "bogus", "file": "x"}])
