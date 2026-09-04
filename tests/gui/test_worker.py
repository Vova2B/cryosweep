import dataclasses
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

def _vsm_inputs(vsm_path):
    rt = load_dat(str(vsm_path))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return rt, RunConfig.load(unit_system="CGS", probe_override="vsm")

def test_run_analysis_matches_direct_pipeline(qapp, vsm_path):
    from cryosweep_gui.worker import run_analysis
    rt, cfg = _vsm_inputs(vsm_path)
    reg = build_default_registry()
    assert run_analysis(rt, cfg, reg).model_dump_json() == analyze_file(rt, cfg, reg).model_dump_json()

def test_run_analysis_never_raises(qapp, vsm_path):
    from cryosweep_gui.worker import run_analysis
    rt, cfg = _vsm_inputs(vsm_path)
    res = run_analysis(rt, cfg, build_default_registry())
    assert res.status in ("ok", "gated", "low_confidence", "error")

def test_analyze_worker_emits_done_with_result(qapp, vsm_path):
    from cryosweep_gui.worker import AnalyzeWorker
    rt, cfg = _vsm_inputs(vsm_path)
    reg = build_default_registry()
    got = []
    w = AnalyzeWorker(rt, cfg, reg)
    w.done.connect(lambda r: got.append(r))     # cross-thread queued connection
    w.start(); w.wait(5000); qapp.processEvents()
    assert w.isFinished() and not w.isRunning()
    assert len(got) == 1 and got[0].status == "ok"
    assert got[0].model_dump_json() == analyze_file(rt, cfg, reg).model_dump_json()
