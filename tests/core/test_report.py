import pathlib
from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.io.loader import load_dat
from cryosweep_core.reports import build_report
from cryosweep_core.config import RunConfig

FIX = pathlib.Path(__file__).parent / "fixtures" / "vsm_synth.dat"

def test_report_markdown_and_json():
    rt = load_dat(FIX); res = VSMAnalyzer().analyze(rt, RunConfig.load())
    rep = build_report(res)
    assert rep["json"]["probe"] == "vsm"
    md = rep["markdown"]
    assert "# CryoSweep Analysis Report" in md
    assert "Curie" in md and "mu_eff" in md
    assert "0.5" in md or "0.50" in md      # C recovered, appears in the table
