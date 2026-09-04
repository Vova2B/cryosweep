from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
from cryosweep_core.reports import build_report

def test_resistivity_report_includes_bridges_rrr_mr_caps(res_path):
    res = ResistivityAnalyzer().analyze(load_dat(res_path), RunConfig())
    rep = build_report(res)
    md = rep["markdown"]
    assert "## Bridges" in md
    assert "18.5" in md                      # ch1 RRR ~18.52 rendered
    assert "## Magnetoresistance" in md
    assert "## Capabilities" in md
    assert "power_law_fit" in md
    assert rep["json"]["probe"] == "resistivity"
    assert len(rep["json"]["bridges"]) == 1              # Ch2 routed out as Hall-wired
    assert "Hall-channel routing" in md and "Ch2" in md
