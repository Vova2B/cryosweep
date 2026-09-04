from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
from cryosweep_core.reports import build_report

def test_report_lists_data_quality_diagnostics(res_path):
    r = ResistivityAnalyzer().analyze(load_dat(res_path), RunConfig())
    rep = build_report(r)
    assert "## Data quality" in rep["markdown"]
    assert "outlier" in rep["markdown"].lower()
    assert "diagnostics" in rep["json"] and len(rep["json"]["diagnostics"]) >= 1

def test_report_omits_section_when_no_diagnostics():
    from cryosweep_core.result import Result, Provenance
    r = Result(status="ok", data={"probe": "resistivity", "bridges": [], "capabilities": []},
               provenance=Provenance(file="x", sha256="ab", app_version=None))
    rep = build_report(r)
    assert "## Data quality" not in rep["markdown"]
