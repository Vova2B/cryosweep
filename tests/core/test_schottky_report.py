import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.reports import build_report

FIX = pathlib.Path(__file__).parent / "fixtures"


def _res(**hc):
    cfg = RunConfig.model_validate({"heatcapacity": {**hc}})
    return analyze_file(load_dat(str(FIX / "hc_schottky_synth.dat")), cfg, build_default_registry())


def test_report_has_no_schottky_section_when_off():
    md = build_report(_res())["markdown"]
    assert "## Schottky (opt-in)" not in md


def test_report_has_schottky_section_when_on():
    md = build_report(_res(schottky_enabled=True))["markdown"]
    assert "## Schottky (opt-in)" in md
    assert "Δ" in md
