# tests/core/test_schottky_export.py
import pathlib, csv
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.io.export import export_result

FIX = pathlib.Path(__file__).parent / "fixtures"
def _res(**hc):
    cfg = RunConfig.model_validate({"heatcapacity": {**hc}})
    return analyze_file(load_dat(str(FIX / "hc_schottky_synth.dat")), cfg, build_default_registry())

def test_no_schottky_csv_when_off(tmp_path):
    out = export_result(_res(), tmp_path / "off")
    assert "schottky" not in out
    assert not list(tmp_path.glob("*.schottky.csv"))

def test_schottky_csv_written_when_on(tmp_path):
    out = export_result(_res(schottky_enabled=True), tmp_path / "on")
    p = pathlib.Path(out["schottky"]); assert p.exists()
    rows = list(csv.reader(p.open()))
    assert rows[0][:4] == ["field_oe", "chosen_model", "param", "value"]
    assert any(r[2] == "Delta" for r in rows[1:])

def test_field_dependence_csv_unchanged_by_schottky(tmp_path):
    a = export_result(_res(), tmp_path / "a")
    b = export_result(_res(schottky_enabled=True), tmp_path / "b")
    assert pathlib.Path(a["field_dependence"]).read_text() == \
           pathlib.Path(b["field_dependence"]).read_text()
