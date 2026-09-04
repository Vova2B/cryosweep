import csv, pathlib
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer, HallData
from cryosweep_core.io.export import export_result
from cryosweep_core.reports import build_report
from cryosweep_core.plotting.render import render_for
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.schema import get_schema, SCHEMA_NAMES

def _res(p):
    return HallAnalyzer().analyze(load_dat(p), RunConfig.load(hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))

def test_schema_hall_registered():
    assert "analyze:hall" in SCHEMA_NAMES
    sch = get_schema("analyze:hall")
    assert "points" in sch["properties"] and "capabilities" in sch["properties"]

def test_schema_round_trips_real_payload(hall_synth_path):
    res = _res(hall_synth_path)
    HallData(**res.data)                              # raises if shape drifts

def test_export_hall_writes_points_derived_caps(tmp_path, hall_synth_path):
    res = _res(hall_synth_path)
    out = export_result(res, str(tmp_path / "hall"), fmt="csv")
    assert "points" in out and "capabilities" in out
    with open(out["points"]) as f:
        rows = list(csv.DictReader(f))
    assert rows and "R_H (m^3/C)" in rows[0]
    assert {"temperature (K)"} <= set(rows[0])

def test_report_hall_renders_sections(hall_synth_path):
    res = _res(hall_synth_path)
    md = build_report(res)["markdown"]
    assert "## Hall" in md and "R_H" in md and "## Capabilities" in md

def test_render_hall_returns_figure(hall_synth_path):
    res = _res(hall_synth_path)
    fig = render_for(res, PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Temperature (K)"

def test_discovery_lists_hall_without_detector():
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.discovery import discover
    d = discover(build_default_registry())
    keys = {p["key"] for p in d["probes"]}
    assert "hall" in keys                             # listed even though it has no detector
