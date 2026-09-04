from cryosweep_core.schema import get_schema, SCHEMA_NAMES

def test_resistivity_schema_registered():
    assert "analyze:resistivity" in SCHEMA_NAMES
    sch = get_schema("analyze:resistivity")
    assert sch["properties"]["probe"]["default"] == "resistivity"
    assert "bridges" in sch["properties"]
    assert "capabilities" in sch["properties"]

def test_resistivity_schema_round_trips_real_payload(res_path):
    # jsonschema is NOT installed in .venv; validate by reconstructing the typed
    # model from the analyzer's JSON payload (same approach as the existing
    # schema tests, which use pydantic models rather than the jsonschema lib).
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer, ResistivityData
    rt = load_dat(res_path)
    res = ResistivityAnalyzer().analyze(rt, RunConfig())
    rd = ResistivityData(**res.data)            # raises ValidationError if shape drifts
    assert rd.probe == "resistivity"
    assert sorted(b.channel for b in rd.bridges) == [1]  # Ch2 routed out as Hall-wired
    assert rd.excluded_hall_channel == 2 and rd.excluded_hall_source == "detected"
    assert {c.name for c in rd.capabilities} >= {"RRR", "magnetoresistance", "curve_separation"}
