import json
import pathlib

import pytest

from cryosweep_core.analyzers.mag import VSMData, VSMAnalyzer
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.schema import get_schema, SCHEMA_NAMES

FIX = pathlib.Path("tests/core/fixtures/vsm_synth.dat")


def test_schema_names_resolve():
    for name in SCHEMA_NAMES:
        s = get_schema(name)
        assert s["type"] == "object" or "properties" in s
        # deterministic + JSON-serializable
        assert json.dumps(s, sort_keys=True) == json.dumps(get_schema(name), sort_keys=True)


def test_schema_analyze_vsm_is_vsmdata():
    s = get_schema("analyze:vsm")
    assert "inv_chi" in s["properties"]


def test_unknown_schema_raises():
    with pytest.raises(KeyError):
        get_schema("nonsense")


def test_vsmdata_schema_and_population():
    props = VSMData.model_json_schema()["properties"]
    for f in ("temperature", "field", "chi_molar_cgs", "inv_chi", "fit"):
        assert f in props
    res = VSMAnalyzer().analyze(load_dat(FIX), RunConfig.load())
    # data still round-trips through VSMData
    vd = VSMData(**res.data)
    assert len(vd.temperature) == len(vd.inv_chi)
    assert vd.probe == "vsm"
