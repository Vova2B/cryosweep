import json
from cryosweep_core.result import Result, FitResult, Gate, Provenance

def test_result_roundtrip_and_status_enum():
    r = Result(status="gated", confidence=0.4,
               confidence_parts={"detector": 0.9, "segmentation": 0.4, "fit": None},
               data={"probe": "vsm"},
               gate=[Gate(need="molar_mass", reason="no MOLWGHT",
                          remedy={"flag": "--molar-mass", "example": "--molar-mass 945.68"})],
               provenance=Provenance(file="x.dat", sha256="ab", app_version="VSM", config={}))
    s = r.model_dump_json()
    assert json.loads(s)["status"] == "gated"

def test_status_rejects_unknown():
    import pytest, pydantic
    with pytest.raises(pydantic.ValidationError):
        Result(status="totally-bogus", confidence=1.0, data={},
               provenance=Provenance(file="x", sha256="y", app_version="z", config={}))

def test_fitresult_carries_covariance():
    fr = FitResult(model="curie_weiss", params={"C": 1.0, "theta": -2.0},
                   sigma={"C": 0.1, "theta": 0.2}, covariance=[[1e-2, 0.0], [0.0, 4e-2]],
                   r2=0.999, chi2_red=1.1, n_points=42, fit_range=[2.0, 300.0],
                   units={"C": "emu*K/mol"}, quality_flags=[])
    assert fr.n_points == 42
    assert "model" in FitResult.model_json_schema()["properties"]

def test_schema_sorts_for_determinism():
    a = json.dumps(Result.model_json_schema(), sort_keys=True)
    b = json.dumps(Result.model_json_schema(), sort_keys=True)
    assert a == b


def test_result_diagnostics_additive_and_roundtrips():
    from cryosweep_core.result import Result, Diagnostic, Provenance
    prov = Provenance(file="x", sha256="ab", app_version=None)
    r0 = Result(status="ok", provenance=prov)
    assert r0.diagnostics == []
    d = Diagnostic(kind="outliers", severity="warning", scope="bridge1 ρ(T) 0.5 Oe",
                   message="7 outlier points (1.2%)", data={"n_outliers": 7})
    r = Result(status="ok", provenance=prov, diagnostics=[d])
    j = r.model_dump_json()
    r2 = Result.model_validate_json(j)
    assert r2.diagnostics[0].kind == "outliers" and r2.diagnostics[0].data["n_outliers"] == 7
