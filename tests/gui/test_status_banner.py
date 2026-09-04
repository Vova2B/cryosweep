import math
from cryosweep_core.result import Result, Gate, Provenance

def _prov():
    return Provenance(file="x", sha256="", app_version="", config={})

def test_banner_shows_status_and_confidence(qapp):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="ok", confidence=0.97, data={"probe": "vsm"}, provenance=_prov()))
    assert "ok" in b.text().lower()
    assert "0.97" in b.text()

def test_banner_coerces_nan_confidence(qapp):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="low_confidence", confidence=float("nan"),
                         data={"probe": "x"}, provenance=_prov()))
    assert "—" in b.text() and "nan" not in b.text().lower()

def test_banner_lists_gate_remedy_dict(qapp):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    g = Gate(need="molar_mass", reason="no MOLWGHT", remedy={"flag": "--molar-mass", "example": "5.0"})
    b.show_result(Result(status="gated", confidence=0.5, data={"probe": "vsm"}, gate=[g], provenance=_prov()))
    t = b.text()
    assert "molar_mass" in t and "--molar-mass" in t      # remedy dict formatted, not "{...}"

def test_banner_shows_errors(qapp):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="error", errors=["boom happened"], data={"probe": "x"}, provenance=_prov()))
    assert "boom happened" in b.text()
