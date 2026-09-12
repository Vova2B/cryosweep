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


# ---- confidence_parts named on screen for the Hall probes (spec §4.5) -----------------
# Task 6 made confidence_parts the whole diagnosis for a low_confidence Hall result, but
# `grep -rn "confidence_parts" cryosweep_gui/` returned nothing -- the GUI never showed
# which of the two ceilings (fit quality vs resolved fraction) actually bound the number.
# A file whose `fit` term binds and a file whose `resolved` term binds are opposite
# problems ("R_xy is not linear in B" vs "R_H is indistinguishable from zero") that used
# to render an identical banner.

def test_banner_names_resolved_fraction_as_the_binding_ceiling(qapp):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="low_confidence", confidence=0.4782608695652174,
                         confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                           "antisym_fraction": 1.0, "fit": None,
                                           "resolved": 0.4782608695652174},
                         data={"probe": "hall_tdep"}, provenance=_prov()))
    t = b.text()
    assert "resolved fraction" in t and "0.478" in t
    # `fit: None` is itself meaningful (no r2 survived the zero-DOF rule) and must read as
    # words, never as a bare dash standing in for "missing".
    assert "—" not in t.split("confidence:")[1].split("|")[0]  # the confidence value itself is real
    assert "zero-degrees-of-freedom" in t or "zero-DOF" in t or "no fit-quality evidence" in t

def test_banner_names_fit_quality_as_the_binding_ceiling(qapp):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="low_confidence", confidence=0.31,
                         confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                           "fit": 0.31, "resolved": 1.0},
                         data={"probe": "hall"}, provenance=_prov()))
    t = b.text()
    assert "fit quality" in t and "0.31" in t

def test_banner_confidence_note_is_confined_to_hall_probes(qapp):
    """A wider change to this shared banner would alter six probes' on-screen text, which
    is its own reviewable change -- this task only touches the Hall probes' wording."""
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="low_confidence", confidence=0.4,
                         confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                           "fit": 0.4},
                         data={"probe": "vsm"}, provenance=_prov()))
    t = b.text()
    assert "binding" not in t and "ceiling" not in t

def test_banner_confidence_note_only_fires_when_status_is_low_confidence(qapp):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="ok", confidence=1.0,
                         confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                           "fit": 1.0, "resolved": 1.0},
                         data={"probe": "hall"}, provenance=_prov()))
    t = b.text()
    assert "binding" not in t and "ceiling" not in t
