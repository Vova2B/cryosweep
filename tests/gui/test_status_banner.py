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


def _banner_text(qapp, probe, parts, conf):
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(Result(status="low_confidence", confidence=conf, confidence_parts=parts,
                         data={"probe": probe}, provenance=_prov()))
    return b.text()


def test_the_two_ceilings_are_separately_labelled_not_stacked_parentheticals(qapp):
    """Review Minor: the clauses used to read `resolved fraction = X (R_H distinguishable
    from zero) (no fit-quality evidence survived ...)`. Two bare parentheticals back to back
    read as though BOTH qualify the resolved fraction, when the second is actually explaining
    why the OTHER ceiling is absent. Each clause now carries its own label and they are
    separated by a semicolon, so no clause can be read as qualifying the one before it.
    This pins the STRUCTURE, not the prose."""
    t = _banner_text(qapp, "hall_tdep", {"fit": None, "resolved": 0.478}, 0.478)
    ceiling = [seg for seg in t.split("|") if "binding ceiling" in seg][0]
    head, semi, tail = ceiling.partition(";")
    assert semi, "the two ceilings are separated by a semicolon, not nested in parens"
    assert "resolved fraction" in head and "fit quality" in tail
    assert ceiling.count("(") == 1, "exactly one parenthetical -- the gloss on the binding term"


def test_a_tie_between_the_two_ceilings_names_both(qapp):
    """The equal-ceilings branch existed but no test reached it (review Minor). A tie is not
    a rounding curiosity: min() picks one arbitrarily, so naming a single winner would imply
    a distinction the numbers do not support."""
    t = _banner_text(qapp, "hall", {"fit": 0.4, "resolved": 0.4}, 0.4)
    ceiling = [seg for seg in t.split("|") if "binding ceiling" in seg][0]
    assert "fit quality" in ceiling and "resolved fraction" in ceiling
    assert "--" not in ceiling, "user-facing text uses an em dash, not ASCII hyphens"
