import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from cryosweep_gui.inputs.base import build_panel
import cryosweep_gui.inputs   # noqa: registers panels


def test_acms_panel_registered_and_round_trips(qapp):
    p = build_panel("acms")
    assert p.probe_key == "acms"
    p.set_state({"molar_mass": "200.0", "mass_mg": "5.0"})
    assert p.build_header_patch() == {"molar_mass": 200.0, "mass_mg": 5.0}
    assert p.get_state()["molar_mass"] == "200.0"


def test_acms_label_present():
    from cryosweep_gui.main_window import _LABELS
    assert _LABELS["acms"] == "AC Susceptibility"
