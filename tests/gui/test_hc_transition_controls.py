import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from cryosweep_gui.inputs.hc import HCInputPanel

_app = QApplication.instance() or QApplication([])


def test_transition_overrides_default_off():
    p = HCInputPanel()
    ov = p.build_overrides()
    assert ov["heatcapacity"]["transitions_enabled"] is False


def test_transition_overrides_when_enabled():
    p = HCInputPanel()
    p.transition_enable.setChecked(True)
    p.transition_form.setCurrentText("jump")
    p.transition_universality.setCurrentText("ising3d")
    p.transition_compare.setChecked(True)
    ov = p.build_overrides()["heatcapacity"]
    assert ov["transitions_enabled"] is True
    assert ov["transition_form"] == "jump"
    assert ov["transition_universality"] == "ising3d"
    assert ov["transition_compare_forms"] is True


def test_transition_state_round_trip():
    p = HCInputPanel()
    p.transition_enable.setChecked(True)
    p.transition_universality.setCurrentText("xy3d")
    st = p.get_state()
    q = HCInputPanel()
    q.set_state(st)
    assert q.transition_enable.isChecked() is True
    assert q.transition_universality.currentText() == "xy3d"
