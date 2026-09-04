# tests/gui/test_hall_tdep_panel.py
from cryosweep_gui.inputs.base import build_panel
from cryosweep_gui.inputs.hall import HallInputPanel


def test_hall_tdep_gets_hall_panel(qapp):
    panel = build_panel("hall_tdep")
    assert isinstance(panel, HallInputPanel)


def test_set_hall_channel_then_overrides(qapp):
    panel = build_panel("hall_tdep")
    panel.set_hall_channel(1)
    assert panel.hall_channel_edit.text() == "1"
    assert panel.build_overrides()["hall"]["hall_channel"] == 1
