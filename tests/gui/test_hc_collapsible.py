from cryosweep_gui.inputs.hc import HCInputPanel
from cryosweep_gui.widgets import CollapsibleGroup


def test_advanced_groups_collapsed_by_default(qapp):
    p = HCInputPanel()
    for key in ("full", "schottky", "transition"):
        assert isinstance(p._adv_groups[key], CollapsibleGroup)
        assert p._adv_groups[key].is_collapsed() is True


def test_get_set_state_survives_collapse(qapp):
    p = HCInputPanel()
    p.schottky_enable.setChecked(True)
    p.transition_enable.setChecked(True)
    state = p.get_state()
    p._adv_groups["schottky"].set_collapsed(False)     # expanding must not disturb params
    q = HCInputPanel()
    q.set_state(state)
    assert q.schottky_enable.isChecked() is True
    assert q.transition_enable.isChecked() is True
