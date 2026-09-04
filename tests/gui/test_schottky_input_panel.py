import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from cryosweep_gui.inputs.hc import HCInputPanel
from cryosweep_core.config import RunConfig

_app = QApplication.instance() or QApplication([])


def test_defaults_neutral():
    """Fresh panel: schottky_enabled is False, schottky_delta_h_model is 'none'."""
    p = HCInputPanel()
    ov = p.build_overrides()["heatcapacity"]
    assert ov["schottky_enabled"] is False
    assert ov["schottky_delta_h_model"] == "none"


def test_toggle_all_five_knobs():
    """Check enable, r=2.0, t5, nuclear, combo=zeeman → build_overrides reflects all 5."""
    p = HCInputPanel()
    p.schottky_enable.setChecked(True)
    p.schottky_r.setValue(2.0)
    p.schottky_t5.setChecked(True)
    p.schottky_nuclear.setChecked(True)
    idx = p.schottky_dh_model.findText("zeeman")
    p.schottky_dh_model.setCurrentIndex(idx)

    ov = p.build_overrides()["heatcapacity"]
    assert ov["schottky_enabled"] is True
    assert ov["schottky_r"] == 2.0
    assert ov["schottky_lattice_t5"] is True
    assert ov["schottky_include_nuclear"] is True
    assert ov["schottky_delta_h_model"] == "zeeman"


def test_state_roundtrip():
    """After toggling, set_state(get_state()) on a fresh panel reproduces same build_overrides."""
    p = HCInputPanel()
    p.schottky_enable.setChecked(True)
    p.schottky_r.setValue(3.5)
    p.schottky_t5.setChecked(True)
    p.schottky_nuclear.setChecked(True)
    idx = p.schottky_dh_model.findText("zfs")
    p.schottky_dh_model.setCurrentIndex(idx)

    st = p.get_state()
    q = HCInputPanel()
    q.set_state(st)
    ov = q.build_overrides()["heatcapacity"]
    assert ov["schottky_enabled"] is True
    assert ov["schottky_r"] == 3.5
    assert ov["schottky_lattice_t5"] is True
    assert ov["schottky_include_nuclear"] is True
    assert ov["schottky_delta_h_model"] == "zfs"


def test_run_config_validates():
    """build_overrides() passes RunConfig.model_validate and schottky_enabled matches."""
    p = HCInputPanel()
    p.schottky_enable.setChecked(True)
    ov = p.build_overrides()
    rc = RunConfig.model_validate(ov)
    assert rc.heatcapacity.schottky_enabled is True
    assert rc.heatcapacity.schottky_delta_h_model == "none"
