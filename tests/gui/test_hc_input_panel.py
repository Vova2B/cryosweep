import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from cryosweep_gui.inputs.hc import HCInputPanel
_app = QApplication.instance() or QApplication([])

def test_full_range_overrides_and_state_roundtrip():
    p = HCInputPanel()
    p.set_fix("n", True); p.set_value("theta_D", 250.0)
    ov = p.build_overrides()["heatcapacity"]
    assert ov["full_fixed"]["n"] is True
    assert ov["full_init"]["theta_D"] == 250.0
    st = p.get_state()
    q = HCInputPanel(); q.set_state(st)
    assert q.build_overrides()["heatcapacity"]["full_init"]["theta_D"] == 250.0

def test_refit_signal_exists():
    p = HCInputPanel()
    assert hasattr(p, "refit_requested")


def test_entropy_overrides_present_when_reference_and_rln_set():
    p = HCInputPanel()
    p.entropy_source.setCurrentIndex(1)          # "Reference file…"
    p.entropy_ref_path.setText("/tmp/ref.dat")
    p.entropy_extrapolate.setChecked(False)
    p.entropy_rln_j.setValue(1.0)
    hc = p.build_overrides()["heatcapacity"]
    assert hc["entropy_extrapolate"] is False
    assert hc["entropy_lattice_ref_file"] == "/tmp/ref.dat"
    assert hc["entropy_rln_j"] == 1.0


def test_entropy_optional_keys_absent_when_default():
    p = HCInputPanel()
    # source = Fitted (index 0), path irrelevant, rln_j = 0 -> auto
    p.entropy_source.setCurrentIndex(0)
    p.entropy_ref_path.setText("")
    p.entropy_rln_j.setValue(0.0)
    hc = p.build_overrides()["heatcapacity"]
    assert "entropy_lattice_ref_file" not in hc
    assert "entropy_rln_j" not in hc
    assert hc["entropy_extrapolate"] is True      # always present, default True


def test_entropy_ref_path_absent_when_source_fitted_even_if_typed():
    p = HCInputPanel()
    p.entropy_source.setCurrentIndex(0)           # Fitted, so path ignored
    p.entropy_ref_path.setText("/tmp/ref.dat")
    hc = p.build_overrides()["heatcapacity"]
    assert "entropy_lattice_ref_file" not in hc


def test_entropy_state_roundtrip():
    p = HCInputPanel()
    p.entropy_source.setCurrentIndex(1)
    p.entropy_ref_path.setText("/tmp/ref.dat")
    p.entropy_extrapolate.setChecked(False)
    p.entropy_rln_j.setValue(2.5)
    q = HCInputPanel(); q.set_state(p.get_state())
    assert q.entropy_source.currentIndex() == 1
    assert q.entropy_ref_path.text() == "/tmp/ref.dat"
    assert q.entropy_extrapolate.isChecked() is False
    assert q.entropy_rln_j.value() == 2.5
    assert q.entropy_ref_path.isEnabled() is True  # enable state synced from source


def test_show_comparison_none_renders_na():
    """A None sub-value in comparison must display 'n/a', not 'None'."""
    p = HCInputPanel()
    p.show_comparison({"gamma": {"lowt": 0.005, "full": None}, "theta_D": {"lowt": "n/a", "full": 220.0}})
    text = p.comparison_label.text()
    assert "None" not in text, f"Got 'None' in label: {text!r}"
    assert "n/a" in text, f"Expected 'n/a' in label: {text!r}"
