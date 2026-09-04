import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from cryosweep_gui.inputs.hc import HCInputPanel
_app = QApplication.instance() or QApplication([])

def test_show_comparison_updates_label():
    p = HCInputPanel()
    p.show_comparison({"gamma": {"lowt": 0.0068, "full": 0.0071},
                       "theta_D": {"lowt": 182.0, "full": 176.0}})
    assert "γ" in p.comparison_label.text() and "182" in p.comparison_label.text()
