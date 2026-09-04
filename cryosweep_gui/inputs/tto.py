from __future__ import annotations
from PySide6.QtWidgets import QLabel
from cryosweep_gui.inputs.base import InputPanel, register_panel


class TTOInputPanel(InputPanel):
    """Thermal Transport needs NO user inputs (D4): kappa and rho are already absolute
    because the sample geometry rides in the file header (SAMPLE_CROSS_SECTION,
    SAMPLE_VLEAD_SEPARATION, SAMPLE_ILEAD_SEPARATION, SAMPLE_EMISSIVITY). The panel exists
    so the tab explains that rather than showing an empty box."""

    def __init__(self):
        super().__init__("tto")
        self._note = QLabel("Sample geometry is read from the file header.")
        self._note.setWordWrap(True)
        self._layout.addWidget(self._note)

    def build_header_patch(self) -> dict:
        return {}

    def get_state(self) -> dict:
        return {}

    def set_state(self, state: dict) -> None:
        pass


register_panel("tto", TTOInputPanel)
