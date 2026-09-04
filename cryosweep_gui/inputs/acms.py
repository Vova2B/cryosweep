from __future__ import annotations
from PySide6.QtWidgets import QFormLayout, QLineEdit
from cryosweep_gui.inputs.base import InputPanel, register_panel, opt_float

class ACMSInputPanel(InputPanel):
    def __init__(self):
        super().__init__("acms")
        form = QFormLayout()
        self.molar_mass_edit = QLineEdit()
        self.molar_mass_edit.setPlaceholderText("g/mol (optional)")
        self.mass_mg_edit = QLineEdit()
        self.mass_mg_edit.setPlaceholderText("mg (optional)")
        form.addRow("Molar mass", self.molar_mass_edit)
        form.addRow("Sample mass", self.mass_mg_edit)
        self._layout.addLayout(form)

    def build_header_patch(self) -> dict:
        patch = {}
        mm = opt_float(self.molar_mass_edit.text())
        ms = opt_float(self.mass_mg_edit.text())
        if mm is not None:
            patch["molar_mass"] = mm
        if ms is not None:
            patch["mass_mg"] = ms
        return patch

    def get_state(self) -> dict:
        return {"molar_mass": self.molar_mass_edit.text(), "mass_mg": self.mass_mg_edit.text()}

    def set_state(self, state: dict) -> None:
        self.molar_mass_edit.setText(state.get("molar_mass", ""))
        self.mass_mg_edit.setText(state.get("mass_mg", ""))

register_panel("acms", ACMSInputPanel)
