from __future__ import annotations
from PySide6.QtWidgets import QFormLayout, QLineEdit, QCheckBox
from cryosweep_gui.inputs.base import InputPanel, register_panel, opt_float

class ResistivityInputPanel(InputPanel):
    def __init__(self):
        super().__init__("resistivity")
        form = QFormLayout()
        self.width_edit = QLineEdit(); self.width_edit.setPlaceholderText("mm (optional)")
        self.thickness_edit = QLineEdit(); self.thickness_edit.setPlaceholderText("mm (optional)")
        self.length_edit = QLineEdit(); self.length_edit.setPlaceholderText("mm (optional)")
        form.addRow("Width", self.width_edit)
        form.addRow("Thickness", self.thickness_edit)
        form.addRow("Length", self.length_edit)
        self.exclude_cb = QCheckBox("Exclude outliers (robust)")
        form.addRow("", self.exclude_cb)
        self._layout.addLayout(form)

    def build_overrides(self) -> dict:
        geom = {}
        for key, edit in (("width_mm", self.width_edit),
                          ("thickness_mm", self.thickness_edit),
                          ("length_mm", self.length_edit)):
            v = opt_float(edit.text())
            if v is not None:
                geom[key] = v
        ov = {"geometry": geom} if geom else {}
        if self.exclude_cb.isChecked():
            ov["quality"] = {"exclude_outliers": True}
        return ov

    def get_state(self) -> dict:
        return {"width": self.width_edit.text(), "thickness": self.thickness_edit.text(),
                "length": self.length_edit.text(), "exclude": self.exclude_cb.isChecked()}

    def set_state(self, state: dict) -> None:
        self.width_edit.setText(state.get("width", ""))
        self.thickness_edit.setText(state.get("thickness", ""))
        self.length_edit.setText(state.get("length", ""))
        self.exclude_cb.setChecked(state.get("exclude", False))

register_panel("resistivity", ResistivityInputPanel)
