from __future__ import annotations
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QLineEdit, QComboBox, QPushButton, QLabel, QFileDialog
from cryosweep_gui.inputs.base import InputPanel, register_panel, opt_float

_UNIT_MM = {"mm": 1.0, "um": 1e-3, "nm": 1e-6}

class HallInputPanel(InputPanel):
    refit_requested = Signal()   # probe_tab connects this to analyze_and_render

    def __init__(self):
        super().__init__("hall")
        self._long_file = None
        form = QFormLayout()
        self.hall_channel_edit = QLineEdit(); self.hall_channel_edit.setPlaceholderText("bridge # (required)")
        self.thickness_edit = QLineEdit(); self.thickness_edit.setPlaceholderText("thickness (for R_H)")
        self.thickness_unit = QComboBox(); self.thickness_unit.addItems(["mm", "um", "nm"])
        self.geometry_sign = QComboBox(); self.geometry_sign.addItems(["+1", "-1"])
        self.long_channel_edit = QLineEdit(); self.long_channel_edit.setPlaceholderText("bridge # (optional, mobility)")
        self.long_file_btn = QPushButton("Choose longitudinal file…")
        self.long_file_label = QLabel("(same file)")
        form.addRow("Hall channel", self.hall_channel_edit)
        form.addRow("Thickness", self.thickness_edit)
        form.addRow("Thickness unit", self.thickness_unit)
        form.addRow("Geometry sign", self.geometry_sign)
        form.addRow("Longitudinal channel", self.long_channel_edit)
        form.addRow("Longitudinal file", self.long_file_btn)
        form.addRow("", self.long_file_label)
        self._layout.addLayout(form)
        self.long_file_btn.clicked.connect(self._choose_long_file)
        # inputs that change the analysis re-run it live (R_H sign/scale, mobility source);
        # setText is signal-free, so set_state/prefill never re-trigger, but the combos must
        # be restored under blockSignals in set_state
        self.geometry_sign.currentTextChanged.connect(lambda *_: self.refit_requested.emit())
        self.thickness_unit.currentTextChanged.connect(lambda *_: self.refit_requested.emit())
        self.thickness_edit.editingFinished.connect(self.refit_requested)
        self.long_channel_edit.editingFinished.connect(self.refit_requested)

    def _choose_long_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Longitudinal file", "", "PPMS data (*.dat);;All files (*)")
        if path:
            self.set_longitudinal_file(path)

    def set_longitudinal_file(self, path) -> None:
        self._long_file = path or None
        self.long_file_label.setText(path if path else "(same file)")

    def set_hall_channel(self, channel) -> None:
        """Pre-fill the Hall channel field (a suggestion the user can overwrite)."""
        self.hall_channel_edit.setText(str(int(channel)))

    def build_overrides(self) -> dict:
        hall = {}
        ch = opt_float(self.hall_channel_edit.text())
        if ch is not None:
            hall["hall_channel"] = int(ch)
        thk = opt_float(self.thickness_edit.text())
        if thk is not None:
            hall["thickness_mm"] = thk * _UNIT_MM[self.thickness_unit.currentText()]
        hall["geometry_sign"] = int(self.geometry_sign.currentText())   # int("+1")==1, int("-1")==-1
        lch = opt_float(self.long_channel_edit.text())
        if lch is not None:
            hall["longitudinal_channel"] = int(lch)
        if self._long_file:
            hall["longitudinal_file"] = self._long_file
        return {"hall": hall}

    def get_state(self) -> dict:
        return {"hall_channel": self.hall_channel_edit.text(),
                "thickness": self.thickness_edit.text(),
                "thickness_unit": self.thickness_unit.currentText(),
                "geometry_sign": self.geometry_sign.currentText(),
                "long_channel": self.long_channel_edit.text(),
                "long_file": self._long_file}

    def set_state(self, state: dict) -> None:
        self.hall_channel_edit.setText(state.get("hall_channel", ""))
        self.thickness_edit.setText(state.get("thickness", ""))
        for combo, key, default in ((self.thickness_unit, "thickness_unit", "mm"),
                                    (self.geometry_sign, "geometry_sign", "+1")):
            combo.blockSignals(True)                 # restore must not emit refit_requested
            combo.setCurrentText(state.get(key, default))
            combo.blockSignals(False)
        self.long_channel_edit.setText(state.get("long_channel", ""))
        self.set_longitudinal_file(state.get("long_file"))

register_panel("hall", HallInputPanel)
register_panel("hall_tdep", HallInputPanel)   # D7: Temp-Dep Hall reuses the Hall inputs (reads cfg.hall)
