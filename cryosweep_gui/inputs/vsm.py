from __future__ import annotations
from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLineEdit
from cryosweep_gui.inputs.base import InputPanel, register_panel, opt_float

#: (state key, widget attribute, cfg.vsm key)
_WINDOW_BOXES = (("cw_min", "cw_min", "cw_fit_min_k"), ("cw_max", "cw_max", "cw_fit_max_k"),
                 ("cw_mod_min", "cw_mod_min", "cw_mod_fit_min_k"),
                 ("cw_mod_max", "cw_mod_max", "cw_mod_fit_max_k"),
                 ("cw_curve_max", "cw_curve_max", "cw_curve_max_k"))


class VSMInputPanel(InputPanel):
    def __init__(self):
        super().__init__("vsm")
        form = QFormLayout()
        self.molar_mass_edit = QLineEdit()
        self.molar_mass_edit.setPlaceholderText("g/mol — required if not in the file")
        self.mass_mg_edit = QLineEdit()
        self.mass_mg_edit.setPlaceholderText("mg — required if not in the file")
        form.addRow("Molar mass", self.molar_mass_edit)
        form.addRow("Sample mass", self.mass_mg_edit)
        self._layout.addLayout(form)

        # Fit windows bound the points each fit is judged on (empty = every point). The
        # curves are drawn and exported from 1/chi = 0 at each fit's theta to "Curve to"
        # regardless (empty = the highest data temperature).
        box = QGroupBox("Curie-Weiss fit"); fl = QFormLayout(box)

        def _pair(lo_attr, hi_attr):
            lo, hi = QLineEdit(), QLineEdit()
            lo.setPlaceholderText("min T (K)"); hi.setPlaceholderText("max T (K)")
            setattr(self, lo_attr, lo); setattr(self, hi_attr, hi)
            row = QHBoxLayout(); row.addWidget(lo); row.addWidget(hi)
            return row

        fl.addRow("Curie-Weiss window", _pair("cw_min", "cw_max"))
        fl.addRow("Modified CW window", _pair("cw_mod_min", "cw_mod_max"))
        self.cw_curve_max = QLineEdit()
        self.cw_curve_max.setPlaceholderText("max T (K, default: data max)")
        self.cw_curve_max.setToolTip(
            "Upper end of the drawn and exported Curie-Weiss curves. Each curve starts "
            "where it reaches 1/χ = 0, at its own θ. This does not change which points are "
            "fitted. A value below the top of a fit window is raised to it (with a warning): "
            "a curve never stops inside its fitted points.")
        fl.addRow("Curve to", self.cw_curve_max)
        self._layout.addWidget(box)

    def build_header_patch(self) -> dict:
        patch = {}
        mm = opt_float(self.molar_mass_edit.text())
        ms = opt_float(self.mass_mg_edit.text())
        if mm is not None:
            patch["molar_mass"] = mm
        if ms is not None:
            patch["mass_mg"] = ms
        return patch

    def build_overrides(self) -> dict:
        vsm = {}
        for _, attr, key in _WINDOW_BOXES:
            val = opt_float(getattr(self, attr).text())
            if val is not None:
                vsm[key] = val
        return {"vsm": vsm} if vsm else {}

    def get_state(self) -> dict:
        state = {"molar_mass": self.molar_mass_edit.text(), "mass_mg": self.mass_mg_edit.text()}
        state.update({sk: getattr(self, attr).text() for sk, attr, _ in _WINDOW_BOXES})
        return state

    def set_state(self, state: dict) -> None:
        self.molar_mass_edit.setText(state.get("molar_mass", ""))
        self.mass_mg_edit.setText(state.get("mass_mg", ""))
        for sk, attr, _ in _WINDOW_BOXES:
            getattr(self, attr).setText(state.get(sk, ""))

register_panel("vsm", VSMInputPanel)
