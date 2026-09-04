from __future__ import annotations
import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QFormLayout, QLineEdit, QGroupBox, QGridLayout,
                             QLabel, QDoubleSpinBox, QCheckBox, QPushButton, QComboBox,
                             QFileDialog, QHBoxLayout, QWidget)
from cryosweep_gui.inputs.base import InputPanel, register_panel, opt_float
from cryosweep_gui.widgets import CollapsibleGroup

_FULL_KEYS = ("theta_D", "n", "gamma", "theta_E1", "theta_E2", "m1", "m2")
_FULL_DEF = {"theta_D": 100.0, "n": 7.0, "gamma": 0.007,
             "theta_E1": 50.0, "theta_E2": 150.0, "m1": 1.0, "m2": 2.0}
_FULL_FIXDEF = {"n": True}

class HCInputPanel(InputPanel):
    refit_requested = Signal()
    #: A USER edited one of the 7 Debye-Einstein parameter boxes. Programmatic writes
    #: (set_state, absorb_result) are quiet — they must not draw a manual model curve.
    param_edited = Signal()

    def __init__(self):
        super().__init__("heatcapacity")
        self._quiet = False                      # True while set_state/absorb write the boxes
        self._adv_groups: dict[str, CollapsibleGroup] = {}
        form = QFormLayout()
        self.n_atoms_edit = QLineEdit()
        self.n_atoms_edit.setPlaceholderText("atoms / f.u. (optional; default from header)")
        form.addRow("N atoms", self.n_atoms_edit)
        self._layout.addLayout(form)

        # low-T fit range
        low = QGroupBox("Low-T fit"); ll = QFormLayout(low)
        self.lowt_min = QLineEdit(); self.lowt_min.setPlaceholderText("min T (K)")
        self.lowt_max = QLineEdit(); self.lowt_max.setPlaceholderText("max T (K, default 10)")
        ll.addRow("T min", self.lowt_min); ll.addRow("T max", self.lowt_max)
        self.refit_lowt_btn = QPushButton("Refit low-T")
        ll.addRow(self.refit_lowt_btn)
        self._layout.addWidget(low)

        # full-range fit
        box = QGroupBox("Full-range fit (Debye-Einstein)"); g = QGridLayout(box)
        self.full_min = QLineEdit(); self.full_min.setPlaceholderText("min T (K)")
        self.full_max = QLineEdit(); self.full_max.setPlaceholderText("max T (K)")
        g.addWidget(QLabel("T range"), 0, 0); g.addWidget(self.full_min, 0, 1); g.addWidget(self.full_max, 0, 2)
        self._val: dict[str, QDoubleSpinBox] = {}
        self._fix: dict[str, QCheckBox] = {}
        for i, k in enumerate(_FULL_KEYS, start=1):
            # 6 decimals: gamma is ~0.01 J/(mol*K^2), and after a fit these boxes SHOW the
            # fitted values — 4 decimals would truncate gamma to two significant digits.
            sb = QDoubleSpinBox(); sb.setRange(-1e6, 1e6); sb.setDecimals(6)
            sb.setValue(_FULL_DEF[k]); self._val[k] = sb
            cb = QCheckBox("fix"); cb.setChecked(_FULL_FIXDEF.get(k, False)); self._fix[k] = cb
            g.addWidget(QLabel(k), i, 0); g.addWidget(sb, i, 1); g.addWidget(cb, i, 2)
            sb.valueChanged.connect(self._emit_param_edited)
        self.run_full_btn = QPushButton("Run full-range fit")
        g.addWidget(self.run_full_btn, len(_FULL_KEYS) + 1, 0, 1, 3)
        cg_full = CollapsibleGroup("Full-range fit (Debye-Einstein)", collapsed=True)
        box.setTitle("")                              # avoid a doubled title under the toggle header
        cg_full.body_layout.addWidget(box)
        self._adv_groups["full"] = cg_full
        self._layout.addWidget(cg_full)

        # Entropy S(T) controls
        ent = QGroupBox("Entropy"); el = QFormLayout(ent)
        self.entropy_source = QComboBox()
        self.entropy_source.addItems(["Fitted (Debye-Einstein)", "Reference file…"])
        self.entropy_source.setCurrentIndex(0)
        el.addRow("Lattice source", self.entropy_source)
        path_row = QWidget(); pr = QHBoxLayout(path_row); pr.setContentsMargins(0, 0, 0, 0)
        self.entropy_ref_path = QLineEdit(); self.entropy_ref_path.setPlaceholderText("reference .dat path")
        self.entropy_ref_browse = QPushButton("Browse…")
        pr.addWidget(self.entropy_ref_path); pr.addWidget(self.entropy_ref_browse)
        el.addRow("Reference file", path_row)
        self.entropy_extrapolate = QCheckBox("Extrapolate to 0 K"); self.entropy_extrapolate.setChecked(True)
        el.addRow(self.entropy_extrapolate)
        self.entropy_rln_j = QDoubleSpinBox()
        self.entropy_rln_j.setRange(0.0, 10.0); self.entropy_rln_j.setSingleStep(0.5)
        self.entropy_rln_j.setValue(0.0)
        el.addRow("Rln J (0 = auto)", self.entropy_rln_j)
        self._layout.addWidget(ent)
        self.entropy_source.currentIndexChanged.connect(self._sync_entropy_ref_enabled)
        self.entropy_ref_browse.clicked.connect(self._browse_entropy_ref)
        self._sync_entropy_ref_enabled()

        # Schottky anomaly controls
        sch = QGroupBox("Schottky (low-T anomaly)"); sl = QFormLayout(sch)
        self.schottky_enable = QCheckBox("Enable Schottky fit")
        self.schottky_enable.setChecked(False)
        sl.addRow(self.schottky_enable)
        self.schottky_r = QDoubleSpinBox(); self.schottky_r.setRange(0.01, 100.0)
        self.schottky_r.setDecimals(2); self.schottky_r.setValue(1.0)
        sl.addRow("g0/g1 ratio (r)", self.schottky_r)
        self.schottky_t5 = QCheckBox("add δT⁵"); self.schottky_t5.setChecked(False)
        sl.addRow(self.schottky_t5)
        self.schottky_nuclear = QCheckBox("nuclear αN/T² tail"); self.schottky_nuclear.setChecked(False)
        sl.addRow(self.schottky_nuclear)
        self.schottky_dh_model = QComboBox()
        self.schottky_dh_model.addItems(["none", "zeeman", "zfs"])
        self.schottky_dh_model.setCurrentIndex(0)
        sl.addRow("ΔH model", self.schottky_dh_model)
        cg_sch = CollapsibleGroup("Schottky (low-T anomaly)", collapsed=True)
        sch.setTitle("")
        cg_sch.body_layout.addWidget(sch)
        self._adv_groups["schottky"] = cg_sch
        self._layout.addWidget(cg_sch)

        # Transition search controls (opt-in)
        trg = QGroupBox("Transition search (opt-in)"); tl = QFormLayout(trg)
        self.transition_enable = QCheckBox("Enable transition search")
        self.transition_enable.setChecked(False)
        tl.addRow(self.transition_enable)
        self.transition_form = QComboBox(); self.transition_form.addItems(["lambda", "jump"])
        tl.addRow("form", self.transition_form)
        self.transition_universality = QComboBox()
        self.transition_universality.addItems(["mean_field", "ising3d", "xy3d"])
        tl.addRow("universality", self.transition_universality)
        self.transition_t5 = QCheckBox("add δT⁵ (background)"); self.transition_t5.setChecked(False)
        tl.addRow(self.transition_t5)
        self.transition_compare = QCheckBox("compare λ vs jump"); self.transition_compare.setChecked(False)
        tl.addRow(self.transition_compare)
        cg_tr = CollapsibleGroup("Transition search (opt-in)", collapsed=True)
        trg.setTitle("")
        cg_tr.body_layout.addWidget(trg)
        self._adv_groups["transition"] = cg_tr
        self._layout.addWidget(cg_tr)

        self.comparison_label = QLabel(""); self.comparison_label.setWordWrap(True)
        self._layout.addWidget(self.comparison_label)

        self.run_full_btn.clicked.connect(self.refit_requested)
        self.refit_lowt_btn.clicked.connect(self.refit_requested)

    # helpers used by tests + wiring
    def set_value(self, key, v): self._val[key].setValue(float(v))
    def set_fix(self, key, on): self._fix[key].setChecked(bool(on))

    def _emit_param_edited(self, *_):
        if not self._quiet:
            self.param_edited.emit()

    # ---- fitted values, both directions (ROADMAP item 2) ----

    @staticmethod
    def _fitted_params(result) -> dict | None:
        """The full-range fit's params, or None when there is no ACCEPTED fit. A declined
        fit (ok False, r² floor, unavailable) must never overwrite the user's guesses."""
        ff = ((getattr(result, "data", None) or {}).get("full_fit") or {}) if result else {}
        if not ff.get("ok"):
            return None
        p = ff.get("params") or {}
        if not all(k in p and p[k] is not None for k in _FULL_KEYS):
            return None
        return {k: float(p[k]) for k in _FULL_KEYS}

    def absorb_result(self, result) -> bool:
        """2(a): write the FITTED values into the parameter boxes (quietly — this is not a
        user edit). Returns True when something was absorbed."""
        vals = self._fitted_params(result)
        if vals is None:
            return False
        self._quiet = True
        try:
            for k, v in vals.items():
                self._val[k].setValue(v)
        finally:
            self._quiet = False
        return True

    def fitted_state_patch(self, state: dict, result) -> dict | None:
        """2(a), stored-state side: a copy of *state* whose 'val' carries the fitted params,
        or None when there is no accepted fit. Pure dict work — no widget writes — so
        ProbeTab.analyze_and_render can fold each entry's own fit into that entry's state
        before its final set_state restore (which would otherwise re-show the guesses)."""
        vals = self._fitted_params(result)
        if vals is None:
            return None
        return {**state, "val": vals}

    def manual_model_curve(self, result):
        """2(b): evaluate the Debye-Einstein model at the CURRENT box values over the
        result's T range. Returns (x, y, label) or None. This is a model evaluation, never
        a refit — whatever the caller draws from it must be labelled a manual model, not a
        fit (physics-integrity rule)."""
        from cryosweep_core.fitting.heat_capacity import specific_heat_full
        d = (getattr(result, "data", None) or {}) if result else {}
        if d.get("probe") != "heatcapacity":
            return None
        grid = (d.get("full_fit") or {}).get("t_grid") or []
        if not grid:
            T = [t for t in (d.get("full_temperature") or []) if t and t > 0]
            if len(T) < 2:
                return None
            grid = np.linspace(min(T), max(T), 300)
        x = np.asarray(grid, float)
        x = x[np.isfinite(x) & (x > 0)]
        if x.size < 2:
            return None
        params = {k: float(self._val[k].value()) for k in _FULL_KEYS}
        try:
            y = specific_heat_full(x, **params)
        except (ValueError, FloatingPointError, OverflowError):
            return None                          # unphysical hand-set params: no curve
        if not np.all(np.isfinite(y)):
            return None
        return x.tolist(), np.asarray(y, float).tolist(), "model (manual)"

    def _entropy_ref_selected(self) -> bool:
        return self.entropy_source.currentIndex() == 1

    def _sync_entropy_ref_enabled(self) -> None:
        on = self._entropy_ref_selected()
        self.entropy_ref_path.setEnabled(on)
        self.entropy_ref_browse.setEnabled(on)

    def _browse_entropy_ref(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select reference .dat", "", "Data files (*.dat);;All files (*)")
        if path:
            self.entropy_ref_path.setText(path)

    def build_header_patch(self) -> dict:
        v = opt_float(self.n_atoms_edit.text())
        return {"n_atoms": v} if v is not None else {}

    def build_overrides(self) -> dict:
        hc = {
            "full_init": {k: float(self._val[k].value()) for k in _FULL_KEYS},
            "full_fixed": {k: bool(self._fix[k].isChecked()) for k in _FULL_KEYS},
        }
        for name, w in (("full_fit_min_k", self.full_min), ("full_fit_max_k", self.full_max),
                        ("lowt_fit_min_k", self.lowt_min), ("lowt_fit_max_k", self.lowt_max)):
            val = opt_float(w.text())
            if val is not None:
                hc[name] = val
        hc["schottky_enabled"] = bool(self.schottky_enable.isChecked())
        hc["schottky_r"] = float(self.schottky_r.value())
        hc["schottky_lattice_t5"] = bool(self.schottky_t5.isChecked())
        hc["schottky_include_nuclear"] = bool(self.schottky_nuclear.isChecked())
        hc["schottky_delta_h_model"] = self.schottky_dh_model.currentText()
        hc["transitions_enabled"] = bool(self.transition_enable.isChecked())
        hc["transition_form"] = self.transition_form.currentText()
        hc["transition_universality"] = self.transition_universality.currentText()
        hc["transition_lattice_t5"] = bool(self.transition_t5.isChecked())
        hc["transition_compare_forms"] = bool(self.transition_compare.isChecked())
        hc["entropy_extrapolate"] = bool(self.entropy_extrapolate.isChecked())
        if self._entropy_ref_selected():
            ref = self.entropy_ref_path.text().strip()
            if ref:
                hc["entropy_lattice_ref_file"] = ref
        rln_j = float(self.entropy_rln_j.value())
        if rln_j > 0:
            hc["entropy_rln_j"] = rln_j
        return {"heatcapacity": hc}

    def get_state(self) -> dict:
        return {"n_atoms": self.n_atoms_edit.text(),
                "lowt_min": self.lowt_min.text(), "lowt_max": self.lowt_max.text(),
                "full_min": self.full_min.text(), "full_max": self.full_max.text(),
                "val": {k: self._val[k].value() for k in _FULL_KEYS},
                "fix": {k: self._fix[k].isChecked() for k in _FULL_KEYS},
                "schottky_enabled": self.schottky_enable.isChecked(),
                "schottky_r": self.schottky_r.value(),
                "schottky_t5": self.schottky_t5.isChecked(),
                "schottky_nuclear": self.schottky_nuclear.isChecked(),
                "schottky_dh_model": self.schottky_dh_model.currentText(),
                "transitions_enabled": self.transition_enable.isChecked(),
                "transition_form": self.transition_form.currentText(),
                "transition_universality": self.transition_universality.currentText(),
                "transition_t5": self.transition_t5.isChecked(),
                "transition_compare": self.transition_compare.isChecked(),
                "entropy_source": self.entropy_source.currentIndex(),
                "entropy_ref_path": self.entropy_ref_path.text(),
                "entropy_extrapolate": self.entropy_extrapolate.isChecked(),
                "entropy_rln_j": self.entropy_rln_j.value()}

    def set_state(self, state: dict) -> None:
        self._quiet = True                       # programmatic restore, not a user edit
        try:
            self._set_state(state)
        finally:
            self._quiet = False

    def _set_state(self, state: dict) -> None:
        self.n_atoms_edit.setText(state.get("n_atoms", ""))
        self.lowt_min.setText(state.get("lowt_min", "")); self.lowt_max.setText(state.get("lowt_max", ""))
        self.full_min.setText(state.get("full_min", "")); self.full_max.setText(state.get("full_max", ""))
        for k, v in (state.get("val") or {}).items():
            if k in self._val: self._val[k].setValue(float(v))
        for k, v in (state.get("fix") or {}).items():
            if k in self._fix: self._fix[k].setChecked(bool(v))
        self.schottky_enable.setChecked(bool(state.get("schottky_enabled", False)))
        self.schottky_r.setValue(float(state.get("schottky_r", 1.0)))
        self.schottky_t5.setChecked(bool(state.get("schottky_t5", False)))
        self.schottky_nuclear.setChecked(bool(state.get("schottky_nuclear", False)))
        dh = state.get("schottky_dh_model", "none")
        idx = self.schottky_dh_model.findText(dh)
        if idx >= 0:
            self.schottky_dh_model.setCurrentIndex(idx)
        self.transition_enable.setChecked(bool(state.get("transitions_enabled", False)))
        fi = self.transition_form.findText(state.get("transition_form", "lambda"))
        if fi >= 0:
            self.transition_form.setCurrentIndex(fi)
        ui = self.transition_universality.findText(state.get("transition_universality", "mean_field"))
        if ui >= 0:
            self.transition_universality.setCurrentIndex(ui)
        self.transition_t5.setChecked(bool(state.get("transition_t5", False)))
        self.transition_compare.setChecked(bool(state.get("transition_compare", False)))
        self.entropy_source.blockSignals(True)
        si = int(state.get("entropy_source", 0))
        self.entropy_source.setCurrentIndex(si if 0 <= si < self.entropy_source.count() else 0)
        self.entropy_source.blockSignals(False)
        self.entropy_ref_path.setText(state.get("entropy_ref_path", ""))
        self.entropy_extrapolate.setChecked(bool(state.get("entropy_extrapolate", True)))
        self.entropy_rln_j.setValue(float(state.get("entropy_rln_j", 0.0)))
        self._sync_entropy_ref_enabled()

    def show_comparison(self, comparison: dict | None) -> None:
        if not comparison:
            self.comparison_label.setText(""); return
        def f(x): return f"{x:.4g}" if isinstance(x, (int, float)) else ("n/a" if x is None else str(x))
        g = comparison.get("gamma", {}); t = comparison.get("theta_D", {})
        r2 = comparison.get("r2", {})
        text = (f"γ: low-T {f(g.get('lowt'))} | full {f(g.get('full'))}    "
                f"θ_D: low-T {f(t.get('lowt'))} | full {f(t.get('full'))}")
        if r2.get("full") is not None:
            text += f"    full R²: {f(r2.get('full'))}"
        self.comparison_label.setText(text)

register_panel("heatcapacity", HCInputPanel)
