from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                             QLineEdit, QPushButton, QLabel, QComboBox, QFormLayout,
                             QCheckBox, QGroupBox, QScrollArea, QDoubleSpinBox, QSpinBox, QToolButton)
from cryosweep_core.plotting.spec import GlobalStyle, PlotLayout, PlotEntry

_KEY_ROLE = Qt.ItemDataRole.UserRole + 2


class CurveChecklist(QWidget):
    """Scroll-safe, grouped, filterable curve on/off list. Fixed-height -> scrolls (P1)."""
    changed = Signal(list)                           # emits checked keys

    def __init__(self, series):
        super().__init__()
        lay = QVBoxLayout(self)
        bar = QHBoxLayout()
        self._filter = QLineEdit(); self._filter.setPlaceholderText("filter…")
        bar.addWidget(self._filter)
        for txt, fn in (("all", self.select_all), ("none", self.select_none), ("inv", self.invert)):
            b = QPushButton(txt); b.setMaximumWidth(48); b.clicked.connect(fn); bar.addWidget(b)
        lay.addLayout(bar)
        self._list = QListWidget()
        self._list.setMaximumHeight(160)                 # bounded -> scrolls, never collapses
        lay.addWidget(self._list)
        self._filter.textChanged.connect(self.set_filter)
        self._list.itemChanged.connect(lambda *_: self._emit())
        self._populate(series)

    def _populate(self, series):
        self._list.blockSignals(True)
        last_group = object()
        for s in series:
            if s.group and s.group != last_group:
                hdr = QListWidgetItem(f"— {s.group} —")
                hdr.setFlags(Qt.ItemFlag.NoItemFlags)
                self._list.addItem(hdr); last_group = s.group
            it = QListWidgetItem(s.label)
            it.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            it.setCheckState(Qt.CheckState.Checked if s.default_on else Qt.CheckState.Unchecked)
            it.setData(_KEY_ROLE, s.key)
            self._list.addItem(it)
        self._list.blockSignals(False)

    def _items(self):
        out = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_KEY_ROLE) is not None and not it.isHidden():
                out.append(it)
        return out

    def set_filter(self, text):
        text = (text or "").lower()
        header = None
        header_has_visible = False
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_KEY_ROLE) is None:                 # group header row
                if header is not None:
                    header.setHidden(not header_has_visible)
                header = it
                header_has_visible = False
            else:
                vis = text in it.text().lower()
                it.setHidden(not vis)
                header_has_visible = header_has_visible or vis
        if header is not None:                             # finalize the last group
            header.setHidden(not header_has_visible)

    def checked_keys(self):
        out = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_KEY_ROLE) is not None and it.checkState() == Qt.CheckState.Checked:
                out.append(it.data(_KEY_ROLE))
        return out

    def _all_data_items(self):
        """All checkable items regardless of filter visibility."""
        out = []
        for i in range(self._list.count()):
            it = self._list.item(i)
            if it.data(_KEY_ROLE) is not None:
                out.append(it)
        return out

    def select_all(self):
        """Check all visible (filtered) items."""
        self._list.blockSignals(True)
        for it in self._items():                          # filtered scope only
            it.setCheckState(Qt.CheckState.Checked)
        self._list.blockSignals(False)
        self._emit()

    def select_none(self):
        """Uncheck ALL items (ignores filter) so subsequent select_all is unambiguous."""
        self._list.blockSignals(True)
        for it in self._all_data_items():                 # global scope
            it.setCheckState(Qt.CheckState.Unchecked)
        self._list.blockSignals(False)
        self._emit()

    def invert(self):
        self._list.blockSignals(True)
        for it in self._items():
            it.setCheckState(Qt.CheckState.Unchecked if it.checkState() == Qt.CheckState.Checked
                             else Qt.CheckState.Checked)
        self._list.blockSignals(False)
        self._emit()

    def _emit(self):
        self.changed.emit(self.checked_keys())


def _opt_float(text):
    text = (text or "").strip()
    try:
        return float(text) if text else None
    except ValueError:
        return None


class AxisStrip(QWidget):
    """Per-plot axis min/max (apply-on-commit) + x/y scale combos + a curve checklist.
    The form + checklist live in a collapsible body, hidden by default so the canvas
    is visible immediately.
    """
    spec_changed = Signal()

    def __init__(self, series, spec, kind):
        super().__init__()
        self._spec = spec
        self._kind = kind
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        self._toggle = QToolButton(); self._toggle.setText("▸ controls")
        self._toggle.setCheckable(True); self._toggle.setChecked(False)
        self._toggle.setStyleSheet("QToolButton { border: none; }")
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle)
        self._body = QWidget(); lay = QVBoxLayout(self._body)
        form = QFormLayout()
        self._xmin = QLineEdit(); self._xmax = QLineEdit()
        self._ymin = QLineEdit(); self._ymax = QLineEdit()
        for e in (self._xmin, self._xmax, self._ymin, self._ymax):
            e.editingFinished.connect(self._commit_axes)
        xb = QHBoxLayout(); xb.addWidget(self._xmin); xb.addWidget(self._xmax)
        yb = QHBoxLayout(); yb.addWidget(self._ymin); yb.addWidget(self._ymax)
        xw = QWidget(); xw.setLayout(xb); yw = QWidget(); yw.setLayout(yb)
        form.addRow("x min/max", xw); form.addRow("y min/max", yw)
        self._xscale = QComboBox(); self._xscale.addItems(["linear", "log"])
        self._yscale = QComboBox(); self._yscale.addItems(["linear", "log"])
        self._xscale.setCurrentText(spec.xscale or kind.default_xscale)
        self._yscale.setCurrentText(spec.yscale or kind.default_yscale)
        self._xscale.currentTextChanged.connect(self._commit_scale)
        self._yscale.currentTextChanged.connect(self._commit_scale)
        form.addRow("x scale", self._xscale); form.addRow("y scale", self._yscale)
        self._robust_cb = QCheckBox("Robust view")
        rv = spec.robust_view
        self._robust_cb.setChecked(True if rv is None else bool(rv))
        self._robust_cb.toggled.connect(self._commit_robust)
        form.addRow("", self._robust_cb)
        # per-plot EXPORT size override (0 = "auto" -> GlobalStyle width/height)
        self._w_mm = QDoubleSpinBox(); self._h_mm = QDoubleSpinBox()
        for sb, val in ((self._w_mm, spec.width_mm), (self._h_mm, spec.height_mm)):
            sb.setRange(0.0, 1000.0); sb.setDecimals(1)
            sb.setSpecialValueText("auto")
            sb.setValue(val if val is not None else 0.0)
            sb.editingFinished.connect(self._commit_mm)
        mm_row = QHBoxLayout(); mm_row.addWidget(self._w_mm); mm_row.addWidget(self._h_mm)
        mmw = QWidget(); mmw.setLayout(mm_row)
        form.addRow("export W/H mm", mmw)
        lay.addLayout(form)
        self.checklist = CurveChecklist(series)
        self.checklist.changed.connect(self._commit_curves)
        lay.addWidget(self.checklist)
        if kind.key == "resistivity_rho_t2":
            fit_row = QHBoxLayout()
            self._fit_linear_cb = QCheckBox("βT² fit")
            self._fit_power_cb = QCheckBox("power-law fit")
            fl = spec.fit_lines
            self._fit_linear_cb.setChecked(fl is None or "linear" in fl)
            self._fit_power_cb.setChecked(fl is None or "power_law" in fl)
            self._fit_linear_cb.toggled.connect(self._commit_fit_lines)
            self._fit_power_cb.toggled.connect(self._commit_fit_lines)
            fit_row.addWidget(self._fit_linear_cb); fit_row.addWidget(self._fit_power_cb)
            fw = QWidget(); fw.setLayout(fit_row); lay.addWidget(fw)
        if kind.key == "cp_over_t":
            from cryosweep_core.plotting.render import _LOWT_FIT_KEYS
            labels = {"debye_t3": "Debye T³", "debye_t3_t5": "Debye T³+T⁵",
                      "spin_fluct_noninteracting": "spin-fl non-int", "spin_fluct_weak": "spin-fl weak"}
            lowt_row = QHBoxLayout(); self._lowt_cbs = {}
            fl = spec.fit_lines
            for key in _LOWT_FIT_KEYS:
                cb = QCheckBox(labels[key]); cb.setChecked(fl is None or key in fl)
                cb.toggled.connect(self._commit_lowt_fit_lines)
                self._lowt_cbs[key] = cb; lowt_row.addWidget(cb)
            lw = QWidget(); lw.setLayout(lowt_row); lay.addWidget(lw)
        if kind.key == "cp_vs_t":
            model_row = QHBoxLayout()
            self._model_cb = QCheckBox("Debye-Einstein model")
            self._model_cb.setChecked(spec.fit_line)
            self._model_cb.toggled.connect(self.set_model_visible)
            model_row.addWidget(self._model_cb)
            mw = QWidget(); mw.setLayout(model_row); lay.addWidget(mw)
        if kind.key == "hc_lowt_multifield":
            models = ["debye_t3", "debye_t3_t5", "spin_fluct_noninteracting", "spin_fluct_weak"]
            _mk_labels = {"debye_t3": "Debye T³", "debye_t3_t5": "Debye T³+T⁵",
                          "spin_fluct_noninteracting": "spin-fl non-int", "spin_fluct_weak": "spin-fl weak"}
            fields = sorted({s.group for s in series})        # e.g. "10000 Oe" (one per field group)
            fl = spec.fit_lines
            self._mf_fit_cbs = {}
            mf_row = QHBoxLayout()
            for fld in fields:
                fnum = fld.split()[0]                          # numeric part -> matches render's model@field key
                for mk in models:
                    lkey = f"{mk}@{fnum}"
                    cb = QCheckBox(f"{_mk_labels[mk]} @ {fnum} Oe"); cb.setChecked(fl is None or lkey in fl)
                    cb.toggled.connect(self._commit_mf_fit_lines)
                    self._mf_fit_cbs[lkey] = cb; mf_row.addWidget(cb)
            mw = QWidget(); mw.setLayout(mf_row); lay.addWidget(mw)
        from cryosweep_core.plotting.render import _TTO_BAND_KINDS, _SHADE_KINDS
        if kind.key in _TTO_BAND_KINDS:
            # I6: PlotSpec.error_band is default-OFF, so without a control it is a feature no
            # user can reach. Same per-plot boolean idiom as cp_vs_t's model checkbox.
            band_row = QHBoxLayout()
            self._error_band_cb = QCheckBox("Error band")
            self._error_band_cb.setChecked(bool(spec.error_band))
            self._error_band_cb.toggled.connect(self.set_error_band)
            band_row.addWidget(self._error_band_cb)
            bw = QWidget(); bw.setLayout(band_row); lay.addWidget(bw)
        if kind.key in _SHADE_KINDS:
            # fit_window_shade is default-OFF too (owner 2026-09-05) — same reachability
            # rule as error_band: a default-OFF spec field needs a control.
            shade_row = QHBoxLayout()
            self._fit_shade_cb = QCheckBox("Fit-window shade")
            self._fit_shade_cb.setChecked(bool(spec.fit_window_shade))
            self._fit_shade_cb.toggled.connect(self.set_fit_window_shade)
            shade_row.addWidget(self._fit_shade_cb)
            sw = QWidget(); sw.setLayout(shade_row); lay.addWidget(sw)
        self._body.setVisible(False)            # collapsed by default -> canvas shows first
        outer.addWidget(self._body)

    def _on_toggle(self, on):
        self._body.setVisible(on)
        self._toggle.setText("▾ controls" if on else "▸ controls")

    def _commit_axes(self):
        self._spec.xmin = _opt_float(self._xmin.text()); self._spec.xmax = _opt_float(self._xmax.text())
        self._spec.ymin = _opt_float(self._ymin.text()); self._spec.ymax = _opt_float(self._ymax.text())
        self.spec_changed.emit()

    def _commit_scale(self, *_):
        self._spec.xscale = self._xscale.currentText()
        self._spec.yscale = self._yscale.currentText()
        self.spec_changed.emit()

    def _commit_robust(self, on):
        self._spec.robust_view = bool(on)
        self.spec_changed.emit()

    def _commit_mm(self):
        w = self._w_mm.value(); h = self._h_mm.value()
        self._spec.width_mm = w if w > 0 else None
        self._spec.height_mm = h if h > 0 else None
        self.spec_changed.emit()

    def _commit_curves(self, keys):
        self._spec.curves = list(keys)
        self.spec_changed.emit()

    def _commit_fit_lines(self):
        wanted = []
        if self._fit_linear_cb.isChecked():
            wanted.append("linear")
        if self._fit_power_cb.isChecked():
            wanted.append("power_law")
        self._spec.fit_lines = tuple(wanted)
        self.spec_changed.emit()

    def _commit_lowt_fit_lines(self):
        wanted = tuple(k for k, cb in self._lowt_cbs.items() if cb.isChecked())
        self._spec.fit_lines = wanted
        self.spec_changed.emit()

    def _commit_mf_fit_lines(self):
        wanted = tuple(k for k, cb in self._mf_fit_cbs.items() if cb.isChecked())
        self._spec.fit_lines = wanted
        self.spec_changed.emit()

    def set_model_visible(self, on):
        self._spec.fit_line = bool(on)
        self.spec_changed.emit()

    def set_error_band(self, on):
        self._spec.error_band = bool(on)
        self.spec_changed.emit()

    def set_fit_window_shade(self, on):
        self._spec.fit_window_shade = bool(on)
        self.spec_changed.emit()

    def set_axis(self, xmin=None, xmax=None, ymin=None, ymax=None):
        for e, v in ((self._xmin, xmin), (self._xmax, xmax), (self._ymin, ymin), (self._ymax, ymax)):
            if v is not None:
                e.setText(repr(v))
        self._commit_axes()

    def yscale_value(self): return self._yscale.currentText()


class PlotControlsPanel(QWidget):
    """Left-column panel: plot kind on/off checkboxes + global styling controls.

    Wrapped in a QScrollArea so it never overflows vertically (P2).
    """
    layout_changed = Signal(PlotLayout)
    style_changed = Signal(GlobalStyle)

    def __init__(self, registry):
        super().__init__()
        self._registry = registry
        self.style = GlobalStyle()
        self._result = None
        self._probe = None
        self._boxes: dict[str, QCheckBox] = {}
        self._busy = False          # guard against toggled→set_kind_enabled recursion

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        host = QWidget()
        self._col = QVBoxLayout(host)
        self._col.setContentsMargins(4, 4, 4, 4)

        self._plots_box = QGroupBox("Plots")
        self._plots_lay = QVBoxLayout(self._plots_box)
        self._col.addWidget(self._plots_box)
        self._col.addWidget(self._build_styling())
        self._col.addWidget(self._build_journal_frame())
        self._col.addStretch(1)

        scroll.setWidget(host)
        outer.addWidget(scroll)

    # ------------------------------------------------------------------ styling

    def _auto_spin(self, hi=72.0):
        sb = QDoubleSpinBox(); sb.setRange(0.0, hi); sb.setSpecialValueText("auto"); sb.setValue(0.0)
        return sb

    def _build_styling(self):
        box = QGroupBox("Styling")
        form = QFormLayout(box)
        self._marker = QComboBox(); self._marker.addItems(["o", "s", "^", "d", "x", "+"])
        self._marker.setCurrentText(self.style.marker); self._marker.currentTextChanged.connect(self.set_marker)
        self._family = QComboBox()
        self._family.addItems(["(default)", "DejaVu Sans", "sans-serif", "serif", "Arial", "Times New Roman"])
        self._family.currentTextChanged.connect(self._set_family)
        self._font = QDoubleSpinBox(); self._font.setRange(4, 24); self._font.setValue(self.style.font_pt)
        self._font.valueChanged.connect(self._set_font)
        self._msize = QDoubleSpinBox(); self._msize.setRange(1, 20); self._msize.setValue(self.style.marker_size)
        self._msize.valueChanged.connect(self._set_msize)
        self._lwidth = QDoubleSpinBox(); self._lwidth.setRange(0.2, 8); self._lwidth.setValue(self.style.line_width)
        self._lwidth.valueChanged.connect(self._set_lwidth)
        self._label_sz = self._auto_spin(); self._label_sz.valueChanged.connect(self._set_label_sz)
        self._title_sz = self._auto_spin(); self._title_sz.valueChanged.connect(self._set_title_sz)
        self._tick_sz = self._auto_spin(); self._tick_sz.valueChanged.connect(self._set_tick_sz)
        self._legend_sz = self._auto_spin(); self._legend_sz.valueChanged.connect(self._set_legend_sz)
        self._w_mm = QDoubleSpinBox(); self._w_mm.setRange(20, 400); self._w_mm.setValue(self.style.width_mm)
        self._w_mm.valueChanged.connect(self._set_w)
        self._h_mm = QDoubleSpinBox(); self._h_mm.setRange(20, 400); self._h_mm.setValue(self.style.height_mm)
        self._h_mm.valueChanged.connect(self._set_h)
        self._dpi = QSpinBox(); self._dpi.setRange(50, 1200); self._dpi.setValue(self.style.dpi)
        self._dpi.valueChanged.connect(self._set_dpi)
        self._edge_color = QComboBox(); self._edge_color.addItems(["(default)", "black", "white", "gray", "red", "blue"])
        self._edge_color.currentTextChanged.connect(self._set_edge_color)
        self._edge_w = self._auto_spin(hi=8.0); self._edge_w.valueChanged.connect(self._set_edge_w)
        self._color = QComboBox(); self._color.addItems(["(auto)", "black", "red", "blue", "green", "orange", "purple"])
        self._color.currentTextChanged.connect(self._set_color)
        self._cmap = QComboBox()
        self._cmap.addItems(["(none)", "viridis", "plasma", "inferno", "magma", "cividis", "coolwarm"])
        self._cmap.currentTextChanged.connect(self._set_cmap)
        self._cmap_rev = QCheckBox("reverse"); self._cmap_rev.toggled.connect(self._set_cmap_rev)
        self._reset_btn = QPushButton("Reset styling"); self._reset_btn.clicked.connect(self._reset_styling)
        for label, w in (("Marker", self._marker), ("Font family", self._family), ("Font pt", self._font),
                         ("Marker size", self._msize), ("Line width", self._lwidth),
                         ("Label size", self._label_sz), ("Title size", self._title_sz),
                         ("Tick size", self._tick_sz), ("Legend size", self._legend_sz),
                         ("Width mm", self._w_mm), ("Height mm", self._h_mm), ("DPI", self._dpi),
                         ("Edge colour", self._edge_color), ("Edge width", self._edge_w),
                         ("Colour", self._color), ("Colormap", self._cmap)):
            form.addRow(label, w)
        form.addRow("", self._cmap_rev)
        form.addRow("", self._reset_btn)
        return box

    def _build_journal_frame(self):
        """Collapsible group exposing the 15 PQ-1 GlobalStyle journal knobs (global)."""
        box = QGroupBox("Journal frame")
        box.setCheckable(True)
        box.setChecked(False)                       # collapsed by default
        outer = QVBoxLayout(box)
        self._journal_content = QWidget()
        form = QFormLayout(self._journal_content)
        s = self.style

        def _hdr(text):
            lbl = QLabel(text)
            lbl.setStyleSheet("font-weight:bold; color:#666;")
            form.addRow(lbl)

        # --- Frame & ticks ---
        _hdr("Frame & ticks")
        self._spine_w = self._auto_spin(hi=8.0)
        self._spine_w.setValue(s.spine_width if s.spine_width is not None else 0.0)
        self._spine_w.valueChanged.connect(self._set_spine_width)
        self._tick_dir = QComboBox(); self._tick_dir.addItems(["in", "out", "inout"])
        self._tick_dir.setCurrentText(s.tick_direction)
        self._tick_dir.currentTextChanged.connect(self._set_tick_direction)
        self._minor_ticks = QCheckBox("minor ticks"); self._minor_ticks.setChecked(s.minor_ticks)
        self._minor_ticks.toggled.connect(self._set_minor_ticks)
        self._ticks_top = QCheckBox("ticks top"); self._ticks_top.setChecked(s.ticks_top)
        self._ticks_top.toggled.connect(self._set_ticks_top)
        self._ticks_right = QCheckBox("ticks right"); self._ticks_right.setChecked(s.ticks_right)
        self._ticks_right.toggled.connect(self._set_ticks_right)
        form.addRow("Spine width", self._spine_w)
        form.addRow("Tick dir", self._tick_dir)
        form.addRow("", self._minor_ticks)
        form.addRow("", self._ticks_top)
        form.addRow("", self._ticks_right)

        # --- Grid ---
        _hdr("Grid")
        self._grid = QCheckBox("show grid"); self._grid.setChecked(s.grid)
        self._grid.toggled.connect(self._set_grid)
        self._grid_style = QComboBox(); self._grid_style.addItems(["-", "--", "-.", ":"])
        self._grid_style.setCurrentText(s.grid_style)
        self._grid_style.currentTextChanged.connect(self._set_grid_style)
        self._grid_alpha = QDoubleSpinBox(); self._grid_alpha.setRange(0.0, 1.0)
        self._grid_alpha.setSingleStep(0.1); self._grid_alpha.setValue(s.grid_alpha)
        self._grid_alpha.valueChanged.connect(self._set_grid_alpha)
        form.addRow("", self._grid)
        form.addRow("Grid style", self._grid_style)
        form.addRow("Grid alpha", self._grid_alpha)

        # --- Legend ---
        _hdr("Legend")
        self._legend_on = QCheckBox("show legend"); self._legend_on.setChecked(s.legend_on)
        self._legend_on.toggled.connect(self._set_legend_on)
        # three modes + the nine explicit matplotlib positions (spec.LegendLoc; KNOWN-ISSUES 4:
        # a user who can see the right spot must be able to say "upper left", not just "inside")
        self._legend_loc = QComboBox()
        self._legend_loc.addItems(["best", "inside", "outside",
                                   "upper right", "upper left", "lower right", "lower left",
                                   "upper center", "lower center",
                                   "center left", "center right", "center"])
        self._legend_loc.setCurrentText(s.legend_loc)
        self._legend_loc.currentTextChanged.connect(self._set_legend_loc)
        self._legend_frame = QCheckBox("legend frame"); self._legend_frame.setChecked(s.legend_frame)
        self._legend_frame.toggled.connect(self._set_legend_frame)
        form.addRow("", self._legend_on)
        form.addRow("Legend loc", self._legend_loc)
        form.addRow("", self._legend_frame)

        # --- Lines & fit ---
        _hdr("Lines & fit")
        self._connect_lines = QCheckBox("connect points"); self._connect_lines.setChecked(s.connect_lines)
        self._connect_lines.toggled.connect(self._set_connect_lines)
        self._fit_color = QComboBox()
        self._fit_color.addItems(["(auto)", "black", "red", "blue", "green", "orange", "purple"])
        self._fit_color.setCurrentText(s.fit_color or "(auto)")
        self._fit_color.currentTextChanged.connect(self._set_fit_color)
        self._fit_linestyle = QComboBox(); self._fit_linestyle.addItems(["-", "--", "-.", ":"])
        self._fit_linestyle.setCurrentText(s.fit_linestyle)
        self._fit_linestyle.currentTextChanged.connect(self._set_fit_linestyle)
        form.addRow("", self._connect_lines)
        form.addRow("Fit colour", self._fit_color)
        form.addRow("Fit style", self._fit_linestyle)

        # --- Numbers ---
        _hdr("Numbers")
        self._thousands = QCheckBox("thousands separator"); self._thousands.setChecked(s.thousands_sep)
        self._thousands.toggled.connect(self._set_thousands)
        form.addRow("", self._thousands)

        outer.addWidget(self._journal_content)
        self._journal_content.setVisible(False)
        box.toggled.connect(self._journal_content.setVisible)
        self._journal_box = box
        return box

    def set_marker(self, m: str):
        self.style.marker = m
        # sync combo without re-triggering signal
        if self._marker.currentText() != m:
            self._marker.blockSignals(True)
            self._marker.setCurrentText(m)
            self._marker.blockSignals(False)
        self.style_changed.emit(self.style)

    def _emit_style(self): self.style_changed.emit(self.style)
    def _set_font(self, v): self.style.font_pt = float(v); self._emit_style()
    def _set_msize(self, v): self.style.marker_size = float(v); self._emit_style()
    def _set_lwidth(self, v): self.style.line_width = float(v); self._emit_style()
    def _set_family(self, t): self.style.font_family = None if t == "(default)" else t; self._emit_style()
    def _set_label_sz(self, v): self.style.label_size = v if v > 0 else None; self._emit_style()
    def _set_title_sz(self, v): self.style.title_size = v if v > 0 else None; self._emit_style()
    def _set_tick_sz(self, v): self.style.tick_size = v if v > 0 else None; self._emit_style()
    def _set_legend_sz(self, v): self.style.legend_size = v if v > 0 else None; self._emit_style()
    def _set_w(self, v): self.style.width_mm = float(v); self._emit_style()
    def _set_h(self, v): self.style.height_mm = float(v); self._emit_style()
    def _set_dpi(self, v): self.style.dpi = int(v); self._emit_style()
    def _set_edge_color(self, t): self.style.edge_color = None if t == "(default)" else t; self._emit_style()
    def _set_edge_w(self, v): self.style.edge_width = v if v > 0 else None; self._emit_style()
    def _set_color(self, t): self.style.color = None if t == "(auto)" else t; self._emit_style()
    def _set_cmap(self, t): self.style.colormap = None if t == "(none)" else t; self._emit_style()
    def _set_cmap_rev(self, on): self.style.colormap_reverse = bool(on); self._emit_style()
    def _set_spine_width(self, v): self.style.spine_width = v if v > 0 else None; self._emit_style()
    def _set_tick_direction(self, t): self.style.tick_direction = t; self._emit_style()
    def _set_minor_ticks(self, on): self.style.minor_ticks = bool(on); self._emit_style()
    def _set_ticks_top(self, on): self.style.ticks_top = bool(on); self._emit_style()
    def _set_ticks_right(self, on): self.style.ticks_right = bool(on); self._emit_style()
    def _set_grid(self, on): self.style.grid = bool(on); self._emit_style()
    def _set_grid_style(self, t): self.style.grid_style = t; self._emit_style()
    def _set_grid_alpha(self, v): self.style.grid_alpha = float(v); self._emit_style()
    def _set_legend_on(self, on): self.style.legend_on = bool(on); self._emit_style()
    def _set_legend_loc(self, t): self.style.legend_loc = t; self._emit_style()
    def _set_legend_frame(self, on): self.style.legend_frame = bool(on); self._emit_style()
    def _set_connect_lines(self, on): self.style.connect_lines = bool(on); self._emit_style()
    def _set_fit_color(self, t): self.style.fit_color = None if t == "(auto)" else t; self._emit_style()
    def _set_fit_linestyle(self, t): self.style.fit_linestyle = t; self._emit_style()
    def _set_thousands(self, on): self.style.thousands_sep = bool(on); self._emit_style()

    def _reset_styling(self):
        self.style = GlobalStyle()
        self._sync_styling_widgets()
        self._emit_style()

    def _sync_styling_widgets(self):
        s = self.style
        for w, val in ((self._marker, s.marker), (self._family, s.font_family or "(default)"),
                       (self._edge_color, s.edge_color or "(default)"), (self._color, s.color or "(auto)"),
                       (self._cmap, s.colormap or "(none)")):
            w.blockSignals(True); w.setCurrentText(val); w.blockSignals(False)
        for w, val in ((self._font, s.font_pt), (self._msize, s.marker_size), (self._lwidth, s.line_width),
                       (self._label_sz, s.label_size or 0.0), (self._title_sz, s.title_size or 0.0),
                       (self._tick_sz, s.tick_size or 0.0), (self._legend_sz, s.legend_size or 0.0),
                       (self._w_mm, s.width_mm), (self._h_mm, s.height_mm), (self._edge_w, s.edge_width or 0.0)):
            w.blockSignals(True); w.setValue(val); w.blockSignals(False)
        self._dpi.blockSignals(True); self._dpi.setValue(s.dpi); self._dpi.blockSignals(False)
        self._cmap_rev.blockSignals(True); self._cmap_rev.setChecked(s.colormap_reverse); self._cmap_rev.blockSignals(False)
        # journal-frame combos
        for w, val in ((self._tick_dir, s.tick_direction), (self._grid_style, s.grid_style),
                       (self._legend_loc, s.legend_loc), (self._fit_color, s.fit_color or "(auto)"),
                       (self._fit_linestyle, s.fit_linestyle)):
            w.blockSignals(True); w.setCurrentText(val); w.blockSignals(False)
        # journal-frame spins
        for w, val in ((self._spine_w, s.spine_width or 0.0), (self._grid_alpha, s.grid_alpha)):
            w.blockSignals(True); w.setValue(val); w.blockSignals(False)
        # journal-frame checkboxes
        for w, val in ((self._minor_ticks, s.minor_ticks), (self._ticks_top, s.ticks_top),
                       (self._ticks_right, s.ticks_right), (self._grid, s.grid),
                       (self._legend_on, s.legend_on), (self._legend_frame, s.legend_frame),
                       (self._connect_lines, s.connect_lines), (self._thousands, s.thousands_sep)):
            w.blockSignals(True); w.setChecked(val); w.blockSignals(False)

    # ------------------------------------------------------------------ plots section

    def set_result(self, result, probe: str):
        """Rebuild the Plots checkboxes for the backed kinds of this result/probe."""
        self._result = result
        self._probe = probe

        # clear existing checkboxes: hide + orphan NOW (bare deleteLater leaves them parented
        # until the event loop runs, painting stacked over the new rows — owner-visible garble)
        while self._plots_lay.count():
            w = self._plots_lay.takeAt(0).widget()
            if w:
                w.hide(); w.setParent(None); w.deleteLater()
        self._boxes = {}

        for k in self._registry.plot_kinds_for(probe):
            if not k.series(result):
                continue                          # capability-gating: skip unbacked kinds
            cb = QCheckBox(k.label)
            cb.setChecked(True)
            cb.toggled.connect(lambda on, key=k.key: self._on_checkbox_toggled(key, on))
            self._plots_lay.addWidget(cb)
            self._boxes[k.key] = cb

        self._emit_layout()

    def _on_checkbox_toggled(self, key: str, on: bool):
        if self._busy:
            return
        self._emit_layout()

    def enabled_kinds(self) -> list[str]:
        return [key for key, cb in self._boxes.items() if cb.isChecked()]

    def set_kind_enabled(self, key: str, on: bool):
        """Programmatically toggle a plot kind on/off."""
        if self._busy:
            return
        self._busy = True
        try:
            if key in self._boxes:
                self._boxes[key].setChecked(on)
        finally:
            self._busy = False
        self._emit_layout()

    def set_enabled_set(self, keys) -> None:
        """Check exactly `keys` (intersected with existing boxes); emits no signal (signal-safe load)."""
        want = set(keys)
        self._busy = True
        try:
            for k, cb in self._boxes.items():
                cb.setChecked(k in want)
        finally:
            self._busy = False

    def set_style(self, style) -> None:
        """Adopt an external GlobalStyle and sync the styling widgets (no emit)."""
        self.style = style
        self._sync_styling_widgets()

    def current_layout(self) -> PlotLayout:
        return PlotLayout(plots=[PlotEntry(kind=k) for k in self._boxes if self._boxes[k].isChecked()])

    def _emit_layout(self):
        self.layout_changed.emit(self.current_layout())
