from __future__ import annotations
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
                             QPushButton, QLabel, QComboBox, QFileDialog)
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry
from cryosweep_core.discovery import discover
from cryosweep_core.plotting.presets import load_store, save_store
from cryosweep_gui.state import AnalysisState
from cryosweep_gui.status_banner import StatusBanner
from cryosweep_gui.probe_tab import ProbeTab
from cryosweep_gui.inputs.base import build_panel
import cryosweep_gui.inputs   # noqa: F401  (registers all input panels via inputs/__init__)
import cryosweep_gui.presets_io as presets_io

_LABELS = {"vsm": "Magnetization", "resistivity": "Resistivity",
           "heatcapacity": "Heat Capacity", "hall": "Hall",
           "hall_tdep": "Temp-Dep Hall", "acms": "AC Susceptibility",
           "tto": "Thermal Transport"}


def _dot_icon(color="#2e7d32", size=10):
    """Small filled-circle QIcon for the 'this tab holds data for the loaded file' marker
    (an icon renders cleanly offscreen and on macOS, unlike per-tab rich-text labels)."""
    pm = QPixmap(size, size)
    pm.fill(Qt.GlobalColor.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.setBrush(QColor(color))
    p.setPen(Qt.PenStyle.NoPen)
    p.drawEllipse(0, 0, size, size)
    p.end()
    return QIcon(pm)

class MainWindow(QMainWindow):
    def __init__(self, preset_path=None):
        super().__init__()
        self.setWindowTitle("CryoSweep")
        self.setMinimumSize(1100, 650)
        self.state = AnalysisState()
        self.registry = build_default_registry()
        self._needs = {p["key"]: p.get("needs", []) for p in discover(self.registry)["probes"]}
        # Migration read (Task 6): gate on `preset_path is None`, NOT on "the new file is
        # absent". One `self.preset_path` serves both load_store and save_store (:247/:259),
        # so returning the legacy path here would also WRITE BACK to it. Load from legacy,
        # but keep self.preset_path on the NEW path so the first save lands there.
        if preset_path is None:
            self.preset_path = presets_io.default_store_path()
            source = self.preset_path
            if not source.exists():
                legacy = presets_io.legacy_store_path()   # module attr: monkeypatchable
                if legacy.exists():
                    source = legacy
        else:
            self.preset_path = preset_path
            source = self.preset_path
        self.preset_store = load_store(source)
        self._save_timer = QTimer(self); self._save_timer.setSingleShot(True)
        self._save_timer.timeout.connect(self._on_save_timer)

        central = QWidget(); root = QVBoxLayout(central)

        # ---- file bar ----
        bar = QHBoxLayout()
        self.load_btn = QPushButton("Load .dat")
        self.load_btn.setObjectName("primaryLoad")
        self.load_btn.setStyleSheet("QPushButton#primaryLoad { font-weight: bold; padding: 4px 12px; }")
        self.load_btn.setToolTip("Load a .dat file as the primary measurement (auto-detects probe).")
        self.path_label = QLabel("(no file)")
        self.chip = QLabel("")
        bar.addWidget(self.load_btn); bar.addWidget(self.path_label, 1); bar.addWidget(self.chip)
        bar.addWidget(QLabel("Unit:"))
        self.unit_combo = QComboBox(); self.unit_combo.addItems(["CGS", "SI"])
        bar.addWidget(self.unit_combo)
        bar.addWidget(QLabel("Field:"))
        self.field_unit_combo = QComboBox(); self.field_unit_combo.addItems(["Oe", "T"])
        self.field_unit_combo.setCurrentText(self.preset_store.global_style.field_unit)
        self.field_unit_combo.setToolTip("Display unit for field legends/axes (storage stays Oe)")
        bar.addWidget(self.field_unit_combo)
        root.addLayout(bar)

        # ---- tabs (one per registry probe; VSM bespoke, others generic) ----
        self.tabs = QTabWidget()
        self._tab_dot = _dot_icon()
        for key in self.registry.analyzer_keys():
            panel = build_panel(key, needs=self._needs.get(key, []))
            tab = ProbeTab(probe=key, panel=panel, registry=self.registry,
                           get_raw=self.state.get_raw, get_unit=lambda: self.unit_combo.currentText())
            tab.bind_window(self)                                   # inject store + _mw
            tab.controls.style_changed.connect(self._on_style_changed)
            self.tabs.addTab(tab, _LABELS.get(key, key))
        root.addWidget(self.tabs, 1)
        self._link_hall_panels()                                 # same sample: type thickness once

        # ---- shared status banner ----
        self.banner = StatusBanner()
        root.addWidget(self.banner)
        for i in range(self.tabs.count()):
            self.tabs.widget(i).banner = self.banner

        self.setCentralWidget(central)
        self.load_btn.clicked.connect(self._on_load)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.unit_combo.currentTextChanged.connect(self._reanalyze_active)
        self.field_unit_combo.currentTextChanged.connect(self._on_field_unit_changed)

    def _link_hall_panels(self):
        """Mirror sample inputs between the hall and hall_tdep panels (same sample, same file):
        thickness/channels via textEdited (user-only, setText doesn't re-emit -> no loops),
        combos via currentTextChanged with the sibling updated under blockSignals so the
        mirror never triggers the sibling's refit — it re-analyzes when its tab is shown.
        Deliberately NOT mirrored: the longitudinal FILE (a per-tab mobility source choice;
        channels/thickness/sign describe the sample and are shared)."""
        panels = [self.tabs.widget(i).panel for i in range(self.tabs.count())
                  if self.tabs.widget(i).probe in ("hall", "hall_tdep")]
        panels = [p for p in panels if hasattr(p, "thickness_edit")]

        def _set_combo(combo, text):
            combo.blockSignals(True)
            combo.setCurrentText(text)
            combo.blockSignals(False)

        for a in panels:
            for b in panels:
                if a is b:
                    continue
                a.hall_channel_edit.textEdited.connect(b.hall_channel_edit.setText)
                a.thickness_edit.textEdited.connect(b.thickness_edit.setText)
                a.long_channel_edit.textEdited.connect(b.long_channel_edit.setText)
                a.thickness_unit.currentTextChanged.connect(
                    lambda t, c=b.thickness_unit: _set_combo(c, t))
                a.geometry_sign.currentTextChanged.connect(
                    lambda t, c=b.geometry_sign: _set_combo(c, t))

    def detect_probe_for(self, path):
        """Detect (score, key, applicable_set) for a file without loading it into any tab.
        applicable_set reuses _hall_autopopulate so a resistivity file counts as usable on
        hall/hall_tdep."""
        rt = load_dat(str(path))
        df, cmap = canonicalize_columns(rt.df, rt.header)
        score, key = detect_probe(rt.header, set(df.columns), self.registry)
        applicable, _det, _note, _long = self._hall_autopopulate(df, cmap, score, key)
        return score, key, applicable

    # ---- programmatic API (used by tests + the Load button) ----
    def load_path(self, path) -> None:
        self.state.load(path)
        self.path_label.setText(str(path))
        rt = self.state.get_raw()
        df, cmap = canonicalize_columns(rt.df, rt.header)   # imported at top (main_window.py:6)
        score, key = detect_probe(rt.header, set(df.columns), self.registry)
        applicable, det, note, long_ch = self._hall_autopopulate(df, cmap, score, key)
        # seed applicable tabs' file lists, clear the rest (D3 seed / D4 stale-clear)
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            tab.set_files([str(path)] if tab.probe in applicable else [])
        self._set_tab_indicators(applicable)
        # pre-fill the Hall channel into the now-seeded hall / hall_tdep tabs (D2)
        if det is not None:
            for p in ("hall", "hall_tdep"):
                if p in applicable:
                    self._prefill_hall_channel(p, det[0], long_ch)
        # chip + landing tab
        if score >= 0.5 and key:
            self.chip.setText(f"detected: {key} ({score:.2f}){note}")
            self.select_probe(key)
        else:
            self.chip.setText(f"undetected (best {key} {score:.2f}) — pick a tab")
            self._reanalyze_active()

    def _hall_autopopulate(self, df, cmap, score, key):
        """Return (applicable_probes:set, detected:(ch,frac)|None, chip_note:str, long_ch:int|None).
        Only resistivity-format files (the shared Hall/resistivity format) populate the Hall tabs."""
        applicable = {key} if (score >= 0.5 and key) else set()
        det = None
        long_ch = None
        note = ""
        if key == "resistivity" and score >= 0.5:
            from cryosweep_core.config import RunConfig
            from cryosweep_core.detect.sweeps import segment_sweeps
            from cryosweep_core.detect.hall_channel import (
                detect_hall_channel, detect_longitudinal_channel,
                hall_field_sweep_applicable, hall_tdep_applicable)
            segs = segment_sweeps(df, cmap, RunConfig())            # D5: segment once, reuse below
            if hall_field_sweep_applicable(cmap, segs):
                applicable.add("hall")
            if hall_tdep_applicable(segs):
                applicable.add("hall_tdep")
            det = detect_hall_channel(df, cmap, segs)
            if det is not None:
                long_ch = detect_longitudinal_channel(df, cmap, segs, det[0])
            labels = [_LABELS.get(p, p) for p in ("hall", "hall_tdep") if p in applicable]
            if labels:
                sfx = f" (Ch{det[0]})" if det else ""
                note = " — also usable in: " + ", ".join(l + sfx for l in labels)
                if det is None:
                    note += " (set Hall channel)"
        return applicable, det, note, long_ch

    def _set_tab_indicators(self, applicable):
        """Green dot on every tab that can analyze the currently loaded file (detection-based,
        not analyzer-run). Cleared and recomputed on every load."""
        for i in range(self.tabs.count()):
            on = self.tabs.widget(i).probe in applicable
            self.tabs.setTabIcon(i, self._tab_dot if on else QIcon())

    def _prefill_hall_channel(self, probe, channel, long_channel=None):
        """setText the auto-detected Hall (and longitudinal, when found) channel into a Hall
        tab's panel and mirror it into the focused file-entry's state (overlay-safety)."""
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if tab.probe == probe and hasattr(tab.panel, "set_hall_channel"):
                tab.panel.set_hall_channel(channel)
                if long_channel is not None:
                    tab.panel.long_channel_edit.setText(str(int(long_channel)))
                tab.commit_focused_params()
                return

    def select_probe(self, probe: str) -> None:
        for i in range(self.tabs.count()):
            if self.tabs.widget(i).probe == probe:
                if self.tabs.currentIndex() != i:
                    self.tabs.setCurrentIndex(i)   # fires currentChanged -> _reanalyze_active
                    return                         # don't analyze+render a second time
                break
        self._reanalyze_active()

    def _reanalyze_active(self, *_):
        tab = self.tabs.currentWidget()
        if tab is None:
            return
        tab.controls.set_style(self.preset_store.global_style)     # global style applied on show
        if getattr(tab, "_is_overlay", None) is not None and tab._is_overlay():
            tab.analyze_and_render()                                # ≥2 included files -> overlay from file list
            return
        if self.state.get_raw() is None:
            return
        res = tab.analyze()
        if res is not None:
            tab.show_result(res, restore_layout=self.preset_store.last_used.get(tab.probe))
            tab.absorb_result(res)               # fitted values -> boxes + focused entry state

    def _on_field_unit_changed(self, text):
        self.preset_store.global_style = self.preset_store.global_style.model_copy(
            update={"field_unit": text})
        self._reanalyze_active()                    # pushes global_style to the active tab + re-renders
        self._save_presets_soon()

    def _on_load(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load .dat", "", "PPMS data (*.dat);;All files (*)")
        if path:
            self.load_path(path)

    def _save_presets_soon(self):
        self._save_timer.start(300)

    def _on_save_timer(self):
        save_store(self.preset_store, self.preset_path)

    def _on_style_changed(self, style):
        self.preset_store.global_style = style
        self._save_presets_soon()

    def closeEvent(self, event):
        self._save_timer.stop()                                    # cancel pending debounce (no double write)
        for i in range(self.tabs.count()):
            tab = self.tabs.widget(i)
            if getattr(tab, "_layout_state", None) is not None:
                self.preset_store.last_used[tab.probe] = tab._layout_state
        save_store(self.preset_store, self.preset_path)
        for i in range(self.tabs.count()):
            self.tabs.widget(i).stop_worker()
        super().closeEvent(event)

    def _on_tab_changed(self, *_):
        self._reanalyze_active()
