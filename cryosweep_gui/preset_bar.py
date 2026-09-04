from __future__ import annotations
import pathlib
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QComboBox, QPushButton, QInputDialog, QFileDialog)
from cryosweep_core.plotting.spec import PlotLayout, GlobalStyle
from cryosweep_core.plotting.presets import NamedPreset, builtin_presets

class PresetBar(QWidget):
    """Per-probe preset library bar. Inert until bind(store, tab, save_cb) is called."""
    _MARK = "★ "

    def __init__(self, probe: str):
        super().__init__()
        self.probe = probe
        self._store = None
        self._tab = None
        self._save = lambda: None
        lay = QHBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.combo = QComboBox()
        self.combo.setToolTip("Named plot-layout presets for this tab (which plots are shown).")
        self.combo.activated.connect(lambda *_: self.load(self.combo.currentText()))
        self.combo.currentIndexChanged.connect(lambda *_: self._on_combo_changed())
        lay.addWidget(self.combo, 1)
        self._btns = {}
        for txt, fn in (("Save As", self._on_save_as), ("Del", self._on_delete),
                        ("Import", self._on_import), ("Export", self._on_export)):
            b = QPushButton(txt); b.clicked.connect(fn); b.setMaximumWidth(64)
            lay.addWidget(b); self._btns[txt] = b
        self._set_enabled(False)

    def bind(self, store, tab, save_cb):
        self._store = store; self._tab = tab; self._save = save_cb
        self.refresh()

    def _set_enabled(self, on):
        for b in self._btns.values():
            b.setEnabled(on)
        self.combo.setEnabled(on)

    def _is_builtin_label(self, text):
        return text.startswith(self._MARK)

    def _strip(self, text):
        return text[len(self._MARK):] if text.startswith(self._MARK) else text

    def refresh(self):
        ready = self._store is not None
        self._set_enabled(ready)
        if not ready:
            return
        self.combo.blockSignals(True); self.combo.clear()
        for bp in builtin_presets(self.probe):
            self.combo.addItem(self._MARK + bp.name)     # built-ins always listed (marked)
        self.combo.addItems([p.name for p in self._store.presets if p.probe == self.probe])
        self.combo.blockSignals(False)
        has_layout = getattr(self._tab, "_layout_state", None) is not None
        self._btns["Save As"].setEnabled(has_layout)
        self._btns["Export"].setEnabled(has_layout or self.combo.count() > 0)
        self._on_combo_changed()

    def _on_combo_changed(self):
        self._btns["Del"].setEnabled(
            self.combo.count() > 0 and not self._is_builtin_label(self.combo.currentText()))

    def save_as(self, name: str):
        name = self._strip((name or "").strip())
        if not name or self._store is None or self._tab._layout_state is None:
            return
        self._store.presets = [p for p in self._store.presets
                               if not (p.name == name and p.probe == self.probe)]
        self._store.presets.append(NamedPreset(name=name, probe=self.probe,
                                               layout=self._tab._layout_state.model_copy(deep=True)))
        self._save(); self.refresh(); self.combo.setCurrentText(name)

    def load(self, name: str):
        if self._store is None or not name:
            return
        raw = self._strip(name)
        merged = {p.name: p for p in builtin_presets(self.probe)}
        for p in self._store.presets:
            if p.probe == self.probe:
                merged[p.name] = p
        preset = merged.get(raw)
        if preset is None or self._tab._last_result is None:
            return
        self._tab.show_result(self._tab._last_result, restore_layout=preset.layout,
                              restore_exact=True)   # a named preset's kind subset is deliberate
        self._tab._persist_last_used()

    def delete(self, name: str):
        if self._store is None or self._is_builtin_label(name):
            return
        raw = self._strip(name)
        self._store.presets = [p for p in self._store.presets
                               if not (p.name == raw and p.probe == self.probe)]
        self._save(); self.refresh()

    def export_to(self, path: str):
        if self._store is None:
            return
        p = pathlib.Path(path)
        if p.suffix == "":
            p = p.with_suffix(".json")
        sel = self.combo.currentText()
        preset = next((x for x in self._store.presets
                       if x.name == sel and x.probe == self.probe), None)
        layout = preset.layout if preset is not None else self._tab._layout_state
        if layout is None:
            return
        p.write_text(layout.model_dump_json(indent=2))
        p.with_suffix(".style.json").write_text(self._store.global_style.model_dump_json(indent=2))

    def import_from(self, path: str, name: str):
        if self._store is None:
            return
        try:
            layout = PlotLayout.model_validate_json(pathlib.Path(path).read_text())
        except Exception:
            return False
        name = (name or "").strip()
        if not name:
            return False
        self._store.presets = [p for p in self._store.presets
                               if not (p.name == name and p.probe == self.probe)]
        self._store.presets.append(NamedPreset(name=name, probe=self.probe, layout=layout))
        # restore the style sidecar written by export_to, if present
        style_path = pathlib.Path(path).with_suffix(".style.json")
        if style_path.exists():
            try:
                style = GlobalStyle.model_validate_json(style_path.read_text())
            except Exception:
                style = None
            if style is not None:
                self._store.global_style = style
                if self._tab is not None and getattr(self._tab, "controls", None) is not None:
                    self._tab.controls.set_style(style)        # syncs widgets, no emit
        self._save(); self.refresh()
        return True

    def _on_save_as(self):
        name, ok = QInputDialog.getText(self, "Save preset", "Name:")
        if ok:
            self.save_as(name)

    def _on_delete(self):
        self.delete(self.combo.currentText())

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export preset", f"{self.probe}_preset.json")
        if path:
            self.export_to(path)

    def _on_import(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import preset", "", "Preset (*.json);;All files (*)")
        if not path:
            return
        name, ok = QInputDialog.getText(self, "Import preset", "Name:")
        if ok:
            self.import_from(path, name)
