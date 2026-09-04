from __future__ import annotations
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
                             QPushButton, QColorDialog, QFileDialog, QMessageBox)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor

class FileManager(QWidget):
    """Per-tab file list: add/remove, per-file include checkbox + colour. Drives the tab's overlay."""
    changed = Signal()                       # files added/removed/toggled/recoloured -> re-render

    def __init__(self, tab):
        super().__init__()
        self._tab = tab
        self._refreshing = False
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget(); self.list.setMaximumHeight(120)
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentRowChanged.connect(self._on_row_changed)
        lay.addWidget(self.list)
        bar = QHBoxLayout()
        for txt, fn in (("Add to compare…", self._on_add), ("Remove", self.remove_current),
                        ("Colour…", self._on_colour)):
            b = QPushButton(txt); b.setMaximumWidth(120); b.clicked.connect(fn); bar.addWidget(b)
            if txt == "Add to compare…":
                b.setToolTip(f"Overlay another {self._tab.probe} file on this tab for comparison. "
                             "To open a different measurement type, use Load .dat at the top.")
        lay.addLayout(bar)

    def refresh(self):
        self._refreshing = True
        self.list.blockSignals(True)
        self.list.clear()
        for i, e in enumerate(self._tab._files):
            it = QListWidgetItem(e.label)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if e.include else Qt.CheckState.Unchecked)
            if e.colour:
                it.setForeground(QColor(e.colour))
            self.list.addItem(it)
        if self.list.count() > 0:                                  # highlight the tab's focused entry (single source of truth)
            self.list.setCurrentRow(min(max(self._tab._focus, 0), self.list.count() - 1))
        self.list.blockSignals(False)
        self._refreshing = False

    def _on_row_changed(self, idx):
        if self._refreshing or idx < 0:
            return
        self._tab.focus_file(idx)

    def _on_item_changed(self, it):
        idx = self.list.row(it)
        self._tab.set_include(idx, it.checkState() == Qt.CheckState.Checked)
        self.changed.emit()

    def _on_add(self):
        path, _ = QFileDialog.getOpenFileName(self, "Add file", "", "PPMS data (*.dat);;All files (*)")
        if path:
            self._guarded_add(path)

    def _mw(self):
        return getattr(self._tab, "_mw", None)

    def _guarded_add(self, path):
        mw = self._mw()
        if mw is None:
            self._tab.add_overlay_path(path); return
        try:
            score, key, applicable = mw.detect_probe_for(path)
        except Exception:
            self._tab.add_overlay_path(path); return
        if score >= 0.5 and key and self._tab.probe not in applicable:
            choice = self._ask_mismatch(key, self._tab.probe)
            if choice == "load":
                mw.load_path(path); return
            if choice == "cancel":
                return
            # "add" falls through to overlay
        self._tab.add_overlay_path(path)

    def _ask_mismatch(self, detected, tabname):
        """Modal 3-way. Returns 'load' | 'add' | 'cancel'. Split out so tests stub the decision."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Different measurement type")
        box.setText(f"This file looks like {detected} data, but this is the {tabname} tab. "
                    f"Add it anyway (it will be analyzed as {tabname} and will likely fail), "
                    "or load it as a new primary file?")
        load_b = box.addButton(f"Load as {detected}", QMessageBox.ButtonRole.AcceptRole)
        add_b = box.addButton("Add anyway", QMessageBox.ButtonRole.DestructiveRole)
        box.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        box.setDefaultButton(load_b)
        box.exec()
        clicked = box.clickedButton()
        if clicked is load_b:
            return "load"
        if clicked is add_b:
            return "add"
        return "cancel"

    def remove_current(self):
        idx = self.list.currentRow()
        if idx >= 0:
            self._tab.remove_file(idx)        # commits + pops + clamps _focus
            self.refresh()                    # highlights the clamped _focus
            self._tab.sync_panel_to_focus()   # panel shows the survivor at _focus (no re-commit)
            self.changed.emit()

    def _on_colour(self):
        idx = self.list.currentRow()
        if idx < 0:
            return
        col = QColorDialog.getColor()
        if col.isValid():
            self._tab.set_colour(idx, col.name()); self.refresh(); self.changed.emit()
