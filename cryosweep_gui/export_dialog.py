"""Batch plot-export dialog (PQ-1 2b): v1 SavePlotsDialog parity + PDF/SVG + exact-mm.

Thin shell: assembles arguments for cryosweep_core.plotting.export.export_plots —
zero savefig logic lives here.
"""
from __future__ import annotations

import pathlib

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                             QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox,
                             QVBoxLayout, QWidget)

_FORMATS = ("png", "pdf", "svg")


class ExportPlotsDialog(QDialog):
    """Select plots + formats + output location; assemble() returns the export args."""

    def __init__(self, entries, style, default_dir, default_prefix, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export plots")
        root = QVBoxLayout(self)

        plots_box = QGroupBox("Plots")
        pb = QVBoxLayout(plots_box)
        self.plot_checks: dict[str, QCheckBox] = {}
        self._first_kind = entries[0][0] if entries else ""
        for kind, label in entries:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(self._sync)
            self.plot_checks[kind] = cb
            pb.addWidget(cb)
        row = QHBoxLayout()
        self.all_btn = QPushButton("All"); self.none_btn = QPushButton("None")
        self.all_btn.clicked.connect(lambda: self._set_all(True))
        self.none_btn.clicked.connect(lambda: self._set_all(False))
        row.addWidget(self.all_btn); row.addWidget(self.none_btn); row.addStretch(1)
        rw = QWidget(); rw.setLayout(row); pb.addWidget(rw)
        root.addWidget(plots_box)

        out_box = QGroupBox("Output")
        form = QFormLayout(out_box)
        fmt_row = QHBoxLayout()
        self.fmt_checks: dict[str, QCheckBox] = {}
        for f in _FORMATS:
            cb = QCheckBox(f.upper())
            cb.setChecked(f == "png")
            cb.toggled.connect(self._sync)
            self.fmt_checks[f] = cb
            fmt_row.addWidget(cb)
        fw = QWidget(); fw.setLayout(fmt_row)
        form.addRow("Formats", fw)
        self.dpi_spin = QSpinBox(); self.dpi_spin.setRange(50, 1200)
        self.dpi_spin.setValue(style.dpi)
        form.addRow("DPI (PNG)", self.dpi_spin)
        self.tight_cb = QCheckBox("Tight crop (overrides exact mm size)")
        form.addRow("", self.tight_cb)
        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit(str(default_dir))
        self.browse_btn = QPushButton("…"); self.browse_btn.setFixedWidth(28)
        self.browse_btn.clicked.connect(self._browse)
        dir_row.addWidget(self.dir_edit); dir_row.addWidget(self.browse_btn)
        dw = QWidget(); dw.setLayout(dir_row)
        form.addRow("Directory", dw)
        self.prefix_edit = QLineEdit(default_prefix)
        self.prefix_edit.textChanged.connect(self._sync)
        form.addRow("Prefix", self.prefix_edit)
        self.example_label = QLabel("")
        self.example_label.setStyleSheet("color:#777;")
        form.addRow("", self.example_label)
        root.addWidget(out_box)

        buttons = QDialogButtonBox()
        self.export_btn = buttons.addButton("Export", QDialogButtonBox.ButtonRole.AcceptRole)
        self.export_btn.setDefault(True)
        buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)
        self._sync()

    # ---- helpers ----

    def _set_all(self, on: bool):
        for cb in self.plot_checks.values():
            cb.setChecked(on)

    def _browse(self):
        d = QFileDialog.getExistingDirectory(self, "Export directory", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def _sync(self, *_):
        kinds = [k for k, cb in self.plot_checks.items() if cb.isChecked()]
        formats = [f for f in _FORMATS if self.fmt_checks[f].isChecked()]
        self.export_btn.setEnabled(bool(kinds) and bool(formats))
        self.dpi_spin.setEnabled(self.fmt_checks["png"].isChecked())
        ex_kind = kinds[0] if kinds else self._first_kind
        ex_fmt = formats[0] if formats else "png"
        self.example_label.setText(f"e.g. {self.prefix_edit.text()}_{ex_kind}.{ex_fmt}")

    def assemble(self) -> dict:
        return {"kinds": [k for k, cb in self.plot_checks.items() if cb.isChecked()],
                "formats": [f for f in _FORMATS if self.fmt_checks[f].isChecked()],
                "dpi": self.dpi_spin.value(),
                "tight": self.tight_cb.isChecked(),
                "out_dir": pathlib.Path(self.dir_edit.text()),
                "prefix": self.prefix_edit.text()}
