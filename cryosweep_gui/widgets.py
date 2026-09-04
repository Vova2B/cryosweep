from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QToolButton


class CollapsibleGroup(QWidget):
    """A titled, borderless-header collapsible section. Callers add rows to `.body_layout`.
    Pure view widget (no analysis coupling); mirrors the AxisStrip pattern in plot_controls.py."""

    def __init__(self, title: str, *, collapsed: bool = True):
        super().__init__()
        self._title = title
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setStyleSheet("QToolButton { border: none; font-weight: bold; }")
        self._toggle.setChecked(not collapsed)
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle)
        self._body = QWidget()
        self.body_layout = QVBoxLayout(self._body)
        self.body_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._body)
        self._body.setVisible(not collapsed)
        self._sync_text()

    def _on_toggle(self, on: bool):
        self._body.setVisible(on)
        self._sync_text()

    def _sync_text(self):
        arrow = "▾" if self._toggle.isChecked() else "▸"
        self._toggle.setText(f"{arrow} {self._title}")

    def is_collapsed(self) -> bool:
        return not self._toggle.isChecked()

    def set_collapsed(self, collapsed: bool) -> None:
        self._toggle.setChecked(not collapsed)
