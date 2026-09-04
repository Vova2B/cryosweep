from __future__ import annotations
import math
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt

_COLORS = {"ok": "#1b5e20", "gated": "#e65100", "low_confidence": "#e65100", "error": "#b71c1c"}

def _fmt_conf(c) -> str:
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return "—"
    return f"{c:.3f}"

class StatusBanner(QLabel):
    def __init__(self):
        super().__init__("")
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def show_result(self, result, notes=()) -> None:
        parts = [f"status: {result.status}", f"confidence: {_fmt_conf(result.confidence)}"]
        for g in (result.gate or []):
            remedy = " ".join(f"{k}={v}" for k, v in (g.remedy or {}).items())
            parts.append(f"gated[{g.need}]: {g.reason} → {remedy}")
        for w in (result.warnings or []):
            parts.append(f"warning: {w}")
        for e in (result.errors or []):
            parts.append(f"error: {e}")
        for n in notes:
            parts.append(f"note: {n}")
        self.setText("   |   ".join(parts))
        color = _COLORS.get(result.status, "#333")
        self.setStyleSheet(f"color: white; background: {color}; padding: 4px;")

    def show_message(self, msg: str) -> None:
        self.setText(msg)
        self.setStyleSheet("color: #333; background: #eee; padding: 4px;")
