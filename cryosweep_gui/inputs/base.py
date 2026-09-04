from __future__ import annotations
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel

# probe key -> InputPanel subclass; populated by inputs.vsm etc. via register_panel()
INPUT_PANELS: dict[str, type] = {}

def opt_float(text: str):
    """Parse a QLineEdit string to float, or None if blank/non-numeric (never raises)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None

def register_panel(probe_key: str, cls: type) -> None:
    INPUT_PANELS[probe_key] = cls

class InputPanel(QWidget):
    """Base: contributes nothing. Subclasses add widgets and override build_*()."""
    def __init__(self, probe_key: str):
        super().__init__()
        self.probe_key = probe_key
        self._layout = QVBoxLayout(self)

    def build_overrides(self) -> dict:
        """RunConfig.load(**overrides) fragments. Nested sub-configs as nested dicts
        (e.g. {"geometry": {...}}, {"hall": {...}})."""
        return {}

    def build_header_patch(self) -> dict:
        """RawTable.header field overrides (e.g. {"molar_mass":..., "mass_mg":...})."""
        return {}

    def get_state(self) -> dict:
        """Return a JSON-serialisable dict of the panel's current widget values."""
        return {}

    def set_state(self, state: dict) -> None:
        """Restore widget values from a dict previously returned by get_state()."""
        pass

class GenericNeedsPanel(InputPanel):
    """Fallback for probes without a bespoke panel (P1): lists the probe's needs[] as a
    read-only hint and contributes no inputs. Bespoke panels arrive in P2/P3."""
    def __init__(self, probe_key: str, needs: list[dict] | None = None):
        super().__init__(probe_key)
        self._needs = needs or []
        self._label = QLabel(self.needs_text())
        self._label.setWordWrap(True)
        self._layout.addWidget(self._label)

    def needs_text(self) -> str:
        keys = ", ".join(n.get("key", "?") for n in self._needs) or "(none)"
        return f"inputs for '{self.probe_key}' arrive in a later phase. needs: {keys}"

def build_panel(probe_key: str, needs: list[dict] | None = None) -> InputPanel:
    cls = INPUT_PANELS.get(probe_key)
    if cls is not None:
        return cls()
    return GenericNeedsPanel(probe_key, needs=needs)
