from __future__ import annotations
import math
from PySide6.QtWidgets import QLabel
from PySide6.QtCore import Qt

_COLORS = {"ok": "#1b5e20", "gated": "#e65100", "low_confidence": "#e65100", "error": "#b71c1c"}

def _fmt_conf(c) -> str:
    if c is None or (isinstance(c, float) and math.isnan(c)):
        return "—"
    return f"{c:.3f}"

_HALL_PROBES = ("hall", "hall_tdep")

def _hall_confidence_note(result) -> str | None:
    """Spec Sec 4.5: confidence is min(fit quality, resolved fraction) for both Hall
    probes, and Task 6 made the parts dict the whole diagnosis -- but confidence_parts
    reached no human surface at all, so a low_confidence banner named no cause. A file
    whose `fit` term binds (poor linear R_xy-vs-B fit) and one whose `resolved` term
    binds (R_H indistinguishable from zero) are opposite problems with opposite remedies,
    and used to render an identical banner. `fit: None` is itself meaningful -- no r2
    survived the zero-degrees-of-freedom rule, so it is worded out rather than shown as a
    dash standing in for "missing".

    Confined to the two Hall probes on purpose: the parts dict's shape differs by probe
    (tto/acms report {detector, grouping}; hc/mag/resistivity report a {..., fit} triple
    with no `resolved` ceiling at all), so this phrasing does not generalise without
    rewriting six probes' on-screen text -- its own reviewable change, not a rider here."""
    if result.status != "low_confidence":
        return None
    if (result.data or {}).get("probe") not in _HALL_PROBES:
        return None
    parts = result.confidence_parts or {}
    if "resolved" not in parts:
        return None
    fit, resolved = parts.get("fit"), parts["resolved"]
    # Each ceiling carries its own label and they are separated by a semicolon. The
    # earlier phrasing put the non-binding one in a bare parenthetical, which landed
    # right after the gloss on the binding one -- two stacked parentheticals that read
    # as though both qualified the same number.
    # 2026-09-14: a None `fit` no longer defaults to 1.0 inside the analyzers -- the term
    # is dropped, the status is capped at low_confidence and the envelope carries the
    # fit_quality_unavailable flag. Say that, rather than naming one of its causes.
    fit_words = ("fit quality — no fit-quality evidence: no published point carries a "
                 "usable r², so the status is capped at low_confidence"
                 if fit is None else f"fit quality r² = {fit:.3f}")
    resolved_words = f"resolved fraction {resolved:.3f} (R_H distinguishable from zero)"
    fit_binds = fit is not None and fit <= resolved
    resolved_binds = fit is None or resolved <= fit
    if fit_binds and resolved_binds:
        return f"binding ceiling — both equally: {fit_words}; {resolved_words}"
    if resolved_binds:
        return f"binding ceiling: {resolved_words}; other ceiling: {fit_words}"
    return f"binding ceiling: {fit_words}; other ceiling: {resolved_words}"

class StatusBanner(QLabel):
    def __init__(self):
        super().__init__("")
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def show_result(self, result, notes=()) -> None:
        parts = [f"status: {result.status}", f"confidence: {_fmt_conf(result.confidence)}"]
        note_ceiling = _hall_confidence_note(result)
        if note_ceiling:
            parts.append(note_ceiling)
        for g in (result.gate or []):
            # `field` is the GUI label for this need. Phrase it -- dumping it as another
            # k=v pair would leave the on-screen diagnosis still pointing only at a CLI
            # flag, which is what made this banner unusable to a GUI user.
            remedy = dict(g.remedy or {})
            field = remedy.pop("field", None)
            tail = " ".join(f"{k}={v}" for k, v in remedy.items())
            if field:
                parts.append(f"gated[{g.need}]: {g.reason} → enter \u201c{field}\u201d "
                             f"in the panel at left ({tail})")
            else:
                parts.append(f"gated[{g.need}]: {g.reason} → {tail}")
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
