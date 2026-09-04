from PySide6.QtWidgets import QLabel
from cryosweep_gui.widgets import CollapsibleGroup


def test_collapsed_by_default_hides_body(qapp):
    g = CollapsibleGroup("Advanced")
    body = QLabel("x"); g.body_layout.addWidget(body)
    assert g.is_collapsed() is True
    assert body.isVisibleTo(g) is False


def test_toggle_shows_and_hides(qapp):
    g = CollapsibleGroup("Advanced")
    body = QLabel("x"); g.body_layout.addWidget(body)
    g.set_collapsed(False)
    assert g.is_collapsed() is False
    assert body.isVisibleTo(g) is True
    g.set_collapsed(True)
    assert body.isVisibleTo(g) is False


def test_expanded_construction(qapp):
    g = CollapsibleGroup("Common", collapsed=False)
    assert g.is_collapsed() is False
