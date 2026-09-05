"""KNOWN-ISSUES #17 — the "Colour…" button must be visible at the default window size.

The left pane opened at a fixed 300 px while its content's minimum width was ~392 px
(the preset bar's four buttons carried a style minimum of 80 px each against their own
64 px maximum), and the scroll area hid its horizontal scrollbar — so the third file-row
button (and the preset row's tail) was clipped with no way to reach it.
"""


def _shown_window(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.show()                      # DEFAULT size — the reproduction condition
    qapp.processEvents()
    win.load_path(vsm_path)
    qapp.processEvents()
    return win, win.tabs.currentWidget()


def _colour_btn(tab):
    from PySide6.QtWidgets import QPushButton
    return next(b for b in tab.file_manager.findChildren(QPushButton)
                if b.text() == "Colour…")


def test_colour_button_fully_inside_the_viewport_at_default_size(qapp, vsm_path):
    win, tab = _shown_window(qapp, vsm_path)
    vp = tab._left_scroll.viewport()
    btn = _colour_btn(tab)
    right = btn.mapTo(vp, btn.rect().topRight()).x()
    assert right <= vp.width(), (right, vp.width())


def test_every_left_panel_button_reachable_or_scrollable(qapp, vsm_path):
    # backstop: even if some future control overflows, the scroll area must be able to
    # show a horizontal scrollbar rather than silently clipping (was ScrollBarAlwaysOff)
    from PySide6.QtCore import Qt
    win, tab = _shown_window(qapp, vsm_path)
    assert tab._left_scroll.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded


def test_preset_buttons_min_does_not_exceed_their_max(qapp, vsm_path):
    win, tab = _shown_window(qapp, vsm_path)
    for b in tab.preset_bar._btns.values():
        assert b.minimumWidth() <= b.maximumWidth() != 0
        assert b.minimumSizeHint().width() > 0
