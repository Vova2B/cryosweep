from PySide6.QtWidgets import QScrollArea
from PySide6.QtCore import Qt


def _hc_tab(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    for i in range(win.tabs.count()):
        if win.tabs.widget(i).probe == "heatcapacity":
            win.tabs.setCurrentIndex(i)
            return win, win.tabs.widget(i)
    raise AssertionError("no heatcapacity tab")


def test_left_pane_is_scroll_area(qapp):
    win, tab = _hc_tab(qapp)
    pane0 = tab._splitter.widget(0)
    assert isinstance(pane0, QScrollArea)
    assert pane0.widgetResizable() is True
    assert pane0.horizontalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAlwaysOff


def test_no_hc_widget_overlap_at_650px(qapp):
    win, tab = _hc_tab(qapp)
    win.resize(1100, 650)
    win.show()
    qapp.processEvents()
    panel = tab.panel
    # the tallest content must be able to exceed the viewport -> the scroll area provides room
    assert tab._left_scroll.widget().sizeHint().height() >= tab._left_scroll.viewport().height()
    win.close()
