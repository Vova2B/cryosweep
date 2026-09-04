import dataclasses, pathlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _setup_vsm(win):
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    tab.show_result(tab.analyze())
    return tab

def test_vsm_tab_shows_three_plots_and_controls(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(vsm_path)); win.select_probe("vsm")
    tab = _setup_vsm(win)
    assert len(tab.output.findChildren(FigureCanvasQTAgg)) == 4
    assert set(tab.controls.enabled_kinds()) == {"inverse_chi", "vsm_moment_t", "vsm_chi_t", "vsm_chi_t_product"}

def test_toggling_plot_off_removes_a_canvas(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(vsm_path)); win.select_probe("vsm")
    tab = _setup_vsm(win)
    tab.controls.set_kind_enabled("vsm_chi_t", False)
    assert len(tab.output.findChildren(FigureCanvasQTAgg)) == 3

def test_global_marker_change_propagates_to_all_cards(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(vsm_path)); win.select_probe("vsm")
    tab = _setup_vsm(win)
    tab.controls.set_marker("s")
    markers = {c.figure.axes[0].lines[0].get_marker() for c in tab.output._cards if c.figure}
    assert markers == {"s"}

def test_P2_no_horizontal_overflow_at_small_window(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.resize(1100, 650); win.load_path(str(vsm_path)); win.select_probe("vsm")
    tab = _setup_vsm(win)
    assert win.minimumWidth() <= 1100
    # controls is now in the right splitter pane — bounded by minimumWidth, not a maximumWidth cap (P2)
    assert tab.controls.minimumWidth() >= 250
    for c in tab.output._cards:
        if c.canvas:
            assert c.canvas.minimumHeight() >= 280
