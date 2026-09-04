import pathlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from cryosweep_core.plotting.presets import builtin_presets
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _win(qapp, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=tmp_path / "presets.json")
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    return win, tab

def test_async_analyze_restores_last_used(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    win._reanalyze_active()
    tab.controls.set_kind_enabled("vsm_moment_t", False)
    tab.controls.set_kind_enabled("vsm_chi_t", False)
    tab.controls.set_kind_enabled("vsm_chi_t_product", False)   # only inverse_chi left -> persisted
    tab.request_analysis(); tab._worker.wait(5000); qapp.processEvents()
    assert {e.kind for e in tab._layout_state.plots} == {"inverse_chi"}   # async path restored, not all-on

def test_global_style_persists_across_tab_switch(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    win._reanalyze_active()
    tab.controls.set_marker("s")                               # global
    win.tabs.setCurrentIndex((win.tabs.currentIndex() + 1) % win.tabs.count())  # switch tab
    other = win.tabs.currentWidget()
    assert other.controls.style.marker == "s"                 # applied on show

def test_first_launch_no_file_no_crash(qapp, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=tmp_path / "fresh.json")     # no file, no .dat
    tab = win.tabs.widget(0)
    # combo is seeded with built-in layout presets even before any data is loaded (Task 12);
    # no user presets exist yet, so only the marked built-ins show, no exception either way.
    names = [tab.preset_bar.combo.itemText(i) for i in range(tab.preset_bar.combo.count())]
    assert names == [tab.preset_bar._MARK + p.name for p in builtin_presets(tab.probe)]
    assert not tab.preset_bar._btns["Save As"].isEnabled()    # gated: nothing to save
