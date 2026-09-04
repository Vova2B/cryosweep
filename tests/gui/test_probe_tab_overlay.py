import pathlib
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
FIX = pathlib.Path(__file__).resolve().parents[2] / "tests" / "core" / "fixtures"

def _win(qapp):
    from cryosweep_gui.main_window import MainWindow
    return MainWindow()

def test_two_files_overlay_renders_two_lines(qapp):
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "act_synth.dat"), str(FIX / "act_synth.dat")])   # same file twice = 2 entries
    tab.analyze_and_render()
    card = tab.output._cards[0]
    assert len([ln for ln in card.figure.axes[0].lines if ln.get_marker() != "None"]) >= 2
    assert tab._is_overlay()           # ≥2 included files

def test_one_file_is_ab_path(qapp):
    win = _win(qapp); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "vsm_synth.dat")])
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    tab.analyze_and_render()
    assert not tab._is_overlay()        # N=1 -> A/B
    assert len(tab.output.findChildren(FigureCanvasQTAgg)) == 4   # 4 backed VSM kinds, A/B layout

def test_include_toggle_excludes_file(qapp):
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "act_synth.dat"), str(FIX / "act_synth.dat")])
    tab.set_include(1, False)           # exclude the 2nd file
    tab.analyze_and_render()
    assert not tab._is_overlay()        # only 1 included -> A/B
