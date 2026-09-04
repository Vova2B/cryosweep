import dataclasses, pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotLayout, PlotEntry, PlotSpec
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _vsm_tab(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    return win, tab

def test_show_result_with_restore_layout_applies_subset(qapp):
    win, tab = _vsm_tab(qapp)
    restore = PlotLayout(plots=[PlotEntry(kind="inverse_chi"), PlotEntry(kind="vsm_chi_t")],
                         known=[k.key for k in build_default_registry().plot_kinds_for("vsm")])
    tab.show_result(tab.analyze(), restore_layout=restore)   # known covers all -> subset is deliberate
    assert {e.kind for e in tab._layout_state.plots} == {"inverse_chi", "vsm_chi_t"}
    assert set(tab.controls.enabled_kinds()) == {"inverse_chi", "vsm_chi_t"}   # checkboxes synced

def test_toggle_preserves_other_plots_edits(qapp):
    win, tab = _vsm_tab(qapp)
    tab.show_result(tab.analyze())                          # 3 plots
    card0 = tab.output._cards[0]
    card0.strip.set_axis(ymin=1.0, ymax=9.0)
    tab.controls.set_kind_enabled("vsm_chi_t", False)
    kept = {e.kind: e.spec for e in tab._layout_state.plots}
    assert "vsm_chi_t" not in kept
    assert kept["inverse_chi"].ymin == 1.0 and kept["inverse_chi"].ymax == 9.0   # edit survived the toggle
