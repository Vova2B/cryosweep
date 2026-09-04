import json, pathlib
from cryosweep_core.plotting.presets import PresetStore, save_store
from cryosweep_core.plotting.spec import GlobalStyle, PlotLayout, PlotEntry
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def test_seeded_store_restores_style_and_layout(qapp, tmp_path):
    path = tmp_path / "presets.json"
    from cryosweep_core.registry import build_default_registry
    vsm_kinds = [k.key for k in build_default_registry().plot_kinds_for("vsm")]
    save_store(PresetStore(global_style=GlobalStyle(marker="s"),
                           last_used={"vsm": PlotLayout(plots=[PlotEntry(kind="inverse_chi")],
                                                        known=vsm_kinds)}), path)   # deliberate subset
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=path)
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    win._reanalyze_active()
    assert tab.controls.style.marker == "s"                       # global style applied
    assert {e.kind for e in tab._layout_state.plots} == {"inverse_chi"}   # last_used restored

def test_style_change_persists_to_store(qapp, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=tmp_path / "presets.json")
    win.tabs.widget(0).controls.set_marker("d")                   # emits style_changed
    assert win.preset_store.global_style.marker == "d"

def test_corrupt_file_still_launches(qapp, tmp_path):
    p = tmp_path / "presets.json"; p.write_text("garbage{{{")
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=p)                               # must not raise
    assert win.preset_store.global_style.marker == "o"

def test_closeevent_writes_last_used(qapp, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    path = tmp_path / "presets.json"
    win = MainWindow(preset_path=path)
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    win._reanalyze_active()
    win.close()
    saved = json.loads(path.read_text())
    assert "vsm" in saved["last_used"]                            # backstop wrote final state
