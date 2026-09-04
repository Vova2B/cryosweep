import pathlib
from cryosweep_core.plotting.spec import PlotLayout, PlotEntry, PlotSpec, GlobalStyle
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _win(qapp, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=tmp_path / "presets.json")
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    win._reanalyze_active()
    return win, tab

def test_save_as_then_combo_lists_it(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.save_as("My Preset")
    names = [tab.preset_bar.combo.itemText(i) for i in range(tab.preset_bar.combo.count())]
    assert "My Preset" in names
    assert any(p.name == "My Preset" and p.probe == "vsm" for p in win.preset_store.presets)

def test_save_as_rejects_blank(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.save_as("   ")
    assert win.preset_store.presets == []          # blank rejected

def test_load_applies_preset(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    from cryosweep_core.plotting.presets import NamedPreset
    win.preset_store.presets.append(
        NamedPreset(name="one", probe="vsm", layout=PlotLayout(plots=[PlotEntry(kind="vsm_moment_t")])))
    tab.preset_bar.refresh()
    tab.preset_bar.load("one")
    assert {e.kind for e in tab._layout_state.plots} == {"vsm_moment_t"}

def test_delete_removes(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.save_as("x"); tab.preset_bar.delete("x")
    assert all(p.name != "x" for p in win.preset_store.presets)

def test_export_then_import_roundtrip(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.save_as("exp")
    out = tmp_path / "shared.json"
    tab.preset_bar.export_to(str(out))
    assert out.exists() and (tmp_path / "shared.style.json").exists()
    tab.preset_bar.import_from(str(out), name="imported")
    assert any(p.name == "imported" for p in win.preset_store.presets)

def test_export_writes_cli_loadable_files(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    out = tmp_path / "p.json"; tab.preset_bar.export_to(str(out))
    PlotLayout.model_validate_json(out.read_text())
    GlobalStyle.model_validate_json((tmp_path / "p.style.json").read_text())
