import pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"


def _win(qapp, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(preset_path=tmp_path / "presets.json")
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    win._reanalyze_active()
    return win, tab


def test_combo_defaults_to_oe(qapp, tmp_path):
    win, _ = _win(qapp, tmp_path)
    assert win.field_unit_combo.currentText() == "Oe"
    assert win.preset_store.global_style.field_unit == "Oe"


def test_toggle_to_tesla_updates_style_and_persists(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    win.field_unit_combo.setCurrentText("T")
    assert win.preset_store.global_style.field_unit == "T"
    assert tab.controls.style.field_unit == "T"
    assert win._save_timer.isActive()               # persistence scheduled
