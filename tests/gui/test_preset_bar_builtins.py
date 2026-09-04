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


def _names(bar):
    return [bar.combo.itemText(i) for i in range(bar.combo.count())]


def test_fresh_store_lists_builtins_first(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    names = _names(tab.preset_bar)
    assert names[:2] == ["★ Journal", "★ All plots"]


def test_delete_disabled_on_builtin(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.combo.setCurrentText("★ Journal")
    tab.preset_bar._on_combo_changed()
    assert tab.preset_bar._btns["Del"].isEnabled() is False


def test_delete_builtin_is_noop(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.delete("★ Journal")
    assert "★ Journal" in _names(tab.preset_bar)          # still present


def test_load_builtin_applies_journal_layout(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.load("★ Journal")
    assert [e.kind for e in tab._layout_state.plots] == ["vsm_moment_t", "vsm_chi_t", "vsm_mh"]


def test_save_as_builtin_name_keeps_builtin(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    tab.preset_bar.save_as("Journal")                     # user override
    assert any(p.name == "Journal" and p.probe == "vsm" for p in win.preset_store.presets)
    assert "★ Journal" in _names(tab.preset_bar)          # built-in still listed


def test_tooltip_has_no_styling_claim(qapp, tmp_path):
    win, tab = _win(qapp, tmp_path)
    assert "styl" not in tab.preset_bar.combo.toolTip().lower()
