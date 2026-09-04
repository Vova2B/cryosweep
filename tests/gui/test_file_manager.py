import pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _win(qapp):
    from cryosweep_gui.main_window import MainWindow
    return MainWindow()

def test_add_file_button_appends_without_redetect(qapp):
    win = _win(qapp); win.load_path(str(FIX / "vsm_synth.dat"))   # detects vsm, selects it
    assert win.tabs.currentWidget().probe == "vsm"
    tab = win.tabs.currentWidget()
    tab.add_overlay_path(str(FIX / "act_synth.dat"))             # a resistivity file ADDED to the VSM tab
    assert win.tabs.currentWidget().probe == "vsm"               # did NOT jump/re-detect
    assert len(tab._files) == 2
    rows = tab.file_manager.list.count()
    assert rows == 2                                            # both shown in the manager

def test_file_manager_lists_and_removes(qapp):
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "act_synth.dat"), str(FIX / "act_synth.dat")]); tab.file_manager.refresh()
    assert tab.file_manager.list.count() == 2
    tab.file_manager.remove_current()       # removes the selected entry
    assert len(tab._files) == 1
