import pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _win(qapp):
    from cryosweep_gui.main_window import MainWindow
    return MainWindow()

def test_resistivity_overlay_endtoend(qapp):
    win = _win(qapp); win.load_path(str(FIX / "act_synth.dat"))   # resistivity, 1 file
    tab = win.tabs.currentWidget()
    assert tab.probe == "resistivity" and not tab._is_overlay()
    tab.add_overlay_path(str(FIX / "act_synth.dat"))              # add a 2nd file -> overlay
    assert tab._is_overlay()
    card = tab.output._cards[0]
    lines = [ln for ln in card.figure.axes[0].lines if ln.get_marker() != "None"]
    assert len(lines) >= 2
    labels = {ln.get_label() for ln in lines}
    assert any(" · " in l for l in labels)                       # filename-tagged legend

def test_focused_file_export_target(qapp):
    win = _win(qapp); win.load_path(str(FIX / "act_synth.dat"))
    tab = win.tabs.currentWidget(); tab.add_overlay_path(str(FIX / "act_synth.dat"))
    tab._focus = 1
    assert tab._files[tab._focus].result is not None             # focused entry has a result for export
