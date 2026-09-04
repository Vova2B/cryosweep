import pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"


def _win(qapp):
    from cryosweep_gui.main_window import MainWindow
    return MainWindow()


def test_unplottable_file_dropped_and_warned(qapp):
    """An unplottable file in a 2-file set is dropped → the one good file renders as a single
    (untagged) plot, not an overlay, and the banner warns that a file was not plotted. A failed
    load (result is None) is the deterministic unplottable case."""
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "act_synth.dat"), str(FIX / "does_not_exist.dat")])
    tab.analyze_and_render()
    assert tab._files[0].result is not None and tab._files[0].result.status in ("ok", "low_confidence")
    assert tab._files[1].result is None                     # load failed -> not plottable
    # only the good file rendered -> single, untagged labels (no " · " file tag)
    card = tab.output._cards[0]
    lines = [ln for ln in card.figure.axes[0].lines if ln.get_marker() != "None"]
    assert lines and all(" · " not in ln.get_label() for ln in lines)
    assert "not plotted" in tab.banner.text()


def test_two_good_files_still_overlay_no_note(qapp):
    """Sanity: when both files are renderable, overlay still happens and no drop note appears."""
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "act_synth.dat"), str(FIX / "act_synth.dat")])
    tab.analyze_and_render()
    assert tab._is_overlay()
    card = tab.output._cards[0]
    lines = [ln for ln in card.figure.axes[0].lines if ln.get_marker() != "None"]
    assert len(lines) >= 2
    assert "not plotted" not in tab.banner.text()


def test_overlay_export_targets_focused_file(qapp):
    """In overlay mode, Export/Save-report act on the FOCUSED file (spec §6), not results[0]."""
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "act_synth.dat"), str(FIX / "act_synth.dat")])
    tab._focus = 1
    tab.analyze_and_render()
    assert tab._is_overlay()
    assert tab._last_result is tab._files[1].result        # focused file is the export target
    assert tab._last_result is not tab._files[0].result    # not the first usable result


def test_remove_reconverges_focus_and_selection(qapp):
    """After removing a file, the highlighted list row matches the tab's authoritative _focus
    (would diverge under the old refresh that preserved the widget's own currentRow)."""
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "act_synth.dat"), str(FIX / "act_synth.dat"), str(FIX / "act_synth.dat")])
    fm = tab.file_manager
    # force a divergence: select row 0 in the widget without firing the handler, focus elsewhere
    fm.list.blockSignals(True); fm.list.setCurrentRow(0); fm.list.blockSignals(False)
    tab._focus = 2
    fm.remove_current()                    # removes widget row 0
    assert len(tab._files) == 2
    assert 0 <= tab._focus < len(tab._files)
    assert fm.list.currentRow() == tab._focus
