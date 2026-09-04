import pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"
RES = str(FIX / "act_synth.dat")          # resistivity-format file
HALL = str(FIX / "hall_synth.dat")


def _win(qapp):
    from cryosweep_gui.main_window import MainWindow
    return MainWindow()


def _tab(win, probe):
    for i in range(win.tabs.count()):
        if win.tabs.widget(i).probe == probe:
            win.tabs.setCurrentIndex(i)
            return win.tabs.widget(i)
    raise AssertionError(probe)


def test_detect_probe_for_returns_applicable_set(qapp):
    win = _win(qapp)
    score, key, applicable = win.detect_probe_for(RES)
    assert key == "resistivity" and score >= 0.5
    assert "resistivity" in applicable                 # and hall/hall_tdep as autopopulate allows


def test_resistivity_on_hc_warns_and_routes_to_load(qapp, monkeypatch):
    win = _win(qapp); tab = _tab(win, "heatcapacity")
    # decision stub -> choose "load as detected"
    monkeypatch.setattr(tab.file_manager, "_ask_mismatch",
                        lambda detected, tabname: "load")
    loaded = {}
    monkeypatch.setattr(win, "load_path", lambda p: loaded.setdefault("path", p))
    tab.file_manager._guarded_add(RES)
    assert loaded.get("path") == RES                   # routed to primary load
    assert len(tab._files) == 0                         # NOT overlaid onto HC


def test_resistivity_on_hall_does_not_warn(qapp, monkeypatch):
    win = _win(qapp); tab = _tab(win, "hall")
    called = {"asked": False}
    monkeypatch.setattr(tab.file_manager, "_ask_mismatch",
                        lambda *a: called.__setitem__("asked", True) or "cancel")
    # HALL is a resistivity-format file with Hall-sweep data -> hall IS in its applicable set,
    # so overlaying it on the Hall tab must NOT warn (tests the applicability-set logic, not
    # detected_key == tab.probe).
    tab.file_manager._guarded_add(HALL)
    assert called["asked"] is False
    assert len(tab._files) == 1                         # overlaid normally


def test_second_hc_on_hc_does_not_warn(qapp, monkeypatch):
    win = _win(qapp); tab = _tab(win, "heatcapacity")
    called = {"asked": False}
    monkeypatch.setattr(tab.file_manager, "_ask_mismatch",
                        lambda *a: called.__setitem__("asked", True) or "cancel")
    tab.file_manager._guarded_add(str(FIX / "hc_synth.dat"))
    assert called["asked"] is False
    assert len(tab._files) == 1


def test_add_button_relabeled(qapp):
    win = _win(qapp); tab = _tab(win, "heatcapacity")
    from PySide6.QtWidgets import QPushButton
    btns = tab.file_manager.findChildren(QPushButton)
    assert any(b.text() == "Add to compare…" for b in btns)
