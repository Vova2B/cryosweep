"""Owner 2026-07-09: R_H/mobility stayed empty because thickness + longitudinal channel had to
be typed into each Hall tab separately (and the longitudinal channel wasn't suggested at all).
Load now prefills the detected longitudinal channel alongside the Hall channel, and sample
inputs typed in one Hall tab mirror into the other (same sample, same file)."""
import pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _hall_panels(win):
    out = {}
    for i in range(win.tabs.count()):
        t = win.tabs.widget(i)
        if t.probe in ("hall", "hall_tdep"):
            out[t.probe] = t.panel
    return out

def test_load_prefills_longitudinal_channel(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(str(hall_path))
    p = _hall_panels(win)["hall"]
    assert p.hall_channel_edit.text() == "1"
    assert p.long_channel_edit.text() == "2"

def test_hall_inputs_mirror_between_hall_tabs(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    panels = _hall_panels(win)
    a, b = panels["hall"], panels["hall_tdep"]
    a.thickness_edit.setText("0.5"); a.thickness_edit.textEdited.emit("0.5")   # user typing
    assert b.thickness_edit.text() == "0.5"
    a.thickness_unit.setCurrentText("um")
    assert b.thickness_unit.currentText() == "um"
    a.geometry_sign.setCurrentText("-1")
    assert b.geometry_sign.currentText() == "-1"
    b.long_channel_edit.setText("2"); b.long_channel_edit.textEdited.emit("2")  # both directions
    assert a.long_channel_edit.text() == "2"

def test_mirror_does_not_trigger_sibling_refit(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    panels = _hall_panels(win)
    a, b = panels["hall"], panels["hall_tdep"]
    hits = []
    b.refit_requested.connect(lambda: hits.append(1))
    a.geometry_sign.setCurrentText("-1")       # a refits itself; b updates silently
    assert b.geometry_sign.currentText() == "-1"
    assert hits == []                          # b re-analyzes when shown, not now
