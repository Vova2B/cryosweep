import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

def _hall_tab(win):
    win.select_probe("hall")
    return win.tabs.currentWidget()

def test_hall_single_file_parity(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hall_path)
    tab = _hall_tab(win)
    tab.panel.hall_channel_edit.setText("1")
    tab.panel.thickness_edit.setText("0.1"); tab.panel.thickness_unit.setCurrentText("mm")
    tab.panel.long_channel_edit.setText("2")
    gui = tab.analyze()
    direct = analyze_file(load_dat(str(hall_path)),
                          RunConfig.load(unit_system="CGS", probe_override="hall",
                                         hall={"hall_channel": 1, "thickness_mm": 0.1,
                                               "geometry_sign": 1, "longitudinal_channel": 2}),
                          build_default_registry())
    assert gui.model_dump_json() == direct.model_dump_json()
    assert gui.status == "ok"

def test_hall_two_file_longitudinal_parity(qapp, hall_path, hall_long_synth_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hall_path)
    tab = _hall_tab(win)
    tab.panel.hall_channel_edit.setText("1")
    tab.panel.thickness_edit.setText("0.1")
    tab.panel.long_channel_edit.setText("2")
    tab.panel.set_longitudinal_file(str(hall_long_synth_path))
    gui = tab.analyze()
    direct = analyze_file(load_dat(str(hall_path)),
                          RunConfig.load(unit_system="CGS", probe_override="hall",
                                         hall={"hall_channel": 1, "thickness_mm": 0.1, "geometry_sign": 1,
                                               "longitudinal_channel": 2, "longitudinal_file": str(hall_long_synth_path)}),
                          build_default_registry())
    assert gui.model_dump_json() == direct.model_dump_json()
    assert gui.status == "ok"
    assert gui.data["longitudinal_source"].startswith("file:")

def test_hall_missing_channel_is_gated(qapp, hall_path):
    # Repinned 2026-09-05 (was status=="error"/export disabled): a missing hall channel now
    # gates like every other missing input. Gated results are exportable by the existing
    # _EXPORTABLE contract (header-only CSVs, same as a gated MPMS file), so the export
    # button is enabled.
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hall_path)
    tab = _hall_tab(win)
    tab.panel.hall_channel_edit.setText("")     # clear the auto-detected channel
    res = tab.analyze()
    assert res.status == "gated"
    assert any(g.need == "hall_channel" for g in res.gate)
    tab.show_result(res)
    assert tab.export_btn.isEnabled()

def test_hall_ui_state_ok(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hall_path)
    tab = _hall_tab(win)
    tab.panel.hall_channel_edit.setText("1"); tab.panel.thickness_edit.setText("0.1")
    tab.panel.long_channel_edit.setText("2")
    tab.show_result(tab.analyze())
    assert "ok" in win.banner.text().lower()
    assert tab.output.last_figure is not None
    assert tab.export_btn.isEnabled() and tab.saveplot_btn.isEnabled()
    rows = {tab.output.table.item(i, 0).text() for i in range(tab.output.table.rowCount())}
    assert any("hall_channel" in r for r in rows)

def test_hall_geometry_sign_flips_R_H(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hall_path)
    tab = _hall_tab(win)
    tab.panel.hall_channel_edit.setText("1"); tab.panel.thickness_edit.setText("0.1")
    pos = tab.analyze().data["points"][0]["R_H"]
    tab.panel.geometry_sign.setCurrentText("-1")
    neg = tab.analyze().data["points"][0]["R_H"]
    assert pos == pytest.approx(-neg, rel=1e-6)
