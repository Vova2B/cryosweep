import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

def test_resistivity_tab_geometry_parity_and_rho_source(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(hall_path)              # hall file detects as resistivity
    win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.panel.width_edit.setText("2.0"); tab.panel.thickness_edit.setText("0.5"); tab.panel.length_edit.setText("3.0")
    gui = tab.analyze()
    direct = analyze_file(load_dat(str(hall_path)),
                          RunConfig.load(unit_system="CGS", probe_override="resistivity",
                                         geometry={"width_mm": 2.0, "thickness_mm": 0.5, "length_mm": 3.0}),
                          build_default_registry())
    assert gui.model_dump_json() == direct.model_dump_json()    # GUI mangles nothing
    assert gui.data["rho_source"] == "geometry"                 # geometry input flowed through

def test_resistivity_tab_no_geometry_uses_instrument(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hall_path); win.select_probe("resistivity")
    res = win.tabs.currentWidget().analyze()                    # no geometry entered
    assert res.data["rho_source"] == "instrument_column"        # silent fallback, surfaced in the table

def test_hc_tab_parity_and_theta_d(qapp, hc_synth_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(hc_synth_path)
    assert win.tabs.currentWidget().probe == "heatcapacity"     # auto-detected + preselected
    tab = win.tabs.currentWidget()
    gui = tab.analyze()
    direct = analyze_file(load_dat(str(hc_synth_path)),
                          RunConfig.load(unit_system="CGS", probe_override="heatcapacity"),
                          build_default_registry())
    assert gui.model_dump_json() == direct.model_dump_json()
    assert gui.status == "ok"
    assert gui.data["fit"]["params"]["theta_D"] == pytest.approx(226.777, rel=1e-3)

def test_hc_tab_ui_state(qapp, hc_synth_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hc_synth_path)
    tab = win.tabs.currentWidget()
    tab.show_result(tab.analyze())
    assert "ok" in win.banner.text().lower()
    assert tab.output.last_figure is not None                   # HC plot rendered
    assert tab.export_btn.isEnabled() and tab.saveplot_btn.isEnabled()
    rows = {tab.output.table.item(i, 0).text() for i in range(tab.output.table.rowCount())}
    assert any("theta_D" in r for r in rows)                    # fit params shown in the generic table

def test_hc_n_atoms_override_changes_theta_d(qapp, hc_synth_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hc_synth_path)
    tab = win.tabs.currentWidget()
    base = tab.analyze().data["fit"]["params"]["theta_D"]       # header n_atoms = 3
    tab.panel.n_atoms_edit.setText("6")                          # override doubles n -> theta_D scales by 2^(1/3)
    overridden = tab.analyze().data["fit"]["params"]["theta_D"]
    assert overridden == pytest.approx(base * (2 ** (1.0 / 3.0)), rel=1e-3)
