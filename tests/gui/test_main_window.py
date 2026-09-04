import json
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

def test_window_has_a_tab_per_registry_probe(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    keys = set(build_default_registry().analyzer_keys())
    tab_probes = {win.tabs.widget(i).probe for i in range(win.tabs.count())}
    assert tab_probes == keys

def test_window_load_detects_and_preselects(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(vsm_path)
    assert "vsm" in win.chip.text().lower()
    assert win.tabs.currentWidget().probe == "vsm"

def test_window_hall_file_chip_notes_caveat(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(hall_path)
    assert "resistivity" in win.chip.text().lower()
    assert win.tabs.currentWidget().probe != "hall"

def test_window_vsm_parity_through_shell(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(vsm_path)
    win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200.0"); tab.panel.mass_mg_edit.setText("5.0")
    gui_res = tab.analyze()
    import dataclasses
    rt = load_dat(str(vsm_path))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    direct = analyze_file(rt, RunConfig.load(unit_system="CGS", probe_override="vsm"), build_default_registry())
    assert gui_res.model_dump_json() == direct.model_dump_json()

def test_window_robustness_vsm_file_on_other_tabs(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(vsm_path)
    for probe in ("resistivity", "heatcapacity", "hall"):
        win.select_probe(probe)
        res = win.tabs.currentWidget().analyze()
        assert res.status in ("error", "low_confidence", "gated", "ok")   # never raises

def test_window_bespoke_panels_for_res_and_hc(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    panels = {win.tabs.widget(i).probe: type(win.tabs.widget(i).panel).__name__
              for i in range(win.tabs.count())}
    assert panels["vsm"] == "VSMInputPanel"
    assert panels["resistivity"] == "ResistivityInputPanel"
    assert panels["heatcapacity"] == "HCInputPanel"
    assert panels["hall"] == "HallInputPanel"
    assert panels["hall_tdep"] == "HallInputPanel"   # D7: Temp-Dep Hall is now drivable

def test_window_no_rawtable_mutation_across_tabs(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(vsm_path)
    rt = win.state.get_raw()
    before = rt.df.to_json()
    for probe in ("vsm", "resistivity", "heatcapacity", "hall", "vsm"):
        win.select_probe(probe); win.tabs.currentWidget().analyze()
    assert win.state.get_raw() is rt                       # same cached object
    assert win.state.get_raw().df.to_json() == before      # dataframe never mutated

def test_window_reload_clears_result_cache(qapp, vsm_path, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(vsm_path)
    win.state.cache_result("vsm", "SENTINEL")
    win.load_path(hall_path)                               # loading a new file clears per-probe results
    assert win.state.get_result("vsm") is None

def test_window_reload_reanalyzes_not_stale(qapp, vsm_path, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(vsm_path); win.select_probe("vsm")
    win.load_path(hall_path)                               # different file; auto-selects resistivity
    win.select_probe("vsm")                                # back to vsm tab
    res = win.tabs.currentWidget().analyze()               # must analyze the HALL file as vsm (error), not show stale vsm-ok
    assert res.status == "error"

def test_load_path_renders_landing_tab_once(qapp, hall_path):
    """Regression (owner 2026-07-09): load_path rendered the landing tab twice — setCurrentIndex
    fires currentChanged -> _reanalyze_active, then select_probe called _reanalyze_active again.
    The first render's cards were torn down before ever being realized, flashing as top-level
    windows on macOS (and every load ran the full analysis twice)."""
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    target = next(win.tabs.widget(i) for i in range(win.tabs.count())
                  if win.tabs.widget(i).probe == "resistivity")
    assert win.tabs.currentWidget() is not target      # landing tab differs from startup tab
    calls = {"n": 0}
    orig = target.show_result
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    target.show_result = counting
    win.load_path(hall_path)                           # resistivity-format file lands on that tab
    assert win.tabs.currentWidget() is target
    assert calls["n"] == 1, f"landing tab rendered {calls['n']}x on one load (expected 1)"
