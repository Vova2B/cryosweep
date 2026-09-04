import pytest

def _vsm_tab(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(vsm_path); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    return win, tab

def test_request_analysis_sets_busy_then_shows_result(qapp, vsm_path):
    win, tab = _vsm_tab(qapp, vsm_path)
    tab.request_analysis()
    busy_disabled = not tab.analyze_btn.isEnabled()  # capture busy state BEFORE waiting
    busy_text = win.banner.text().lower()
    tab._worker.wait(5000); qapp.processEvents()      # ALWAYS join before any assertion (no dangling thread on failure)
    assert busy_disabled
    assert "analyz" in busy_text
    assert tab.analyze_btn.isEnabled()
    assert tab.output.last_figure is not None
    assert "ok" in win.banner.text().lower()

def test_close_event_joins_running_worker(qapp, vsm_path):
    win, tab = _vsm_tab(qapp, vsm_path)
    tab.request_analysis()
    win.close()                                       # closeEvent -> stop_worker() waits on each tab
    assert not tab._worker.isRunning()

def test_request_analysis_result_matches_sync(qapp, vsm_path):
    win, tab = _vsm_tab(qapp, vsm_path)
    sync = tab.analyze()
    tab.request_analysis(); tab._worker.wait(5000); qapp.processEvents()
    assert tab._last_result.model_dump_json() == sync.model_dump_json()

def test_request_analysis_no_file_no_worker(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    tab = win.tabs.widget(0)
    tab.request_analysis()
    assert tab._worker is None
    assert tab.analyze_btn.isEnabled()

def test_request_analysis_reentrant_ignored(qapp, vsm_path):
    win, tab = _vsm_tab(qapp, vsm_path)
    class _FakeRunning:
        def isRunning(self): return True
    tab._worker = _FakeRunning()
    tab.request_analysis()
    assert isinstance(tab._worker, _FakeRunning)
    tab._worker = None

def test_request_analysis_each_probe_no_crash(qapp, hall_path, hc_synth_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(hall_path)
    win.select_probe("resistivity")
    rt_tab = win.tabs.currentWidget()
    rt_tab.request_analysis(); rt_tab._worker.wait(5000); qapp.processEvents()
    assert rt_tab._last_result.status in ("ok", "low_confidence")
    win.select_probe("hall")
    h = win.tabs.currentWidget()
    h.panel.hall_channel_edit.setText("1"); h.panel.thickness_edit.setText("0.1")
    h.request_analysis(); h._worker.wait(5000); qapp.processEvents()
    assert h._last_result.status == "ok" and h.output.last_figure is not None
