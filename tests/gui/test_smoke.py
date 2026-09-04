import json, subprocess, sys, pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]

def test_window_constructs_and_each_tab_renders_offscreen(qapp, vsm_path, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    # VSM file: every tab analyze()+show_result must not raise (P0 + guards)
    win.load_path(vsm_path)
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        tab = win.tabs.currentWidget()
        res = tab.analyze()
        tab.show_result(res)                        # render (or placeholder) without exception
        assert res.status in ("ok", "gated", "low_confidence", "error")
    # Hall-format file likewise
    win.load_path(hall_path)
    for i in range(win.tabs.count()):
        win.tabs.setCurrentIndex(i)
        win.tabs.currentWidget().show_result(win.tabs.currentWidget().analyze())

def test_python_m_cryosweep_gui_is_importable_offscreen():
    # `python -m cryosweep_gui` would block on exec(); instead prove the module + build_window work headless.
    code = "import os; os.environ['QT_QPA_PLATFORM']='offscreen'; from cryosweep_gui.app import build_window; build_window(); print('GUI_OK')"
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert "GUI_OK" in out.stdout, out.stderr
