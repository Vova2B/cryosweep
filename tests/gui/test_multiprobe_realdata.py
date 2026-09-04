# tests/gui/test_multiprobe_realdata.py
import pytest
from cryosweep_gui.main_window import MainWindow


def test_real_hall_file_populates_all_hall_tabs(qapp, hall_real_path):
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file not present (gitignored)")
    win = MainWindow(); win.load_path(str(hall_real_path))
    # both Hall tabs got the file + the auto-detected channel (Ch1) and analyze without erroring
    for probe in ("hall", "hall_tdep"):
        win.select_probe(probe)
        tab = win.tabs.currentWidget()
        assert len(tab._files) == 1
        assert tab.panel.hall_channel_edit.text() == "1"
        assert tab.analyze().status in ("ok", "low_confidence")    # not "error"
