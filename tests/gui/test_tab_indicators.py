# tests/gui/test_tab_indicators.py — green dot on each tab that can analyze the loaded file
# (spec 2026-07-02 part 2: applicability comes from detection, recomputed on every load).
import pytest
from cryosweep_gui.main_window import MainWindow


def _dots(win):
    return {win.tabs.widget(i).probe: not win.tabs.tabIcon(i).isNull()
            for i in range(win.tabs.count())}


def test_resistivity_file_lights_applicable_tabs(qapp, hall_path):
    win = MainWindow(); win.load_path(str(hall_path))
    d = _dots(win)
    assert d["resistivity"] is True and d["hall"] is True    # field-sweep Hall applicable
    assert d["hall_tdep"] is False                           # no temp-ramp segments
    assert d["vsm"] is False and d["heatcapacity"] is False


def test_reload_clears_and_recomputes_dots(qapp, hall_path, vsm_path):
    win = MainWindow(); win.load_path(str(hall_path))
    assert _dots(win)["resistivity"] is True
    win.load_path(str(vsm_path))                             # VSM file: dots flip
    d = _dots(win)
    assert d["vsm"] is True
    assert d["resistivity"] is False and d["hall"] is False and d["hall_tdep"] is False


def test_real_hall_file_lights_all_three_transport_tabs(qapp, hall_real_path):
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file not present (gitignored)")
    win = MainWindow(); win.load_path(str(hall_real_path))
    d = _dots(win)
    assert d["resistivity"] is True and d["hall"] is True and d["hall_tdep"] is True
    assert d["vsm"] is False and d["heatcapacity"] is False
