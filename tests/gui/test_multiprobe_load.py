# tests/gui/test_multiprobe_load.py
import pytest
from cryosweep_gui.main_window import MainWindow


def test_load_prefills_hall_channel(qapp, hall_path):
    win = MainWindow(); win.load_path(hall_path)            # hall_synth detects as resistivity, ch1
    win.select_probe("hall")
    assert win.tabs.currentWidget().panel.hall_channel_edit.text() == "1"
    assert "resistivity" in win.chip.text().lower()
    assert "hall" in win.chip.text().lower()


def test_load_seeds_applicable_tabs_only(qapp, hall_path):
    win = MainWindow(); win.load_path(hall_path)
    seeded = {win.tabs.widget(i).probe: len(win.tabs.widget(i)._files)
              for i in range(win.tabs.count())}
    assert seeded["resistivity"] == 1 and seeded["hall"] == 1     # field-sweep applicable
    assert seeded["vsm"] == 0 and seeded["heatcapacity"] == 0     # not applicable


def test_load_tdep_only_file_prompts_for_channel(qapp, hall_tdep_synth_path):
    # temp-ramp-only file: Temp-Dep Hall applicable but channel undetectable -> seeded, empty, prompt.
    win = MainWindow(); win.load_path(str(hall_tdep_synth_path))
    win.select_probe("hall_tdep")
    tab = win.tabs.currentWidget()
    assert len(tab._files) == 1
    assert tab.panel.hall_channel_edit.text() == ""              # not pre-filled (det is None)
    assert "temp-dep hall" in win.chip.text().lower()
    assert "set hall channel" in win.chip.text().lower()


def test_reload_clears_stale_nonapplicable_tab(qapp, vsm_path, hall_path):
    win = MainWindow(); win.load_path(vsm_path)
    win.select_probe("vsm")
    assert len(win.tabs.currentWidget()._files) == 1             # vsm seeded
    win.load_path(hall_path)                                      # hall file: vsm not applicable
    win.select_probe("vsm")
    assert win.tabs.currentWidget()._files == []                 # stale vsm file cleared (D4)
