"""ROADMAP item 3 — entropy off by default in the GUI.

The default HC panel sends entropy_enabled=False, so a GUI analysis carries no entropy
backing and the EXISTING capability gating (empty series -> kind skipped) removes the
hc_entropy_vs_t card, its plot-checklist checkbox and the entropy warnings from the banner.
Checking the box and re-analyzing brings all of them back through the same machinery.
Headless behaviour (CLI/JSON/CSV) is pinned separately in tests/core/test_hc_full_analyzer.py.
"""
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pathlib

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
HC_EXAMPLE = EXAMPLES / "heat_capacity.dat"


def _hc_tab(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.state.load(str(HC_EXAMPLE))
    win.select_probe("heatcapacity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(HC_EXAMPLE)])
    return win, tab


def _card(tab, kind):
    return next((c for c in tab.output._cards if c.entry.kind == kind), None)


def test_gui_default_analysis_has_no_entropy(qapp):
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()
    d = tab._files[0].result.data
    assert d["entropy_available"] is False
    assert "hc_entropy_vs_t" not in tab.controls._boxes        # plot checklist
    assert _card(tab, "hc_entropy_vs_t") is None               # cards
    assert "R ln" not in win.banner.text()                     # no entropy warning surfaced


def test_gui_entropy_checkbox_on_restores_everything(qapp):
    win, tab = _hc_tab(qapp)
    tab.panel.entropy_enable.setChecked(True)
    tab.analyze_and_render()
    d = tab._files[0].result.data
    assert d["entropy_available"] is True
    assert "hc_entropy_vs_t" in tab.controls._boxes
    assert _card(tab, "hc_entropy_vs_t") is not None


def test_gui_entropy_enable_survives_focus_state_roundtrip(qapp):
    """analyze_and_render ends with a set_state restore of the focused entry; the checkbox
    must come back checked, not silently reset to the default."""
    win, tab = _hc_tab(qapp)
    tab.panel.entropy_enable.setChecked(True)
    tab.analyze_and_render()
    assert tab.panel.entropy_enable.isChecked() is True
