# tests/gui/test_hall_pq2_kinds.py — PQ-2 Task 4: GUI checklist + preset-round-trip
# coverage for the Hall journal-pass kinds (2 upgraded + 4 new composites).
#
# New kinds under test:
#   hall_two_panel       (probe hall)       "Hall | Longitudinal"
#   hall_rh_n_twin       (probe hall)       "R_H + carrier n vs T"
#   hall_tdep_summary    (probe hall_tdep)  "R_H / μ / J vs T (summary)"
#   hall_tdep_rh_n_twin  (probe hall_tdep)  "R_H + carrier n vs T"
# Upgraded kinds (Tasks 1-2, already shipped, re-asserted present here):
#   hall_raw_vs_asym     (probe hall)       "Antisymmetrization"
#   hall_tdep_stages     (probe hall_tdep)  "Stage diagnostics"
from cryosweep_core.plotting.presets import reconcile_layout
from cryosweep_core.plotting.spec import PlotEntry, PlotLayout
from cryosweep_core.registry import build_default_registry


def _hall_tab_with_longitudinal(win):
    """hall probe, configured so hall_two_panel/hall_rh_n_twin have backing data."""
    win.select_probe("hall")
    tab = win.tabs.currentWidget()
    tab.panel.hall_channel_edit.setText("1")
    tab.panel.thickness_edit.setText("0.1")
    tab.panel.long_channel_edit.setText("2")
    tab.show_result(tab.analyze())
    return tab


def _hall_tdep_tab(win, hall_tdep_synth_path):
    win.load_path(str(hall_tdep_synth_path))
    win.select_probe("hall_tdep")
    tab = win.tabs.currentWidget()
    tab.panel.hall_channel_edit.setText("1")
    tab.panel.thickness_edit.setText("0.05")
    tab.panel.long_channel_edit.setText("2")
    tab.show_result(tab.analyze())
    return tab


# ---------------------------------------------------------------------------
# (a) new/upgraded kinds appear in the Plots checklist with their labels
# ---------------------------------------------------------------------------

def test_hall_probe_checklist_has_pq2_kinds(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(hall_path))
    tab = _hall_tab_with_longitudinal(win)
    assert "hall_two_panel" in tab.controls.enabled_kinds()
    assert "hall_rh_n_twin" in tab.controls.enabled_kinds()
    assert "hall_raw_vs_asym" in tab.controls.enabled_kinds()
    labels = {cb.text() for cb in tab.controls._boxes.values()}
    assert "Hall | Longitudinal" in labels
    assert "R_H + carrier n vs T" in labels
    assert "Antisymmetrization" in labels


def test_hall_tdep_checklist_has_pq2_kinds(qapp, hall_tdep_synth_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    tab = _hall_tdep_tab(win, hall_tdep_synth_path)
    assert "hall_tdep_summary" in tab.controls.enabled_kinds()
    assert "hall_tdep_rh_n_twin" in tab.controls.enabled_kinds()
    assert "hall_tdep_stages" in tab.controls.enabled_kinds()
    labels = {cb.text() for cb in tab.controls._boxes.values()}
    assert "R_H / μ / J vs T (summary)" in labels
    assert "R_H + carrier n vs T" in labels
    assert "Stage diagnostics" in labels


# ---------------------------------------------------------------------------
# (b) a composite gated OFF by data (no crash; current convention = it never
#     gets a checkbox — set_result() skips kinds whose series() is empty)
# ---------------------------------------------------------------------------

def test_hall_two_panel_gated_without_longitudinal_no_crash(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(hall_path))
    win.select_probe("hall")
    tab = win.tabs.currentWidget()
    tab.panel.hall_channel_edit.setText("1")
    tab.panel.thickness_edit.setText("0.1")
    tab.panel.long_channel_edit.setText("")   # load now prefills it; this test wants it absent
    # no long_channel_edit -> no rho_xx -> hall_two_panel's series() == []
    result = tab.analyze()
    tab.show_result(result)                       # must not raise
    assert result.status == "ok"
    assert "hall_two_panel" not in tab.controls.enabled_kinds()
    assert "hall_two_panel" not in tab.controls._boxes
    assert "ok" in win.banner.text().lower()               # banner reflects the ok result, no crash


# ---------------------------------------------------------------------------
# (c) preset/layout round-trip: a gated kind in a restored layout survives
#     reconcile_layout as a placeholder entry (kept, not dropped -- only
#     entries whose *kind* is unknown for the probe get dropped).
# ---------------------------------------------------------------------------

def test_gated_kind_survives_layout_roundtrip(qapp, hall_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(hall_path))
    win.select_probe("hall")
    tab = win.tabs.currentWidget()
    tab.panel.hall_channel_edit.setText("1")
    tab.panel.thickness_edit.setText("0.1")
    result = tab.analyze()
    assert result.status == "ok"

    layout = PlotLayout(plots=[PlotEntry(kind="hall_two_panel"), PlotEntry(kind="hall_rh_n_twin")])
    dumped = layout.model_dump()
    restored = PlotLayout.model_validate(dumped)
    assert {e.kind for e in restored.plots} == {"hall_two_panel", "hall_rh_n_twin"}

    reconciled = reconcile_layout(restored, result, build_default_registry())
    kinds = {e.kind for e in reconciled.plots}
    assert "hall_two_panel" in kinds       # gated (series==[]) but kind is known -> kept as placeholder
    assert "hall_rh_n_twin" in kinds       # backed kind -> kept


def test_new_kinds_round_trip_through_layout_restore(qapp, hall_tdep_synth_path):
    """A PlotLayout containing the new kinds, dumped/reloaded, still reconciles intact
    once real analysis data backs them (non-gated case of the round trip)."""
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    tab = _hall_tdep_tab(win, hall_tdep_synth_path)
    result = tab._last_result
    assert result.status == "ok"

    layout = PlotLayout(plots=[PlotEntry(kind="hall_tdep_summary"),
                                PlotEntry(kind="hall_tdep_rh_n_twin")],
                        known=[k.key for k in build_default_registry().plot_kinds_for("hall_tdep")])
    restored = PlotLayout.model_validate(layout.model_dump())   # known survives the round trip too
    reconciled = reconcile_layout(restored, result, build_default_registry())
    kinds = {e.kind for e in reconciled.plots}
    assert kinds == {"hall_tdep_summary", "hall_tdep_rh_n_twin"}
