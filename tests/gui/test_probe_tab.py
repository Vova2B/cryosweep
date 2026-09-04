import json
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
import cryosweep_gui.inputs.vsm   # noqa: F401  (register VSM panel)

def _make_tab(probe, get_raw, unit="CGS"):
    from cryosweep_gui.probe_tab import ProbeTab
    from cryosweep_gui.inputs.base import build_panel
    reg = build_default_registry()
    return ProbeTab(probe=probe, panel=build_panel(probe), registry=reg,
                    get_raw=get_raw, get_unit=lambda: unit)


# ── Task-2: 3-zone splitter tests ────────────────────────────────────────────

def test_probe_tab_has_horizontal_splitter_with_3_panes(qapp):
    """Root contains a QSplitter(Horizontal) with 3 panes: [left-inputs, output, controls]."""
    from PySide6.QtWidgets import QSplitter
    from PySide6.QtCore import Qt
    tab = _make_tab("vsm", get_raw=lambda: None)
    splitters = tab.findChildren(QSplitter)
    assert len(splitters) == 1, f"Expected 1 QSplitter, found {len(splitters)}"
    sp = splitters[0]
    assert sp.orientation() == Qt.Orientation.Horizontal
    assert sp.count() == 3, f"Expected 3 panes, got {sp.count()}"
    # pane 1 = output, pane 2 = controls
    assert sp.widget(1) is tab.output, "Center pane must be tab.output"
    assert sp.widget(2) is tab.controls, "Right pane must be tab.controls"


def test_probe_tab_controls_not_width_capped(qapp):
    """controls.maximumWidth() is Qt default — the 360 cap is gone.
    The right pane has a sensible minimumWidth instead."""
    tab = _make_tab("vsm", get_raw=lambda: None)
    assert tab.controls.maximumWidth() == 16777215, (
        f"Expected Qt default (16777215), got {tab.controls.maximumWidth()} — "
        "setMaximumWidth(360) must be removed"
    )
    assert tab.controls.minimumWidth() >= 250, (
        f"Right pane minimumWidth should be >= 250, got {tab.controls.minimumWidth()}"
    )


def test_probe_tab_splitter_initial_sizes_set(qapp):
    """Splitter initial sizes are set: center dominates, left/right panes are non-trivially wide."""
    from PySide6.QtWidgets import QSplitter, QApplication
    tab = _make_tab("vsm", get_raw=lambda: None)
    sp = tab.findChildren(QSplitter)[0]
    # Show the tab so the layout manager allocates real pixel widths.
    tab.resize(1400, 600)
    tab.show()
    QApplication.processEvents()
    sizes = sp.sizes()
    assert len(sizes) == 3
    assert sizes[0] >= 200, f"Left pane too narrow: {sizes[0]}"
    assert sizes[2] >= 200, f"Right pane too narrow: {sizes[2]}"
    assert sizes[1] > sizes[0], f"Center pane should dominate over left: {sizes}"
    assert sizes[1] > sizes[2], f"Center pane should dominate over right: {sizes}"


def test_probe_tab_all_preserved_attributes_exist(qapp):
    """All preserved attributes still exist after layout rework."""
    tab = _make_tab("vsm", get_raw=lambda: None)
    required = ("controls", "output", "panel", "file_manager", "preset_bar",
                 "analyze_btn", "export_btn", "report_btn", "saveplot_btn",
                 "_layout_state", "_files", "banner")
    for attr in required:
        assert hasattr(tab, attr), f"Missing required attribute: {attr}"


def test_probe_tab_layout_changed_signal_fires_after_reparent(qapp, vsm_path):
    """controls.layout_changed still fires and triggers _apply_layout after reparenting."""
    from cryosweep_core.plotting.spec import PlotLayout
    rt = load_dat(str(vsm_path))
    tab = _make_tab("vsm", get_raw=lambda: rt)
    tab.panel.molar_mass_edit.setText("200.0")
    tab.panel.mass_mg_edit.setText("5.0")
    tab.show_result(tab.analyze())

    received: list[PlotLayout] = []
    tab.controls.layout_changed.connect(received.append)
    tab.controls.set_kind_enabled("vsm_moment_t", False)
    assert len(received) == 1
    assert isinstance(received[0], PlotLayout)

def test_probe_tab_analyze_returns_result_no_file(qapp):
    tab = _make_tab("vsm", get_raw=lambda: None)
    assert tab.analyze() is None                      # no file loaded

def test_probe_tab_vsm_parity_with_direct_pipeline(qapp, vsm_path):
    rt = load_dat(str(vsm_path))
    tab = _make_tab("vsm", get_raw=lambda: rt, unit="CGS")
    tab.panel.molar_mass_edit.setText("200.0")
    tab.panel.mass_mg_edit.setText("5.0")
    gui_res = tab.analyze()
    import dataclasses
    rt2 = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    direct = analyze_file(rt2, RunConfig.load(unit_system="CGS", probe_override="vsm"),
                          build_default_registry())
    assert gui_res.model_dump_json() == direct.model_dump_json()
    assert gui_res.status == "ok"

def test_probe_tab_button_gating(qapp, vsm_path):
    rt = load_dat(str(vsm_path))
    tab = _make_tab("vsm", get_raw=lambda: rt)
    tab.panel.molar_mass_edit.setText("200.0"); tab.panel.mass_mg_edit.setText("5.0")
    tab.show_result(tab.analyze())
    assert tab.export_btn.isEnabled() and tab.report_btn.isEnabled() and tab.saveplot_btn.isEnabled()

def test_probe_tab_error_disables_buttons(qapp, hall_path):
    rt = load_dat(str(hall_path))                     # hall file analyzed as vsm -> error
    tab = _make_tab("vsm", get_raw=lambda: rt)
    res = tab.analyze()
    assert res.status == "error"
    tab.show_result(res)
    assert not tab.export_btn.isEnabled()
    assert not tab.report_btn.isEnabled()
    assert not tab.saveplot_btn.isEnabled()


# ── Task-3: collapsible right controls pane ───────────────────────────────────

def test_controls_visible_default_true(qapp):
    """controls_visible is True and the controls pane is visible (not explicitly hidden) at startup."""
    from PySide6.QtWidgets import QApplication
    tab = _make_tab("vsm", get_raw=lambda: None)
    assert hasattr(tab, "controls_visible"), "ProbeTab must expose controls_visible"
    assert tab.controls_visible is True, "Default controls_visible must be True"
    # Verify visible when the tab is actually shown
    tab.resize(1400, 600)
    tab.show()
    QApplication.processEvents()
    assert tab.controls.isVisible(), "controls widget must be visible after show()"


def test_set_controls_visible_false_hides_and_updates_state(qapp):
    """set_controls_visible(False) hides the right pane, controls_visible becomes False,
    the controls widget is NOT destroyed (attributes still exist)."""
    from PySide6.QtWidgets import QApplication
    tab = _make_tab("vsm", get_raw=lambda: None)
    tab.resize(1400, 600)
    tab.show()
    QApplication.processEvents()

    tab.set_controls_visible(False)
    QApplication.processEvents()

    assert tab.controls_visible is False, "controls_visible should be False after hiding"
    assert not tab.controls.isVisible(), "controls widget should not be visible"
    # controls is NOT destroyed — its child attributes still exist
    assert hasattr(tab.controls, "layout_changed"), "controls widget must not be destroyed"


def test_set_controls_visible_true_restores_nonzero_width(qapp):
    """After hide then show, the right pane returns to a non-zero width close to its pre-hide size."""
    from PySide6.QtWidgets import QApplication
    from PySide6.QtWidgets import QSplitter
    tab = _make_tab("vsm", get_raw=lambda: None)
    tab.resize(1400, 600)
    tab.show()
    QApplication.processEvents()

    sp = tab.findChildren(QSplitter)[0]
    sizes_before = sp.sizes()
    right_before = sizes_before[2]

    tab.set_controls_visible(False)
    QApplication.processEvents()

    tab.set_controls_visible(True)
    QApplication.processEvents()

    sizes_after = sp.sizes()
    right_after = sizes_after[2]
    assert right_after > 0, f"Right pane width must be > 0 after restore, got {right_after}"
    # Restored width should be within 5px of original (cache/restore is exact)
    assert abs(right_after - right_before) <= 5, (
        f"Restored right-pane width {right_after} diverged too far from pre-hide {right_before}"
    )
    assert tab.controls_visible is True, "controls_visible should be True after showing"


def test_toggle_button_drives_controls_visible(qapp):
    """Clicking the toggle button flips controls_visible."""
    from PySide6.QtWidgets import QApplication
    tab = _make_tab("vsm", get_raw=lambda: None)
    tab.resize(1400, 600)
    tab.show()
    QApplication.processEvents()

    assert hasattr(tab, "_toggle_btn"), "ProbeTab must have a _toggle_btn QPushButton"
    assert tab.controls_visible is True

    # Click once → hide
    tab._toggle_btn.click()
    QApplication.processEvents()
    assert tab.controls_visible is False, "After first click controls should be hidden"

    # Click again → show
    tab._toggle_btn.click()
    QApplication.processEvents()
    assert tab.controls_visible is True, "After second click controls should be visible"


def test_splitter_still_3_panes_after_toggle(qapp):
    """Hiding controls does NOT change the splitter structure (still 3 panes, widget(2) is controls)."""
    from PySide6.QtWidgets import QSplitter, QApplication
    tab = _make_tab("vsm", get_raw=lambda: None)
    tab.resize(1400, 600)
    tab.show()
    QApplication.processEvents()

    tab.set_controls_visible(False)
    QApplication.processEvents()

    sp = tab.findChildren(QSplitter)
    assert len(sp) == 1
    assert sp[0].count() == 3, "Splitter must still have 3 panes after hiding controls"
    assert sp[0].widget(2) is tab.controls, "widget(2) must still be tab.controls"
