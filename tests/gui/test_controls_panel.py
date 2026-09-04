import dataclasses, pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _vsm():
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())

def test_plot_checkboxes_list_only_backed_kinds(qapp):
    from cryosweep_gui.plot_controls import PlotControlsPanel
    from cryosweep_core.registry import build_default_registry
    res = _vsm()
    panel = PlotControlsPanel(build_default_registry())
    panel.set_result(res, "vsm")
    assert set(panel.enabled_kinds()) == {"inverse_chi", "vsm_moment_t", "vsm_chi_t", "vsm_chi_t_product"}

def test_toggling_plot_off_emits_layout(qapp):
    from cryosweep_gui.plot_controls import PlotControlsPanel
    from cryosweep_core.registry import build_default_registry
    seen = []
    panel = PlotControlsPanel(build_default_registry())
    panel.layout_changed.connect(lambda lay: seen.append([e.kind for e in lay.plots]))
    panel.set_result(_vsm(), "vsm")
    panel.set_kind_enabled("vsm_chi_t", False)
    assert seen[-1] == ["inverse_chi", "vsm_moment_t", "vsm_chi_t_product"]

def test_styling_change_mutates_globalstyle_and_emits(qapp):
    from cryosweep_gui.plot_controls import PlotControlsPanel
    from cryosweep_core.registry import build_default_registry
    seen = []
    panel = PlotControlsPanel(build_default_registry())
    panel.style_changed.connect(lambda st: seen.append(st.marker))
    panel.set_marker("s")
    assert panel.style.marker == "s" and seen[-1] == "s"

def test_set_result_hides_and_orphans_stale_checkboxes(qapp):
    """Regression (owner: garbled overlapping labels at the checklist bottom): set_result
    cleared old checkboxes with bare deleteLater(), so until the event loop ran its deferred
    deletes they stayed parented to the Plots group and painted stacked over the new rows.
    They must be hidden AND orphaned immediately, not just queued for deletion."""
    from cryosweep_gui.plot_controls import PlotControlsPanel
    from cryosweep_core.registry import build_default_registry
    panel = PlotControlsPanel(build_default_registry())
    panel.show()
    panel.set_result(_vsm(), "vsm")
    old = list(panel._boxes.values())
    assert len(old) >= 3
    panel.set_result(_vsm(), "vsm")            # rebuild, no event-loop turn between
    for cb in old:
        assert cb.parent() is None, f"stale checkbox still parented: {cb.text()}"
        assert cb.isHidden(), f"stale checkbox still visible: {cb.text()}"
