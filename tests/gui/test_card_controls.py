import dataclasses, pathlib
FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

def _res():
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_axis_commit_changes_only_that_card(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry, PlotSpec
    res = _res()
    p = OutputPanel()
    p.show_result(res, PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t")]))
    card = p._cards[0]
    card.strip.set_axis(ymin=1e-5, ymax=1e-2)            # simulate editingFinished
    assert card.entry.spec.ymin == 1e-5 and card.entry.spec.ymax == 1e-2
    assert card.figure.axes[0].get_ylim() == (1e-5, 1e-2)

def test_scale_combo_initialised_from_kind_default(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry
    p = OutputPanel()
    p.show_result(_res(), PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t")]))
    assert p._cards[0].strip.yscale_value() == "linear"  # resistivity_rho_t kind default (D3: linear headline)

def test_curve_toggle_rerenders_card(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry
    p = OutputPanel()
    p.show_result(_res(), PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t")]))
    card = p._cards[0]
    card.strip.checklist.select_all()                    # select every ramp
    assert card.entry.spec.curves == card.strip.checklist.checked_keys()
    # count DATA lines only (gid is None) — Task 6 adds power-law fit overlays (gid="fit")
    data_lines = [l for l in card.figure.axes[0].lines if l.get_gid() is None]
    assert len(data_lines) == len(card.entry.spec.curves)

def test_rho_t2_fit_line_checkboxes_toggle_spec(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry
    p = OutputPanel()
    p.show_result(_res(), PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t2")]))
    strip = p._cards[0].strip
    # both checkboxes exist and start checked (spec.fit_lines is None -> draw all)
    assert hasattr(strip, "_fit_linear_cb") and hasattr(strip, "_fit_power_cb")
    assert strip._fit_linear_cb.isChecked() and strip._fit_power_cb.isChecked()
    # unchecking power-law leaves only the βT² fit selected
    strip._fit_power_cb.setChecked(False)
    assert p._cards[0].entry.spec.fit_lines == ("linear",)

def test_non_rho_t2_card_has_no_fit_line_checkboxes(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry
    p = OutputPanel()
    p.show_result(_res(), PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t")]))
    strip = p._cards[0].strip
    assert not hasattr(strip, "_fit_linear_cb")
