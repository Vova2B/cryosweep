def test_set_enabled_set_checks_subset_and_emits_nothing(qapp):
    import dataclasses, pathlib
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    from cryosweep_gui.plot_controls import PlotControlsPanel
    FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    res = analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())
    p = PlotControlsPanel(build_default_registry())
    p.set_result(res, "vsm")
    seen = []
    p.layout_changed.connect(lambda lay: seen.append([e.kind for e in lay.plots]))
    p.set_enabled_set(["inverse_chi"])
    assert p.enabled_kinds() == ["inverse_chi"]   # only that one checked
    assert seen == []                             # signal-safe: no emit
