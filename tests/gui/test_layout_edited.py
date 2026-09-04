def test_card_edit_bubbles_layout_edited(qapp):
    import pathlib
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.plotting.spec import PlotLayout, PlotEntry
    from cryosweep_gui.output_panel import OutputPanel
    FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"
    res = analyze_file(load_dat(str(FIX / "act_synth.dat")),
                       RunConfig.load(probe_override="resistivity"), build_default_registry())
    p = OutputPanel()
    p.show_result(res, PlotLayout(plots=[PlotEntry(kind="resistivity_rho_t")]))
    seen = []
    p.layout_edited.connect(lambda: seen.append(1))
    p._cards[0].strip.set_axis(ymin=1e-5, ymax=1e-2)     # a per-card edit
    assert seen == [1]
