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

def test_card_strip_collapsed_by_default(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm()
    p = OutputPanel()
    p.show_result(res)
    card = p._cards[0]
    # isHidden() checks the explicit flag, independent of whether ancestors are shown
    assert card.strip._body.isHidden() is True          # collapsed by default; canvas shows first
    card.strip._toggle.setChecked(True)
    assert card.strip._body.isHidden() is False          # expands on demand

def test_card_title_uses_human_label(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    p = OutputPanel(); p.show_result(_vsm())
    titles = {c.title.text() for c in p._cards}
    assert "1/χ vs T" in titles                          # kind.label, not the raw key "inverse_chi"

def test_show_result_renders_stack_once(qapp, monkeypatch):
    # Guard against the double full-stack render: building N cards should call render_kind N times, not 2N.
    import cryosweep_gui.output_panel as op
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    import pathlib
    win.load_path(str(FIX / "vsm_synth.dat")); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    calls = {"n": 0}
    real = op.render_kind
    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)
    monkeypatch.setattr(op, "render_kind", counting)
    tab.show_result(tab.analyze())
    assert calls["n"] == 4                                # 4 VSM kinds, rendered once each (not 8)
