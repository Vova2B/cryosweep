# Example-defect D5: a PlotCard whose kind simply has no data in this file must read as a calm
# "not applicable" note, while a genuine rendering failure must stay loudly visible - a grey
# "no plottable data (ValueError)" for both exposed a Python exception class to scientists and
# made real failures indistinguishable from absent data.


def _rho_result():
    from cryosweep_core.result import Result, Provenance
    # a resistivity file with temperature ramps only: every MR kind has nothing to plot
    data = {"probe": "resistivity", "bridges": [], "capabilities": []}
    return Result(status="ok", data=data,
                  provenance=Provenance(file="x", sha256="ab", app_version=None))


def test_card_says_not_applicable_when_kind_has_no_data(qapp):
    from cryosweep_gui.output_panel import PlotCard
    from cryosweep_core.plotting.spec import GlobalStyle, PlotEntry
    card = PlotCard([_rho_result()], PlotEntry(kind="resistivity_mr"), GlobalStyle())
    assert card.canvas is None and card._placeholder is not None
    text = card._placeholder.text()
    assert "not applicable" in text
    assert "ValueError" not in text and "Error" not in text


def test_card_keeps_genuine_failures_visible_and_distinct(qapp, monkeypatch):
    import cryosweep_gui.output_panel as op
    from cryosweep_core.plotting.spec import GlobalStyle, PlotEntry

    def boom(*a, **k):
        raise RuntimeError("synthetic renderer crash")
    monkeypatch.setattr(op, "render_kind", boom)
    card = op.PlotCard([_rho_result()], PlotEntry(kind="resistivity_rho_t"), GlobalStyle())
    text = card._placeholder.text()
    assert "rendering failed" in text and "RuntimeError" in text \
        and "synthetic renderer crash" in text
    assert "not applicable" not in text
