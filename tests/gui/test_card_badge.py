# This codebase has no pytest-qt; widget tests use the session `qapp` fixture
# (see conftest.py) and construct widgets directly.

def test_plotcard_shows_outlier_badge_when_diagnostics_present(qapp):
    from cryosweep_gui.output_panel import PlotCard
    from cryosweep_core.plotting.spec import GlobalStyle, PlotEntry
    from cryosweep_core.result import Result, Provenance, Diagnostic
    d = Diagnostic(kind="outliers", severity="warning", scope="bridge1 rho(T) 0.0 Oe",
                   message="7 outlier points", data={"n_outliers": 7})
    data = {"probe": "resistivity", "bridges": [], "capabilities": []}
    r = Result(status="ok", data=data, diagnostics=[d],
               provenance=Provenance(file="x", sha256="ab", app_version=None))
    card = PlotCard([r], PlotEntry(kind="resistivity_rho_t"), GlobalStyle())
    assert card._badge is not None and "outlier" in card._badge.toolTip().lower()
    assert "7" in card._badge.text()

def test_plotcard_badge_not_duplicated_on_rerender(qapp):
    from PySide6.QtWidgets import QLabel
    from cryosweep_gui.output_panel import PlotCard
    from cryosweep_core.plotting.spec import GlobalStyle, PlotEntry
    from cryosweep_core.result import Result, Provenance, Diagnostic

    def badge_count(card):
        return sum(1 for c in card.findChildren(QLabel)
                   if "outliers" in c.text().lower())

    d = Diagnostic(kind="outliers", severity="warning", scope="bridge1 rho(T) 0.0 Oe",
                   message="7 outlier points", data={"n_outliers": 7})
    data = {"probe": "resistivity", "bridges": [], "capabilities": []}
    r = Result(status="ok", data=data, diagnostics=[d],
               provenance=Provenance(file="x", sha256="ab", app_version=None))
    card = PlotCard([r], PlotEntry(kind="resistivity_rho_t"), GlobalStyle())
    first_badge = card._badge
    assert first_badge is not None
    assert badge_count(card) == 1

    # re-render via the spec-changed path the AxisStrip controls drive
    card.entry.spec.robust_view = False
    card._on_spec_changed()
    # ...and via a direct render() call
    card.render(card._results, card._style, card._overlay)

    assert card._badge is first_badge          # same object, not replaced
    assert badge_count(card) == 1              # not duplicated across re-renders

def test_plotcard_no_badge_when_clean(qapp):
    from cryosweep_gui.output_panel import PlotCard
    from cryosweep_core.plotting.spec import GlobalStyle, PlotEntry
    from cryosweep_core.result import Result, Provenance
    data = {"probe": "resistivity", "bridges": [], "capabilities": []}
    r = Result(status="ok", data=data, provenance=Provenance(file="x", sha256="ab", app_version=None))
    card = PlotCard([r], PlotEntry(kind="resistivity_rho_t"), GlobalStyle())
    assert card._badge is None

def test_plotcard_shows_setpoint_warning_badge(qapp):
    from cryosweep_gui.output_panel import PlotCard
    from cryosweep_core.plotting.spec import GlobalStyle, PlotEntry
    from cryosweep_core.result import Result, Provenance, Diagnostic
    d = Diagnostic(kind="duplicate_setpoints", severity="warning",
                   scope="field setpoints 9.5/10 K",
                   message="near-duplicate setpoints: groups 9.5 K and 10 K are 0.4 K apart")
    data = {"probe": "resistivity", "bridges": [], "capabilities": []}
    r = Result(status="ok", data=data, diagnostics=[d],
               provenance=Provenance(file="x", sha256="ab", app_version=None))
    card = PlotCard([r], PlotEntry(kind="resistivity_mr_pct"), GlobalStyle())
    assert card._dup_badge is not None
    assert "setpoint" in card._dup_badge.text().lower()
    assert "9.5" in card._dup_badge.toolTip() or "near-duplicate" in card._dup_badge.toolTip().lower()
    # the outlier badge stays independent (None here -- no outlier diagnostics)
    assert card._badge is None

def test_plotcard_shows_both_badges_independently(qapp):
    from cryosweep_gui.output_panel import PlotCard
    from cryosweep_core.plotting.spec import GlobalStyle, PlotEntry
    from cryosweep_core.result import Result, Provenance, Diagnostic
    do = Diagnostic(kind="outliers", severity="warning", scope="bridge1 rho(H) 300.0 K",
                    message="7 outlier points", data={"n_outliers": 7})
    dd = Diagnostic(kind="duplicate_setpoints", severity="warning",
                    scope="field setpoints 9.5/10 K", message="near-duplicate setpoints")
    data = {"probe": "resistivity", "bridges": [], "capabilities": []}
    r = Result(status="ok", data=data, diagnostics=[do, dd],
               provenance=Provenance(file="x", sha256="ab", app_version=None))
    card = PlotCard([r], PlotEntry(kind="resistivity_mr_pct"), GlobalStyle())
    # both badges present and independent
    assert card._badge is not None and "7" in card._badge.text()
    assert card._dup_badge is not None and "setpoint" in card._dup_badge.text().lower()
    assert card._badge is not card._dup_badge
