import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.result import Result, Provenance

def _vsm_ok(vsm_path):
    rt = load_dat(str(vsm_path))
    cfg = RunConfig.load(probe_override="vsm")
    import dataclasses
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, cfg, build_default_registry())

def test_output_panel_renders_ok_vsm(qapp, vsm_path):
    from cryosweep_gui.output_panel import OutputPanel, flatten_rows
    res = _vsm_ok(vsm_path)
    assert res.status == "ok"
    p = OutputPanel()
    p.show_result(res)
    assert p.last_figure is not None
    assert not p.placeholder_shown
    assert p.table.rowCount() == len(flatten_rows(res.data))
    assert p.table.rowCount() >= 1

def test_plot_cards_are_parented_not_toplevel(qapp, vsm_path):
    """Regression: PlotCards must be parented to the OutputPanel at creation, else on macOS each
    parentless card flashes as its own top-level window before _relayout_grid reparents them."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm_ok(vsm_path)
    p = OutputPanel()
    p.show_result(res)
    assert len(p._cards) >= 1
    assert all(c.parent() is not None for c in p._cards), "a PlotCard was created without a parent"


def test_capability_hint_from_unmet_capabilities():
    from cryosweep_gui.output_panel import _capability_hint
    data = {"capabilities": [
        {"name": "hall_coefficient", "applicable": False, "reason": "thickness required for R_H"},
        {"name": "antisymmetrization", "applicable": True, "reason": "loops span +/-H"},
        {"name": "mobility", "applicable": False, "reason": "no longitudinal channel/file supplied"},
        {"name": "rich_field_recommended", "applicable": True, "reason": "advisory only"},
    ]}
    hint = _capability_hint(data)
    assert "R_H: thickness required for R_H" in hint
    assert "mobility: no longitudinal channel" in hint
    assert "antisymmetrization" not in hint          # applicable -> not nagged
    assert "advisory only" not in hint               # advisory capability skipped
    # all-applicable -> empty
    assert _capability_hint({"capabilities": [{"name": "x", "applicable": True, "reason": "ok"}]}) == ""
    assert _capability_hint({}) == "" and _capability_hint(None) == ""


def test_cap_strip_visible_for_hall_without_thickness(qapp, hall_real_path):
    """Loading a Hall file without thickness must surface WHY R_H etc. are empty, not stay silent."""
    from cryosweep_gui.output_panel import OutputPanel
    rt = load_dat(str(hall_real_path))
    cfg = RunConfig.load(probe_override="hall", hall={"hall_channel": 1})
    res = analyze_file(rt, cfg, build_default_registry())
    p = OutputPanel()
    p.show_result(res)
    assert not p._cap_strip.isHidden()          # logically shown (window not on-screen in test)
    assert "thickness" in p._cap_strip.text()


def test_flatten_rows_emit_mr_percent():
    from cryosweep_gui.output_panel import flatten_rows
    data = {"probe": "resistivity",
            "bridges": [{"channel": 2, "rho_h_curves": [
                {"held_temp_k": 2.0, "mr_percent_at_max_field": 90.6, "low_confidence": False},
                {"held_temp_k": 10.0, "mr_percent_at_max_field": None},
            ]}]}
    rows = dict(flatten_rows(data))
    assert rows["ch2.mr%@2.0K"] == "90.60%"
    assert "ch2.mr%@10.0K" not in rows


def test_flatten_rows_mr_low_confidence_flagged():
    from cryosweep_gui.output_panel import flatten_rows
    data = {"probe": "resistivity",
            "bridges": [{"channel": 1, "rho_h_curves": [
                {"held_temp_k": 5.0, "mr_percent_at_max_field": 3.2, "low_confidence": True}]}]}
    rows = dict(flatten_rows(data))
    assert rows["ch1.mr%@5.0K"] == "3.20% (low confidence)"


def test_output_panel_placeholder_on_error(qapp):
    from cryosweep_gui.output_panel import OutputPanel
    res = Result(status="error", errors=["no data"], data={"probe": "vsm"},
                 provenance=Provenance(file="x", sha256="", app_version="", config={}))
    p = OutputPanel()
    p.show_result(res)
    assert p.last_figure is None
    assert p.placeholder_shown

def test_flatten_rows_picks_scalars_fit_and_capabilities():
    from cryosweep_gui.output_panel import flatten_rows
    data = {"probe": "hall", "thickness_m": 1e-4, "points": [1, 2, 3],
            "fit": {"params": {"slope": 2.0}, "r2": 0.99},
            "capabilities": [{"name": "mobility", "applicable": True, "reason": "ok"}]}
    rows = flatten_rows(data)
    labels = [r[0] for r in rows]
    assert "probe" in labels and "thickness_m" in labels
    assert "points" not in labels
    assert "fit.slope" in labels and "fit.r2" in labels
    assert any(l.startswith("capability:mobility") for l in labels)

def test_output_panel_swaps_canvas_without_leak(qapp, vsm_path):
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm_ok(vsm_path)
    p = OutputPanel()
    p.show_result(res); first = p.last_figure
    p.show_result(res); second = p.last_figure
    assert first is not second

def test_output_panel_no_canvas_accumulation(qapp, vsm_path):
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm_ok(vsm_path)
    p = OutputPanel()
    for _ in range(5):
        p.show_result(res)
    n = len(p.findChildren(FigureCanvasQTAgg))
    assert n == 4 and n == len(p._cards)        # one canvas per backed VSM kind, no accumulation

def _teardown_order_spy(widget, rec):
    """Shadow hide/setParent on one widget instance to record explicit Python-level call order."""
    orig_hide, orig_set = widget.hide, widget.setParent
    def hide():
        rec.append("hide"); orig_hide()
    def setParent(p):
        rec.append(("setParent", p is None)); orig_set(p)
    widget.hide = hide; widget.setParent = setParent

def test_cards_hidden_before_orphaned_on_rerender(qapp, vsm_path):
    """Regression (owner 2026-07-09): _clear_cards called setParent(None) on cards still flagged
    visible. On macOS the cocoa reparent briefly realizes each orphan as its own top-level window
    -> figures flash one by one and steal activation from the main window. Offscreen Qt can't
    show native windows, so we assert the call contract verified live on macOS: an explicit
    hide() must precede setParent(None) for every torn-down card."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm_ok(vsm_path)
    p = OutputPanel()
    p.show()
    p.show_result(res)
    old = list(p._cards)
    recs = []
    for c in old:
        rec = []; recs.append(rec); _teardown_order_spy(c, rec)
    p.show_result(res)                       # immediate re-render, no event-loop turn between
    for rec in recs:
        assert ("setParent", True) in rec, f"card was not orphaned: {rec}"
        assert "hide" in rec and rec.index("hide") < rec.index(("setParent", True)), \
            f"card orphaned while visible (macOS window flash): {rec}"

def test_canvas_hidden_before_orphaned_on_card_rerender(qapp, vsm_path):
    """Same contract for PlotCard.render's old-canvas teardown (per-card spec/style changes):
    the canvas must be hidden before it is reparented to None."""
    from cryosweep_gui.output_panel import OutputPanel
    res = _vsm_ok(vsm_path)
    p = OutputPanel()
    p.show()
    p.show_result(res)
    card = p._cards[0]
    rec = []
    _teardown_order_spy(card.canvas, rec)
    card.render(card._results, card._style)   # what _on_spec_changed does
    assert ("setParent", True) in rec, f"canvas was not orphaned: {rec}"
    assert "hide" in rec and rec.index("hide") < rec.index(("setParent", True)), \
        f"canvas orphaned while visible (macOS window flash): {rec}"
