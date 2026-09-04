"""ROADMAP item 2 — Debye-Einstein parameters, both directions.

(a) After the 7-parameter fit runs, the parameter boxes show what was fitted — and the
    fitted values survive the `set_state` restore at the end of `analyze_and_render`.
(b) Editing a parameter draws a live dashed "model (manual)" curve: a model evaluation,
    never a refit. The fitted curve and every export from the analysis result stay
    untouched — a hand-set value must never be presented as a fit.
"""
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pathlib
from types import SimpleNamespace

import numpy as np
import pytest

from cryosweep_gui.inputs.hc import HCInputPanel, _FULL_KEYS
from cryosweep_core.fitting.heat_capacity import specific_heat_full

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
HC_EXAMPLE = EXAMPLES / "heat_capacity.dat"

_FITTED = {"theta_D": 226.7, "n": 3.0, "gamma": 0.0098,
           "theta_E1": 114.7, "theta_E2": 432.7, "m1": 0.5, "m2": 1.5}


def _result_with_fit(ok=True, t_grid=None):
    ff = {"ok": ok, "params": dict(_FITTED), "r2": 0.9999,
          "t_grid": list(t_grid) if t_grid is not None else [2.0, 150.0, 300.0],
          "cp_fit": [0.0, 0.0, 0.0]}
    return SimpleNamespace(data={"probe": "heatcapacity", "full_fit": ff,
                                 "full_temperature": [2.0, 300.0]})


# ---------------- panel-level (a) ----------------

def test_absorb_result_writes_fitted_values_quietly(qapp):
    p = HCInputPanel()
    fired = []
    p.param_edited.connect(lambda: fired.append(1))
    assert p.absorb_result(_result_with_fit()) is True
    for k in _FULL_KEYS:
        assert p._val[k].value() == pytest.approx(_FITTED[k])
    assert not fired, "absorbing a fit is not a user edit; it must not draw a manual curve"


def test_absorb_declined_fit_leaves_guesses(qapp):
    p = HCInputPanel()
    before = {k: p._val[k].value() for k in _FULL_KEYS}
    assert p.absorb_result(_result_with_fit(ok=False)) is False
    assert p.absorb_result(SimpleNamespace(data={})) is False
    assert p.absorb_result(None) is False
    assert {k: p._val[k].value() for k in _FULL_KEYS} == before


def test_fitted_state_patch_updates_val_only(qapp):
    p = HCInputPanel()
    state = p.get_state()
    state["n_atoms"] = "3"
    patched = p.fitted_state_patch(state, _result_with_fit())
    assert patched is not None
    assert patched["val"]["theta_D"] == pytest.approx(226.7)
    assert patched["n_atoms"] == "3"                      # everything else preserved
    assert state["val"]["theta_D"] != pytest.approx(226.7)  # input state not mutated
    assert p.fitted_state_patch(state, _result_with_fit(ok=False)) is None


def test_param_edited_fires_on_user_edit_not_on_set_state(qapp):
    p = HCInputPanel()
    fired = []
    p.param_edited.connect(lambda: fired.append(1))
    st = p.get_state(); st["val"]["theta_D"] = 250.0
    p.set_state(st)
    assert not fired, "set_state is programmatic, not a user edit"
    p._val["theta_D"].setValue(300.0)                     # a user edit
    assert fired


# ---------------- panel-level (b): the model evaluation ----------------

def test_manual_model_curve_is_model_at_box_values(qapp):
    p = HCInputPanel()
    grid = np.linspace(2.0, 300.0, 50)
    r = _result_with_fit(t_grid=grid)
    p._val["theta_D"].setValue(300.0)                     # hand-set, differs from the fit
    x, y, label = p.manual_model_curve(r)
    params = {k: p._val[k].value() for k in _FULL_KEYS}
    np.testing.assert_allclose(y, specific_heat_full(np.asarray(x), **params))
    assert "manual" in label and "fit" not in label.lower()


def test_manual_model_curve_without_accepted_fit_uses_data_range(qapp):
    p = HCInputPanel()
    r = SimpleNamespace(data={"probe": "heatcapacity",
                              "full_fit": {"ok": False, "t_grid": []},
                              "full_temperature": [2.0, 150.0, 300.0]})
    x, y, label = p.manual_model_curve(r)
    assert min(x) >= 2.0 and max(x) == pytest.approx(300.0)
    assert len(x) >= 50


def test_manual_model_curve_declines_on_invalid_params(qapp):
    p = HCInputPanel()
    p._val["theta_D"].setValue(-5.0)                      # unphysical
    assert p.manual_model_curve(_result_with_fit()) is None


# ---------------- tab-level: the set_state trap and the live curve ----------------

def _hc_tab(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.state.load(str(HC_EXAMPLE))               # raw for the analyze()/async path
    win.select_probe("heatcapacity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(HC_EXAMPLE)])
    return win, tab


def _card(tab, kind):
    return next((c for c in tab.output._cards if c.entry.kind == kind), None)


def _lines_by_gid(card, gid):
    return [ln for ln in card.figure.axes[0].get_lines() if ln.get_gid() == gid]


def test_refit_puts_fitted_values_in_boxes_and_stored_state(qapp):
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()                              # ends with the set_state restore
    ff = tab._files[0].result.data["full_fit"]
    assert ff["ok"] is True
    for k in _FULL_KEYS:
        assert tab.panel._val[k].value() == pytest.approx(ff["params"][k], abs=5e-6), k
        assert tab._files[0].state["val"][k] == pytest.approx(ff["params"][k], abs=5e-6), k


def test_reanalyze_active_absorbs_on_probe_select(qapp):
    # loading a file and landing on the HC tab runs the fit (MainWindow._reanalyze_active);
    # the boxes must show what was fitted there too, not only on an explicit refit
    win, tab = _hc_tab(qapp)                              # select_probe triggered the analysis
    ff = win.state.get_raw() and tab._last_result.data["full_fit"]
    assert ff and ff["ok"] is True
    assert tab.panel._val["theta_D"].value() == pytest.approx(ff["params"]["theta_D"], abs=5e-6)


def test_async_result_absorbed_into_boxes_and_state(qapp):
    win, tab = _hc_tab(qapp)
    res = tab.analyze()                                   # what the worker would deliver
    tab._on_analyzed(res)
    ff = res.data["full_fit"]
    assert tab.panel._val["theta_D"].value() == pytest.approx(ff["params"]["theta_D"], abs=5e-6)
    assert tab._files[0].state["val"]["theta_D"] == pytest.approx(ff["params"]["theta_D"], abs=5e-6)


def test_edit_draws_manual_curve_and_leaves_fit_untouched(qapp):
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()
    card = _card(tab, "hc_full_cp_t")
    assert card is not None and card.figure is not None
    n_fit_before = len(_lines_by_gid(card, "fit"))
    assert n_fit_before >= 1
    tab.panel._val["theta_D"].setValue(300.0)             # user edit
    manual = _lines_by_gid(card, "manual_model")
    assert len(manual) == 1
    ln = manual[0]
    assert ln.get_label() == "model (manual)"
    assert ln.get_linestyle() == "--"
    params = {k: tab.panel._val[k].value() for k in _FULL_KEYS}
    np.testing.assert_allclose(ln.get_ydata(),
                               specific_heat_full(np.asarray(ln.get_xdata(), float), **params))
    assert len(_lines_by_gid(card, "fit")) == n_fit_before   # fitted curve untouched
    legend = card.figure.axes[0].get_legend()
    assert legend is not None
    assert any(t.get_text() == "model (manual)" for t in legend.get_texts())
    # a second edit updates in place, it does not stack lines
    tab.panel._val["theta_D"].setValue(280.0)
    assert len(_lines_by_gid(card, "manual_model")) == 1


def test_export_render_from_result_never_carries_manual_curve(qapp):
    from cryosweep_core.plotting.render import render_kind
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()
    tab.panel._val["theta_D"].setValue(300.0)
    fig = render_kind([tab._files[0].result], "hc_full_cp_t",
                      _card(tab, "hc_full_cp_t").entry.spec, tab.controls.style)
    gids = [ln.get_gid() for ax in fig.axes for ln in ax.get_lines()]
    assert "manual_model" not in gids
    labels = [ln.get_label() for ax in fig.axes for ln in ax.get_lines()]
    assert not any("manual" in str(l) for l in labels)


def test_refit_after_edit_clears_manual_curve_and_reabsorbs(qapp):
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()
    tab.panel._val["theta_D"].setValue(300.0)
    assert _lines_by_gid(_card(tab, "hc_full_cp_t"), "manual_model")
    tab.analyze_and_render()                              # refit: edited values are the new init
    card = _card(tab, "hc_full_cp_t")
    assert not _lines_by_gid(card, "manual_model")
    ff = tab._files[0].result.data["full_fit"]
    assert ff["ok"] is True
    assert tab.panel._val["theta_D"].value() == pytest.approx(ff["params"]["theta_D"], abs=5e-6)


def test_focus_change_clears_manual_curve(qapp):
    win, tab = _hc_tab(qapp)
    tab.set_files([str(HC_EXAMPLE), str(HC_EXAMPLE)])
    tab.analyze_and_render()
    tab.panel._val["theta_D"].setValue(300.0)
    assert _lines_by_gid(_card(tab, "hc_full_cp_t"), "manual_model")
    tab.focus_file(1)
    assert not _lines_by_gid(_card(tab, "hc_full_cp_t"), "manual_model")


def test_overlay_entries_each_store_their_own_fit(qapp):
    win, tab = _hc_tab(qapp)
    tab.set_files([str(HC_EXAMPLE), str(HC_EXAMPLE)])
    tab.analyze_and_render()
    for e in tab._files:
        ff = e.result.data["full_fit"]
        assert ff["ok"] is True
        assert e.state["val"]["theta_D"] == pytest.approx(ff["params"]["theta_D"], abs=5e-6)
