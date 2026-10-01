"""Curie-Weiss fit windows in the VSM tab.

The panel carries a Curie-Weiss window, a separate modified-CW window and a "Curve to"
limit. Empty boxes mean the default (no window), the state round-trips, and a re-analysis
fits inside the windows. The output table names the window on the reported theta and shows
the modified fit (its parameters, or why there are none).
"""
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pathlib

import pytest

from cryosweep_gui.inputs.vsm import VSMInputPanel
from cryosweep_gui.output_panel import flatten_rows

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
VSM_EXAMPLE = EXAMPLES / "magnetization_vsm.dat"


def test_empty_boxes_contribute_nothing(qapp):
    assert VSMInputPanel().build_overrides() == {}


def test_windows_reach_the_config(qapp):
    p = VSMInputPanel()
    p.cw_min.setText("150"); p.cw_max.setText("280")
    p.cw_mod_min.setText("50"); p.cw_mod_max.setText("abc")      # non-numeric -> omitted
    p.cw_curve_max.setText("350")
    assert p.build_overrides() == {"vsm": {"cw_fit_min_k": 150.0, "cw_fit_max_k": 280.0,
                                           "cw_mod_fit_min_k": 50.0,
                                           "cw_curve_max_k": 350.0}}


def test_state_round_trips(qapp):
    p = VSMInputPanel()
    p.molar_mass_edit.setText("200"); p.cw_min.setText("150"); p.cw_mod_max.setText("300")
    p.cw_curve_max.setText("400")
    q = VSMInputPanel(); q.set_state(p.get_state())
    assert q.get_state() == p.get_state()
    assert q.cw_min.text() == "150" and q.cw_mod_max.text() == "300"
    VSMInputPanel().set_state({"molar_mass": "1", "mass_mg": "2"})   # older saved state


def _vsm_tab(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.state.load(str(VSM_EXAMPLE))
    win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.set_files([str(VSM_EXAMPLE)])
    return win, tab


def test_reanalysis_fits_inside_the_windows(qapp):
    win, tab = _vsm_tab(qapp)
    tab.panel.cw_min.setText("150")
    tab.panel.cw_mod_min.setText("50"); tab.panel.cw_mod_max.setText("200")
    d = tab.analyze().data
    assert d["fit"]["fit_range"][0] >= 150.0
    lo, hi = d["fit_modified"]["fit_range"]
    assert lo >= 50.0 and hi <= 200.0
    assert d["fit_curve"]["t_grid"][-1] == max(d["temperature"])


# ---------------- output rows ----------------

def _data(window=(None, None), flags=(), fit_modified=True, sensitive=True):
    d = {"probe": "vsm",
         "fit": {"params": {"C": 1.5, "theta": -50.27, "mu_eff": 4.499},
                 "sigma": {"C": 0.01, "theta": 0.99, "mu_eff": 0.01}, "r2": 0.9965,
                 "quality_flags": ["window_sensitive"] if sensitive else []},
         "cw_ladder": [
             {"tmin_k": 200.0, "theta_k": -37.55, "sigma_theta_k": 0.6, "mu_eff": 4.392,
              "sigma_mu_eff": 0.005, "r2": 0.9998, "n_points": 50},
             {"tmin_k": 250.0, "theta_k": -36.0, "sigma_theta_k": 0.9, "mu_eff": 4.39,
              "sigma_mu_eff": 0.006, "r2": 0.9997, "n_points": 30}],
         "theta_spread_k": 14.3,
         "fit_curve": {"window_k": list(window)},
         "fit_modified": None,
         "fit_modified_curve": None}
    if fit_modified:
        d["fit_modified"] = {"params": {"C": 4.6, "theta": -112.2, "chi0": -0.0059,
                                        "mu_eff": 6.09},
                             "units": {"C": "emu*K/(mol*Oe)", "theta": "K",
                                       "chi0": "emu/(mol*Oe)", "mu_eff": "mu_B"},
                             "r2": 0.9675, "fit_range": [3.0, 297.9],
                             "quality_flags": list(flags)}
        d["fit_modified_curve"] = {"window_k": [None, None], "zero_crossing": not flags,
                                   "reason": flags[0] if flags else None}
    return d


def test_headline_names_the_window_when_one_is_set():
    rows = dict(flatten_rows(_data(window=(150.0, None))))
    assert rows["Curie-Weiss θ"].startswith("θ = -50.3 K (fit window T ≥ 150 K — REPORTED)")
    rows = dict(flatten_rows(_data(window=(150.0, 280.0))))
    assert rows["Curie-Weiss θ"].startswith("θ = -50.3 K (fit window 150–280 K — REPORTED)")
    rows = dict(flatten_rows(_data()))
    assert rows["Curie-Weiss θ"].startswith("θ = -50.3 K (full-window fit — REPORTED)")


def test_modified_row_shows_parameters_and_window():
    row = dict(flatten_rows(_data()))["Modified Curie-Weiss"]
    assert "θ = -112 K" in row.replace("−", "-") and "χ₀ = -0.0059" in row.replace("−", "-")
    assert "fitted 3–298 K" in row


def test_modified_row_spells_out_a_flag():
    row = dict(flatten_rows(_data(flags=("theta_out_of_range",))))["Modified Curie-Weiss"]
    assert "flags: theta_out_of_range" in row
    assert "window only" in row


def test_modified_row_says_when_there_is_no_fit():
    row = dict(flatten_rows(_data(fit_modified=False)))["Modified Curie-Weiss"]
    assert row.startswith("not fitted")


def test_cw_row_says_when_the_window_lost_the_fit():
    d = _data(window=(299.5, None))
    d["fit"] = None
    d["fit_curve"] = None
    d["cw_ladder"] = None
    rows = dict(flatten_rows(d))
    assert rows["Curie-Weiss θ"].startswith("not fitted")


def test_results_without_curve_keys_get_no_new_rows():
    d = _data()
    for k in ("fit_curve", "fit_modified_curve", "fit_modified"):
        d.pop(k)
    rows = dict(flatten_rows(d))
    assert "Modified Curie-Weiss" not in rows
    assert rows["Curie-Weiss θ"].startswith("θ = -50.3 K (full-window fit — REPORTED)")


def test_a_window_on_a_clean_fit_is_still_named():
    d = _data(window=(150.0, None), sensitive=False)
    d["fit"]["fit_range"] = [150.3, 300.0]
    d["fit"]["n_points"] = 150
    rows = dict(flatten_rows(d))
    assert "Curie-Weiss θ" not in rows                       # not window-sensitive
    assert rows["Curie-Weiss window"] == "fit window T ≥ 150 K; fitted 150–300 K (150 points)"
    assert "Curie-Weiss window" not in dict(flatten_rows(_data(sensitive=False)))


def test_none_curve_keys_make_no_raw_rows():
    d = _data(fit_modified=False)
    d["fit_curve"] = None
    keys = [k for k, _ in flatten_rows(d)]
    assert "fit_curve" not in keys and "fit_modified_curve" not in keys


def test_declined_modified_parameters_read_as_declined():
    d = _data(flags=("chi0_unresolved",))
    d["fit_modified"]["params"]["chi0"] = None
    d["fit_modified"]["sigma"] = {"chi0": None}
    row = dict(flatten_rows(d))["Modified Curie-Weiss"]
    assert "χ₀ unresolved" in row and "flags: chi0_unresolved" in row
    d = _data(flags=("C_unresolved", "theta_unresolved"))
    d["fit_modified"]["params"].update(C=None, theta=None, mu_eff=None)
    row = dict(flatten_rows(d))["Modified Curie-Weiss"]
    assert row.startswith("declined") and "C_unresolved" in row and "None" not in row


def test_declined_plain_parameter_is_not_printed_as_none():
    d = _data()
    d["fit"]["params"]["mu_eff"] = None
    assert dict(flatten_rows(d))["fit.mu_eff"] != "None"


def test_inverse_chi_checklist_lists_no_fit_entries(qapp):
    """The stored fit curves are drawn by the fit-line toggles, not the curve checklist; an
    entry there would do nothing when unticked."""
    win, tab = _vsm_tab(qapp)
    tab.analyze_and_render()
    card = next(c for c in tab.output._cards if c.entry.kind == "inverse_chi")
    from cryosweep_gui.plot_controls import _KEY_ROLE
    keys = [it.data(_KEY_ROLE) for it in card.strip.checklist._items()]
    assert "cw_fit" not in keys and "cw_modified_fit" not in keys
    assert keys                                          # the data curve is still listed


def test_overlay_checklist_lists_no_fit_entries(qapp):
    from cryosweep_core.plotting.catalog import OverlayFile, get_kind, overlay_series
    win, tab = _vsm_tab(qapp)
    r = tab.analyze()
    ss = overlay_series(get_kind("inverse_chi"), [r, r],
                        [OverlayFile(0, "a"), OverlayFile(1, "b")])
    assert {s.key for s in ss if s.role == "fit"} == {"0::cw_fit", "0::cw_modified_fit",
                                                       "1::cw_fit", "1::cw_modified_fit"}
    from cryosweep_gui.output_panel import _CHECKLIST_HIDES_FIT
    assert "inverse_chi" in _CHECKLIST_HIDES_FIT
