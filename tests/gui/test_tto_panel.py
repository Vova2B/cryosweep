import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pathlib                                                   # noqa: E402

import pytest                                                    # noqa: E402

from cryosweep_gui.inputs.base import build_panel                     # noqa: E402
import cryosweep_gui.inputs   # noqa: F401,E402  registers panels


def test_tto_panel_registered_and_contributes_nothing(qapp):
    p = build_panel("tto")
    assert p.probe_key == "tto"
    assert p.build_header_patch() == {}
    assert p.build_overrides() == {}
    assert p.get_state() == {}
    p.set_state({"anything": "ignored"})                          # must not raise
    assert p.get_state() == {}


def test_tto_panel_explains_where_the_geometry_comes_from(qapp):
    from PySide6.QtWidgets import QLabel
    p = build_panel("tto")
    texts = [w.text() for w in p.findChildren(QLabel)]
    assert "Sample geometry is read from the file header." in texts


def test_tto_tab_label():
    from cryosweep_gui.main_window import _LABELS
    assert _LABELS["tto"] == "Thermal Transport"


def test_capability_labels_are_friendly():
    from cryosweep_gui.output_panel import _CAP_LABELS
    assert _CAP_LABELS["thermal_conductivity"] == "κ(T)"
    assert _CAP_LABELS["seebeck"] == "Seebeck"
    assert _CAP_LABELS["wiedemann_franz"] == "Wiedemann-Franz"
    assert _CAP_LABELS["power_factor"] == "power factor"
    assert _CAP_LABELS["figure_of_merit"] == "ZT"
    assert _CAP_LABELS["rrr"] == "RRR"
    # Cross-probe asymmetry closed while we are here: resistivity.py:372 emits the capability
    # under the name "RRR", so without this entry that probe keeps showing its raw key.
    assert _CAP_LABELS["RRR"] == "RRR"


def test_deferred_capabilities_do_not_nag_in_the_hint_strip():
    # The four recognized-but-deferred capabilities are permanently inapplicable with
    # reason="deferred". They must be ADVISORY: on a probe with no user inputs (D4), telling
    # the user to "Set the required inputs in the panel at left" is a false remedy that would
    # show on every clean load, forever.
    from cryosweep_gui.output_panel import _capability_hint
    hint = _capability_hint(_tto_data())
    # The four DEFERRED stubs still never nag (that is what this test guards). The κ_ph fit
    # is a REAL capability now (E5): when it declines, its reason is exactly what the strip
    # exists to show, so it — and only it — appears here.
    assert hint == ("Not computed — κ_ph power law: "
                    "needs >=10 finite kappa_ph > 0 points below 10 K.")
    for word in ("Callaway", "callaway", "boundary", "diffusive", "field sweep"):
        assert word not in hint


def test_the_remedy_clause_is_dropped_on_a_probe_with_no_inputs():
    # The remedy was UNCONDITIONAL: on tto_norho_synth the strip told the user to "Set the
    # required inputs in the panel at left" directly above a panel whose only content is one
    # explanatory sentence. Every TTO reason is a property of the FILE, so on this probe the
    # diagnosis alone is the whole message.
    from cryosweep_gui.output_panel import _capability_hint
    hint = _capability_hint(_tto_data("tto_norho_synth"))
    assert hint.startswith("Not computed — Wiedemann-Franz: requires finite ρ > 0")
    assert hint.endswith("RRR: no zero-field ρ(T) ramp · κ_ph power law: "
                         "needs >=10 finite kappa_ph > 0 points below 10 K.")
    assert "panel at left" not in hint


def test_other_probes_keep_the_remedy_clause():
    # Additive only: the clause must be untouched for every probe that HAS input fields.
    from cryosweep_gui.output_panel import _capability_hint
    data = {"probe": "hall",
            "capabilities": [{"name": "mobility", "applicable": False,
                              "reason": "no longitudinal channel"}]}
    assert _capability_hint(data) == ("Not computed — mobility: no longitudinal channel."
                                      "   Set the required inputs in the panel at left.")


def test_window_builds_a_tto_tab(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    labels = {win.tabs.widget(i).probe: win.tabs.tabText(i) for i in range(win.tabs.count())}
    # probe-key membership alone was already true before the tab existed (Task 3's registry);
    # the visible tab TEXT is what this test is actually for.
    assert labels.get("tto") == "Thermal Transport"


def _tto_data(fixture="tto_synth"):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    from cryosweep_core.config import RunConfig
    from cryosweep_core.io.loader import load_dat
    return TTOAnalyzer().analyze(load_dat(str(
        pathlib.Path(f"tests/core/fixtures/{fixture}.dat"))), RunConfig()).data


def test_flatten_rows_reports_per_curve_and_scalar_quantities():
    # Values, not key existence: 6 of 12 mutants survived the old `any(... in k)` checks —
    # including κ@Tmin <-> κ@Tmax swapped and S@Tmax reporting seebeck[0] (0.02), both of which
    # silently invert the T-ascending assumption documented at output_panel.py:142.
    from cryosweep_gui.output_panel import flatten_rows
    rows = dict(flatten_rows(_tto_data()))
    assert rows["0Oe,down.n"] == "150"                            # curve tag carries field+direction
    assert rows["9e+04Oe,down.n"] == "30"
    assert rows["0Oe,down.κ@Tmin/Tmax"] == "5.61 / 74.3"          # T_min FIRST (curves ascend in T)
    assert rows["0Oe,down.S@Tmax"] == "3 µV/K"                    # seebeck[-1]; seebeck[0] is 0.02
    assert rows["classification"] == "metallic"
    assert rows["RRR"] == "8.373 ± 0.066"
    # The synth's ZT rises with T, so its maximum is the last measured point — the row label
    # has to say so rather than claim an observed peak (I3).
    # was: assert rows["ZT peak (at T range edge)"] == "0.0003634 @ 300.0 K"
    assert rows["ZT peak (at T range edge)"].startswith("0.0003634 ± ")   # value @ T, not T @ T
    assert rows["ZT peak (at T range edge)"].endswith(" @ 300.0 K")
    assert "ZT peak" not in rows
    assert rows["PF @ T_high"].endswith("W/(K²·m)")                # spec oracle scalar, surfaced
    # The error-row count is shown ONCE: the generic scalar walker already emits it, and the
    # TTO branch used to add a second "error rows" row carrying the same number.
    assert rows["n_error_rows"] == "3"
    assert "error rows" not in rows


def test_flatten_rows_emits_nothing_tto_specific_for_another_probe():
    # `if data.get("probe") == "tto"` -> `if True` survived: TTO rows then leaked into every
    # other probe's table.
    from cryosweep_gui.output_panel import flatten_rows
    rows = dict(flatten_rows({"probe": "resistivity", "n_error_rows": 7,
                              "rrr": {"rrr": 2.0, "classification": "metallic"},
                              "summary": {"pf_at_thigh": 1e-6, "zt_peak": 1e-4,
                                          "zt_peak_t_k": 300.0}}))
    assert "PF @ T_high" not in rows
    assert "RRR" not in rows and "classification" not in rows
    assert not any(k.startswith("ZT peak") for k in rows)


def test_flatten_rows_survives_a_result_with_no_seebeck():
    from cryosweep_gui.output_panel import flatten_rows
    rows = dict(flatten_rows(_tto_data("tto_gap_synth")))          # must not raise
    assert not any(".S@Tmax" in k for k in rows)
    assert rows["0Oe,down.n"] == "147"                             # the gapped curve is shorter


def test_flatten_rows_survives_a_result_with_no_rrr_and_no_zt():
    # tto_gap_synth HAS rrr (8.37, metallic) and a ZT peak, so it never reached the
    # `rrr is None` / `zt_peak is None` branches. tto_norho_synth is the fixture with both.
    from cryosweep_gui.output_panel import flatten_rows
    d = _tto_data("tto_norho_synth")
    assert d.get("rrr") is None and (d.get("summary") or {}).get("zt_peak") is None
    rows = dict(flatten_rows(d))                                  # must not raise
    assert "RRR" not in rows and "classification" not in rows
    assert not any(k.startswith("ZT peak") for k in rows)
    assert rows["n_error_rows"] == "0"


def test_real_file_loads_into_the_tto_tab(qapp, tto_real_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.load_path(str(tto_real_path))
    assert win.tabs.currentWidget().probe == "tto"
    assert "tto" in win.chip.text().lower()


def test_the_new_capability_is_not_advisory_so_its_reason_reaches_the_hint_strip():
    # E5: boundary_scattering_fit stays deferred/advisory; this one is a REAL capability now.
    from cryosweep_gui.output_panel import _CAP_ADVISORY
    assert "kappa_ph_power_fit" not in _CAP_ADVISORY
    assert "boundary_scattering_fit" in _CAP_ADVISORY


def test_capability_label_for_the_new_fit():
    from cryosweep_gui.output_panel import _CAP_LABELS
    assert _CAP_LABELS["kappa_ph_power_fit"] == "κ_ph power law"


def test_kappa_ph_row_is_absent_when_the_fit_declined():
    from cryosweep_gui.output_panel import flatten_rows
    rows = dict(flatten_rows(_tto_data()))              # tto_synth: only 5 pts below 10 K
    assert "κ_ph ~ T^n" not in rows


def _ladder2():
    return [{"cutoff_k": 10.0, "method": "curve_fit", "n": 2.03,
             "sigma": 0.006, "r2": 0.99, "n_points": 163},
            {"cutoff_k": 30.0, "method": "curve_fit", "n": 1.31,
             "sigma": 0.008, "r2": 0.99, "n_points": 543}]


def test_kappa_ph_row_never_writes_n_with_a_plus_minus_glyph():
    # Task-8 review: "±" is the universal glyph for "uncertainty on the preceding number", and
    # a "(stat)" two tokens later does not override it -- the eye binds "2.03 ± 0.0062" as a unit.
    # The stat sigma is therefore NAMED (σ_stat) and moved LAST, behind the sensitivities that
    # actually dominate it. Pinned on the flag-carrying AND the flag-free shape.
    from cryosweep_gui.output_panel import _kappa_ph_row, flatten_rows
    sensitive = _kappa_ph_row({"n": 2.0266, "n_sigma": 0.0062, "n_spread": 0.71,
                               "n_loglog": 2.01, "n_method_delta": 0.02,
                               "window_k": [2.16, 10.0], "ladder": _ladder2(),
                               "quality_flags": ["window_sensitive"]})
    assert "±" not in sensitive
    assert "±" not in dict(flatten_rows(_tto_data("tto_powerlaw_synth")))["κ_ph ~ T^n"]
    # the sigma clause is LAST and says what it is NOT
    assert sensitive.endswith(
        "σ_stat 0.0062 (fit scatter only, not the uncertainty on n)")


def test_kappa_ph_row_names_the_window_sensitive_flag_and_rounds_the_headline():
    # (b) + (c): the analyzer COMPUTES window_sensitive and the CSV carries it, so the GUI row
    # must say the word; and when it is set the headline is rounded to the determined precision
    # (n_spread 0.71 -> three significant digits on n is a fiction).
    from cryosweep_gui.output_panel import _kappa_ph_row
    v = _kappa_ph_row({"n": 2.0266, "n_sigma": 0.0062, "n_spread": 0.71, "n_loglog": 2.01,
                       "n_method_delta": 0.02, "window_k": [2.16, 10.0],
                       "ladder": _ladder2(), "quality_flags": ["window_sensitive"]})
    assert v == ("n \u2248 2.0 (\u226410 K fit) \u2014 WINDOW-SENSITIVE: n(10\u219230 K) = 2.03\u21921.31; "
                 "log-log 2.01; flags: window_sensitive; "
                 "\u03c3_stat 0.0062 (fit scatter only, not the uncertainty on n)")


def test_kappa_ph_row_omits_the_window_sensitive_marker_when_the_flag_is_absent():
    # The flag-ABSENT variant. tto_powerlaw_synth is an exact power law -> quality_flags == [],
    # so no "WINDOW-SENSITIVE" marker anywhere and the headline keeps 3 significant digits.
    from cryosweep_gui.output_panel import _kappa_ph_row, flatten_rows
    real = dict(flatten_rows(_tto_data("tto_powerlaw_synth")))["\u03ba_ph ~ T^n"]
    assert "WINDOW-SENSITIVE" not in real and "\u2248" not in real
    assert real.startswith("n = 3 (\u226410 K fit); n(10\u219230 K) = 3\u21923; log-log 3; \u03c3_stat ")
    # same shape from an in-memory fit carrying no quality_flags key at all
    v = _kappa_ph_row({"n": 2.0266, "n_sigma": 0.0062, "n_spread": 0.71, "n_loglog": 2.01,
                       "n_method_delta": 0.02, "window_k": [2.16, 10.0],
                       "ladder": _ladder2()})
    assert v == ("n = 2.03 (\u226410 K fit); n(10\u219230 K) = 2.03\u21921.31; log-log 2.01; "
                 "\u03c3_stat 0.0062 (fit scatter only, not the uncertainty on n)")


def test_kappa_ph_row_spells_out_EVERY_quality_flag_not_only_window_sensitive():
    """Final-review C1(a). `window_sensitive` was the only flag that reached the screen, so a
    fit carrying `n_at_bound` / `kappa_e_dominant` read as a clean measurement on the one
    surface a user actually looks at. The clause is verbatim and in the analyzer's order, so it
    is directly comparable with the CSV's `kappa_ph_flags` cell."""
    from cryosweep_gui.output_panel import _kappa_ph_row
    v = _kappa_ph_row({"n": 2.0266, "n_sigma": 0.0062, "n_spread": 0.71, "n_loglog": 2.01,
                       "n_method_delta": 0.02, "window_k": [2.16, 10.0],
                       "ladder": _ladder2(),
                       "quality_flags": ["window_sensitive", "kappa_e_dominant"]})
    assert "flags: window_sensitive, kappa_e_dominant" in v
    # ... and the sigma caveat is STILL last (the Task-8 rule the flags clause must not displace)
    assert v.endswith("σ_stat 0.0062 (fit scatter only, not the uncertainty on n)")
    # a flag with no other surface at all
    only = _kappa_ph_row({"n": 3.0, "n_sigma": 0.01, "n_spread": None, "n_loglog": None,
                          "n_method_delta": None, "window_k": [2.0, 10.0],
                          "ladder": [{"cutoff_k": 10.0, "method": "curve_fit", "n": 3.0,
                                      "sigma": 0.01, "r2": 0.99, "n_points": 40}],
                          "quality_flags": ["ladder_incomplete"]})
    assert "flags: ladder_incomplete" in only


def test_a_declined_bound_pinned_fit_shows_no_row_and_states_why_in_the_hint_strip():
    """Final-review C1(b), end to end on the GUI surfaces: tto_deltat_synth's kappa_ph is flat
    below 10 K, so the exponent used to arrive as `n = 0.5 (≤10 K fit); n(10→30 K) = 0.5→0.5`
    — an assertion of PERFECT window stability produced by a degenerate fit. There must now be
    no κ_ph row at all, and the strip must say what happened."""
    from cryosweep_gui.output_panel import flatten_rows, _capability_hint
    d = _tto_data("tto_deltat_synth")
    assert "κ_ph ~ T^n" not in dict(flatten_rows(d))
    hint = _capability_hint(d)
    assert "κ_ph power law: kappa_ph is not a power law below 10 K" in hint
    assert "pinned at the search bound (n = 0.5)" in hint
    assert "worse than a constant (r2 = -3.57e+13)" in hint
    assert "0.5→0.5" not in hint


def test_kappa_ph_row_says_not_measured_when_the_spread_was_never_measured():
    from cryosweep_gui.output_panel import _kappa_ph_row
    v = _kappa_ph_row({"n": 2.03, "n_sigma": 0.006, "n_spread": None, "n_loglog": 2.01,
                       "n_method_delta": 0.02, "window_k": [2.16, 10.0], "ladder": [
                           {"cutoff_k": 10.0, "method": "curve_fit", "n": 2.03,
                            "sigma": 0.006, "r2": 0.99, "n_points": 163}]})
    assert "n(window spread) = not measured" in v
    assert v.endswith("\u03c3_stat 0.006 (fit scatter only, not the uncertainty on n)")


def test_kappa_ph_row_omits_the_loglog_clause_entirely_when_it_is_none():
    from cryosweep_gui.output_panel import _kappa_ph_row
    v = _kappa_ph_row({"n": 2.03, "n_sigma": 0.006, "n_spread": 0.71, "n_loglog": None,
                       "n_method_delta": None, "window_k": [2.16, 10.0],
                       "ladder": _ladder2()})
    assert "log-log" not in v
    assert v == ("n = 2.03 (\u226410 K fit); n(10\u219230 K) = 2.03\u21921.31; "
                 "\u03c3_stat 0.006 (fit scatter only, not the uncertainty on n)")


def test_rrr_row_has_no_sigma_clause_when_rrr_std_is_absent():
    from cryosweep_gui.output_panel import flatten_rows
    rows = dict(flatten_rows({"probe": "tto", "curves": [],
                              "rrr": {"rrr": 2.0, "classification": "metallic",
                                      "rrr_std": None}}))
    assert rows["RRR"] == "2"


def test_error_band_checkbox_is_shown_only_for_band_carrying_tto_kinds(qapp, hc_path):
    from PySide6.QtWidgets import QCheckBox
    from cryosweep_core.plotting.catalog import get_kind
    from cryosweep_core.plotting.spec import PlotSpec
    from cryosweep_gui.plot_controls import AxisStrip

    def _boxes(kind_key):
        kind = get_kind(kind_key)
        series = kind.series(_result(kind.probe, hc_path))
        strip = AxisStrip(series, PlotSpec(), kind)
        return [cb.text() for cb in strip.findChildren(QCheckBox)]

    assert "Error band" in _boxes("tto_kappa_t")
    assert "Error band" in _boxes("tto_summary_t")
    assert "Error band" not in _boxes("tto_wf_t")          # not in _TTO_BAND_KINDS
    assert "Error band" not in _boxes("cp_vs_t")           # not a TTO kind at all


def test_the_result_table_shows_the_whole_kappa_ph_row_instead_of_truncating_it(qapp):
    """Task-8 review, IMPORTANT 1. The table's default 100 px columns cut the kappa_ph row down
    to "n = 2.03 ± ..." on screen (measured: 371 px of text in a 100 px column, with ~550 px of
    the widget empty to its right) -- the headline number with every corrective clause removed,
    i.e. EXACTLY the over-claim this slice exists to prevent. Task 8's own report read
    item.text() and so never saw it. This test measures what is RENDERABLE, not what is stored.
    """
    from PySide6.QtWidgets import QHeaderView
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    from cryosweep_core.config import RunConfig
    from cryosweep_core.io.loader import load_dat
    from cryosweep_gui.output_panel import OutputPanel
    res = TTOAnalyzer().analyze(
        load_dat("tests/core/fixtures/tto_real_subset.dat"), RunConfig())
    panel = OutputPanel()
    panel.resize(1000, 800)
    panel.show_result(res)
    tbl = panel.table
    hh = tbl.horizontalHeader()
    assert hh.sectionResizeMode(0) == QHeaderView.ResizeMode.ResizeToContents
    assert hh.sectionResizeMode(1) == QHeaderView.ResizeMode.Stretch
    hits = [r for r in range(tbl.rowCount())
            if tbl.item(r, 0) and tbl.item(r, 0).text() == "\u03ba_ph ~ T^n"]
    assert hits, "the fixture must produce a kappa_ph row for this test to mean anything"
    item = tbl.item(hits[0], 1)
    # (1) every corrective clause survives ...
    for clause in ("WINDOW-SENSITIVE", "log-log", "\u03c3_stat"):
        assert clause in item.text()
    # (2) ... and is actually DRAWABLE in the cell: the text wrapped to the column's width needs
    #     no more height than the row was given. (Unwrapped it is 809 px of text -- wider than
    #     the stretched column at any ordinary window size, which is why word wrap is on.)
    from PySide6.QtCore import QRect, Qt
    idx = tbl.model().index(hits[0], 1)
    fm = tbl.fontMetrics()
    box = fm.boundingRect(QRect(0, 0, tbl.columnWidth(1) - 8, 10_000),
                          int(Qt.TextFlag.TextWordWrap), item.text())
    assert box.height() <= tbl.rowHeight(hits[0]), (box.height(), tbl.rowHeight(hits[0]))
    assert tbl.visualRect(idx).width() >= tbl.columnWidth(1) - 1
    # (3) belt and braces for narrower windows: the full text is always in the tooltip
    assert item.toolTip() == item.text()


@pytest.mark.parametrize("width", [520, 600, 650, 700, 1100])
def test_the_kappa_ph_row_survives_a_NARROW_output_panel(qapp, width):
    """Final-review I2. The sibling of the truncation bug already fixed: row heights were
    recomputed on repopulate and on `sectionResized`, but the STRETCH column's final width is
    settled by the header after a VIEW resize, and `sectionResized` does not fire for that.
    Measured before the fix (real file, panel resized 1100 -> 520 -> 600 -> 650 px): the row
    height was one resize STALE at every step -- 31 px (two lines) where the text needed 3.7 --
    so `log-log 2.01` and the whole "fit scatter only, not the uncertainty on n" caveat were
    elided. The default main window is 1100 px, i.e. right at the boundary out of the box.

    A caveat that only survives at wide windows is not a caveat, so this asserts DRAWABILITY at
    every width, not merely that the string is stored."""
    from PySide6.QtCore import QRect, Qt
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    from cryosweep_core.config import RunConfig
    from cryosweep_core.io.loader import load_dat
    from cryosweep_gui.output_panel import OutputPanel
    res = TTOAnalyzer().analyze(
        load_dat("tests/core/fixtures/tto_real_subset.dat"), RunConfig())
    panel = OutputPanel()
    panel.resize(1100, 700)
    panel.show()          # WITHOUT this the layout never activates and the table keeps a
    panel.show_result(res)  # constant 430 px value column -- i.e. the test would be vacuous
    panel.resize(width, 700)
    qapp.processEvents()
    tbl = panel.table
    assert tbl.textElideMode() == Qt.TextElideMode.ElideNone
    # non-vacuity: the value column really does follow the panel width
    assert tbl.columnWidth(1) == pytest.approx(width - 271, abs=40), tbl.columnWidth(1)
    row = [r for r in range(tbl.rowCount())
           if tbl.item(r, 0) and tbl.item(r, 0).text() == "κ_ph ~ T^n"][0]
    item = tbl.item(row, 1)
    assert item.text().endswith("(fit scatter only, not the uncertainty on n)")
    box = tbl.fontMetrics().boundingRect(
        QRect(0, 0, tbl.columnWidth(1) - 8, 10_000),
        int(Qt.TextFlag.TextWordWrap), item.text())
    assert box.height() <= tbl.rowHeight(row), (width, box.height(), tbl.rowHeight(row))
    # non-vacuous at the narrow end: the row really does need more than the two lines the old
    # code capped it at.
    if width <= 650:
        assert box.height() > 2 * tbl.fontMetrics().height()


def test_error_band_checkbox_widget_is_actually_wired_to_the_setter(qapp):
    # Task-8 review, IMPORTANT 4: deleting `self._error_band_cb.toggled.connect(...)` passed
    # BOTH suites -- the existing tests are existence-only or call set_error_band() directly, so
    # the feature could silently become unreachable again (the original spec-review finding).
    # This drives the WIDGET the user actually clicks.
    from PySide6.QtWidgets import QCheckBox
    from cryosweep_core.plotting.catalog import get_kind
    from cryosweep_core.plotting.spec import PlotSpec
    from cryosweep_gui.plot_controls import AxisStrip
    kind = get_kind("tto_kappa_t")
    spec = PlotSpec()
    strip = AxisStrip(kind.series(_result("tto")), spec, kind)
    cb = [c for c in strip.findChildren(QCheckBox) if c.text() == "Error band"]
    assert len(cb) == 1
    seen = []
    strip.spec_changed.connect(lambda: seen.append(True))
    assert spec.error_band is not True             # default-OFF
    cb[0].setChecked(True)
    assert spec.error_band is True and seen == [True]
    cb[0].setChecked(False)
    assert spec.error_band is False and len(seen) == 2


def test_error_band_checkbox_commits_to_the_spec_and_emits(qapp):
    from cryosweep_core.plotting.catalog import get_kind
    from cryosweep_core.plotting.spec import PlotSpec
    from cryosweep_gui.plot_controls import AxisStrip
    kind = get_kind("tto_kappa_t")
    spec = PlotSpec()
    strip = AxisStrip(kind.series(_result("tto")), spec, kind)
    seen = []
    strip.spec_changed.connect(lambda: seen.append(True))
    strip.set_error_band(True)
    assert spec.error_band is True and seen == [True]
    strip.set_error_band(False)
    assert spec.error_band is False and len(seen) == 2


def _result(probe, path=None):
    """An analyzed Result for a probe, for building an AxisStrip with real series."""
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    from cryosweep_core.config import RunConfig
    from cryosweep_core.io.loader import load_dat
    if probe == "tto":
        return TTOAnalyzer().analyze(load_dat("tests/core/fixtures/tto_synth.dat"),
                                     RunConfig())
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    return analyze_file(load_dat(str(path)), RunConfig(), build_default_registry())
