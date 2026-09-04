"""TTO single plot kinds: series contracts, gating, legend folding, low-T inset (spec §4)."""
import matplotlib; matplotlib.use("Agg")       # noqa: E702
import pathlib

import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.plotting.catalog import (BUILTIN_PLOTKINDS, series_tto_kappa_t,
                                        series_tto_seebeck_t, series_tto_wf_t,
                                        series_tto_zt_t, _tto_label)
from cryosweep_core.plotting.render import _CONNECT_KINDS, render_kind
from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec

FX = pathlib.Path("tests/core/fixtures")
SINGLES = ("tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_wf_t")
# Component legend labels for tto_wf_t. Mathtext, not "κ_e"/"κ_ph": a literal underscore
# renders AS an underscore in the legend, which is wrong in a journal figure.
K, KE, KPH = r"$\kappa$", r"$\kappa_\mathrm{e}$", r"$\kappa_\mathrm{ph}$"


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def test_four_kinds_registered_for_the_tto_probe():
    kinds = {k.key: k for k in BUILTIN_PLOTKINDS}
    for key in SINGLES:
        assert kinds[key].probe == "tto"
        assert kinds[key].group_colored is True
    assert kinds["tto_kappa_t"].label == "κ vs T"
    assert kinds["tto_seebeck_t"].label == "S vs T"
    assert kinds["tto_zt_t"].label == "ZT vs T"
    assert kinds["tto_wf_t"].label == "κ decomposition vs T"


def test_all_five_kinds_are_connect_kinds():
    for key in SINGLES + ("tto_summary_t",):
        assert key in _CONNECT_KINDS


def test_series_key_scheme_is_the_pinned_contract():
    s = series_tto_kappa_t(_run(FX / "tto_synth.dat"))
    keys = sorted(x.key for x in s)
    assert keys == ["kappa:0:down", "kappa:90000:down"]


def test_series_builders_default_field_unit_so_pq_compare_can_call_them():
    # pq_compare._render_v2 calls KINDS[kind].series(result) with NO field_unit.
    r = _run(FX / "tto_synth.dat")
    for fn in (series_tto_kappa_t, series_tto_seebeck_t, series_tto_zt_t, series_tto_wf_t):
        assert fn(r)                              # must not raise TypeError


def test_label_omits_a_sub_50_oe_field_but_shows_a_real_one():
    assert _tto_label({"field_oe": 0.077, "direction": "down"}) == "cooling"
    assert _tto_label({"field_oe": 90000.0, "direction": "down"}) == "90000 Oe, cooling"
    assert _tto_label({"field_oe": 90000.0, "direction": "down"}, "T") == "9 T, cooling"


def test_wf_series_carry_three_component_groups_and_linestyles():
    # SINGLE-field file: linestyle is the component's secondary cue (colour is primary).
    s = series_tto_wf_t(_run(FX / "tto_gap_synth.dat"))
    assert {x.group for x in s} == {K, KE, KPH}
    styles = {x.group: x.linestyle for x in s}
    assert styles == {K: "-", KE: "--", KPH: ":"}
    assert all(" · " in x.label for x in s)


def test_wf_linestyle_encodes_the_field_when_several_field_groups_are_present():
    # V3: on a multi-field file both field groups would otherwise share the three component
    # colours AND the direction marker, leaving the field unrecoverable. Colour MUST keep
    # meaning component (the brief's requirement); the field moves onto the linestyle.
    s = series_tto_wf_t(_run(FX / "tto_synth.dat"))
    assert {x.group for x in s} == {K, KE, KPH}          # colour = component, still
    by_field = {}
    for x in s:
        by_field.setdefault(x.key.split(":")[1], set()).add(x.linestyle)
    assert by_field == {"0": {"-"}, "90000": {"--"}}


def test_wf_series_empty_when_no_curve_carries_kappa_e():
    assert series_tto_wf_t(_run(FX / "tto_norho_synth.dat")) == []


def test_wf_series_empty_when_kappa_e_is_a_list_of_nulls():
    # F6: a non-empty list of Nones is TRUTHY, so a bare `any(c.get("kappa_e"))` would
    # advertise the kind and draw three all-NaN lines under a 3-entry legend. Not reachable
    # from today's analyzer (kappa_e is None or fully populated), so exercised in memory.
    r = _run(FX / "tto_synth.dat")
    d = dict(r.data)
    d["curves"] = [dict(c, kappa_e=[None] * len(c["t"]), kappa_ph=[None] * len(c["t"]))
                   for c in d["curves"]]
    assert series_tto_wf_t(r.model_copy(update={"data": d})) == []


def test_none_holes_become_nan_in_plot_space():
    # F7: the None -> NaN conversion in _tto_curve_series is unreachable from the fixtures
    # (no file has interior holes), so pin it on an in-memory result. NaN keeps the x/y
    # arrays aligned and breaks the connecting line at the hole instead of bridging it.
    import math
    r = _run(FX / "tto_synth.dat")
    d = dict(r.data)
    d["curves"] = [dict(d["curves"][0], kappa=[None] + list(d["curves"][0]["kappa"][1:]))]
    s = series_tto_kappa_t(r.model_copy(update={"data": d}))
    assert len(s) == 1 and len(s[0].y) == len(s[0].x)
    assert math.isnan(s[0].y[0]) and not math.isnan(s[0].y[1])


def test_seebeck_series_empty_on_the_gap_fixture_but_wf_still_renders():
    r = _run(FX / "tto_gap_synth.dat")
    assert series_tto_seebeck_t(r) == []
    assert series_tto_wf_t(r)
    render_kind(r, "tto_wf_t", PlotSpec(), GlobalStyle())   # must not raise


def test_empty_series_kind_renders_an_empty_figure_instead_of_raising():
    # Spec §6: tto_seebeck_t must still render on a Seebeck-less file, producing an
    # empty-series figure. (This deliberately diverges from _plot_data's ValueError.)
    fig = render_kind(_run(FX / "tto_gap_synth.dat"), "tto_seebeck_t", PlotSpec(),
                      GlobalStyle())
    ax = fig.axes[0]
    assert [ln for ln in ax.lines if ln.get_gid() is None] == []
    assert ax.get_ylabel() == "S (µV/K)"


def test_all_four_kinds_render_on_the_synth_fixture():
    r = _run(FX / "tto_synth.dat")
    for key in SINGLES:
        fig = render_kind(r, key, PlotSpec(), GlobalStyle())
        assert fig is not None and fig.axes


def test_kappa_kind_draws_a_low_t_inset_and_respects_the_gate():
    r = _run(FX / "tto_synth.dat")
    with_inset = render_kind(r, "tto_kappa_t", PlotSpec(), GlobalStyle())
    without = render_kind(r, "tto_kappa_t", PlotSpec(lowt_inset=False), GlobalStyle())
    assert len(with_inset.axes) == len(without.axes) + 1


def test_seebeck_kind_draws_the_zero_reference_line():
    # NOT contradictory with test_no_zero_refline_when_seebeck_never_crosses_zero (Task 7,
    # tests/core/test_tto_headline.py) even though both use this fixture, where S never
    # crosses zero: spec §4 gives the two kinds DIFFERENT rules on purpose — the standalone
    # `tto_seebeck_t` draws a HARD zero line unconditionally (S is a signed quantity and the
    # sign is the physics), while the stacked `tto_summary_t` panel draws one only when the
    # data actually crosses zero, to keep the cramped middle panel clean. Do not "fix" the
    # apparent inconsistency by aligning them.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_seebeck_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    assert any(ln.get_gid() == "refline" for ln in ax.lines)


def test_legend_is_drawn_even_for_a_single_curve(tto_real_path):
    fig = render_kind(_run(tto_real_path), "tto_kappa_t", PlotSpec(), GlobalStyle())
    leg = fig.axes[0].get_legend()
    assert leg is not None and len(leg.get_texts()) == 1
    assert leg.get_texts()[0].get_text() == "cooling"


def test_real_file_kappa_and_zt_full_ranges_stay_inside_the_axes(tto_real_path):
    # Robust-view clipping was a repeat defect in PQ-3/PQ-5; kappa spans 0.0368-4.289 and
    # ZT spans 2.2e-10-3.9e-4 on this file. Both must remain visible.
    #
    # kappa passes with _apply_robust_view UNCHANGED (measured: its robust top 14.51 exceeds
    # the data max, so the helper no-ops). ZT does NOT: _apply_robust_view narrows the top to
    # 3.5130e-05, 11x below the peak. That is why render_tto_zt_t calls _tto_full_view.
    # If this fails on tto_zt_t, the override was dropped — restore it. NEVER loosen these
    # bounds: they are the measured data extremes, and the peak IS the headline number.
    r = _run(tto_real_path)
    for key, lo, hi in (("tto_kappa_t", 0.0368108, 4.28858),
                        ("tto_zt_t", 2.21665e-10, 3.923224e-4)):
        ax = render_kind(r, key, PlotSpec(), GlobalStyle()).axes[0]
        bottom, top = ax.get_ylim()
        assert bottom <= lo and top >= hi, (key, bottom, top)


def test_zt_view_override_never_fights_an_explicit_user_limit():
    # _tto_full_view must honour the SAME bypass guards as _apply_robust_view, so it stays a
    # targeted correction rather than an unconditional ylim clamp. An explicit spec limit is
    # the observable one: the override would otherwise widen the axis back to the data range.
    # (The old robust_view=False variant of this test was vacuous — firing the override also
    # shows everything — so it only re-proved "not clipped".)
    ax = render_kind(_run(FX / "tto_synth.dat"), "tto_zt_t", PlotSpec(ymax=1e-4),
                     GlobalStyle()).axes[0]
    assert ax.get_ylim()[1] == 1e-4


def test_zt_view_override_leaves_autoscale_alone_when_robust_view_is_off():
    # The `not use` bypass: with the robust view off, matplotlib's own autoscale (data range
    # + a 5% margin) already shows everything, so the override must set NO ylim at all. The
    # y-limits alone cannot tell the two apart — they coincide numerically — but `set_ylim`
    # LATCHES the axis (autoscaley off), which is the observable difference.
    ax = render_kind(_run(FX / "tto_synth.dat"), "tto_zt_t", PlotSpec(robust_view=False),
                     GlobalStyle()).axes[0]
    assert ax.get_autoscaley_on() is True


def test_overlay_mode_still_labels_each_file():
    # Multi-file overlay (GUI Sub-project C, "Add to compare…"): `_plot_data`'s overlay branch
    # labels every line "<file label> · <series label>" (render.py:263), and `_tto_single` must
    # let `_finish` draw that standard per-file legend — the `_acms_single` convention
    # (render.py:2523 passes legend_handles=None with draw_legend at its default True). Without
    # it a TTO overlay renders N unlabelled curves with no way to tell the files apart.
    from cryosweep_core.plotting.catalog import OverlayFile
    r = _run(FX / "tto_synth.dat")
    fig = render_kind([r, r], "tto_kappa_t", PlotSpec(), GlobalStyle(),
                      overlay=[OverlayFile(0, "A"), OverlayFile(1, "B")])
    leg = fig.axes[0].get_legend()
    assert leg is not None
    # tto_synth has two curves (0 Oe -> "cooling", 90000 Oe -> "90000 Oe, cooling")
    assert {t.get_text() for t in leg.get_texts()} == {
        "A · cooling", "A · 90000 Oe, cooling",
        "B · cooling", "B · 90000 Oe, cooling"}


def test_wf_figure_draws_three_distinct_colours_on_the_canvas():
    # F1: asserting three distinct Series.group values only restates the implementation —
    # dropping the per-group colour lookup in _tto_draw (color=gcolor[s.group] -> "C0") left
    # the whole suite green while the figure collapsed to one colour, defeating the entire
    # reason tto_wf_t groups by component. Assert on the rendered ARTISTS instead.
    ax = render_kind(_run(FX / "tto_synth.dat"), "tto_wf_t", PlotSpec(), GlobalStyle()).axes[0]
    data_lines = [ln for ln in ax.lines if ln.get_gid() is None]
    assert len(data_lines) == 6                                  # 2 curves x 3 components
    assert len({ln.get_color() for ln in data_lines}) == 3


def test_wf_folded_legend_names_the_fields_on_a_multi_field_file():
    # V3: with the field on the linestyle, the folded legend must carry a proxy per field —
    # otherwise it lists only κ/κ_e/κ_ph and the field is still unrecoverable.
    leg = render_kind(_run(FX / "tto_synth.dat"), "tto_wf_t", PlotSpec(),
                      GlobalStyle()).axes[0].get_legend()
    texts = [t.get_text() for t in leg.get_texts()]
    assert texts[:3] == [K, KE, KPH]
    assert texts[3:] == ["0 Oe", "90000 Oe"]
    styles = {h.get_linestyle() for h in leg.legend_handles[3:]}
    assert len(styles) == 2


def test_wf_folded_legend_has_no_field_proxies_on_a_single_field_file():
    leg = render_kind(_run(FX / "tto_gap_synth.dat"), "tto_wf_t", PlotSpec(),
                      GlobalStyle()).axes[0].get_legend()
    assert [t.get_text() for t in leg.get_texts()] == [K, KE, KPH]


def test_lowt_inset_is_suppressed_when_the_window_holds_under_five_points():
    # F3: the `< 5 points inside [0, 30] K` guard (same rule as _rho_lowt_inset) was unpinned.
    # Built in memory — no fixture .dat may be touched (byte-identity guard).
    r = _run(FX / "tto_synth.dat")
    c = r.data["curves"][0]
    keep = [i for i, t in enumerate(c["t"]) if t <= 30.0][:4] + \
           [i for i, t in enumerate(c["t"]) if t > 30.0]
    thin = {k: ([v[i] for i in keep] if isinstance(v, list) and len(v) == len(c["t"]) else v)
            for k, v in c.items()}
    d = dict(r.data); d["curves"] = [thin]
    r2 = r.model_copy(update={"data": d})
    assert sum(1 for t in thin["t"] if 0 <= t <= 30.0) == 4        # under the 5-point floor
    with_gate = render_kind(r2, "tto_kappa_t", PlotSpec(), GlobalStyle())
    without = render_kind(r2, "tto_kappa_t", PlotSpec(lowt_inset=False), GlobalStyle())
    assert len(with_gate.axes) == len(without.axes)                # no inset axis added


def test_folded_legend_is_placed_by_occupancy_not_forced_outside():
    # F4 -> KNOWN-ISSUES 5: the old rule forced the folded legend outside-right
    # unconditionally, spending ~25% of canvas width even when the panel had room. The
    # occupancy chooser keeps it inside here (this figure has a clear band) without growing
    # the canvas; an explicit legend_loc="outside" must still force the relocation.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_wf_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    fig.canvas.draw()
    assert ax.get_legend() is not None
    assert not getattr(fig, "_cryosweep_legend_grown", False)
    fig2 = render_kind(_run(FX / "tto_synth.dat"), "tto_wf_t",
                       PlotSpec(legend_loc="outside"), GlobalStyle())
    ax2 = fig2.axes[0]
    fig2.canvas.draw()
    assert ax2.get_legend().get_window_extent().x0 >= ax2.get_window_extent().x1


def test_full_view_override_is_reachable_from_zt_only():
    # F5: LOAD-BEARING contract (spec §4) — κ measures clean under _apply_robust_view, and
    # only ZT's heavy tail needs the override; wiring it into another kind would silently
    # disable the robust view there. Behaviourally invisible, so pinned at source level.
    import inspect
    import cryosweep_core.plotting.render as R
    src = inspect.getsource(R)
    assert src.count("_tto_full_view(") == 2                       # one def + one call
    assert "_tto_full_view(" in inspect.getsource(R.render_tto_zt_t)


def test_y_axis_uses_a_mathtext_sci_offset_and_only_on_tto_kinds():
    # V1: ZT runs ~1e-4, which plain ticks render as "0.0000 … 0.0004". A mathtext
    # ScalarFormatter lifts the exponent into a "x10^-4" offset label instead.
    from matplotlib.ticker import ScalarFormatter
    ax = render_kind(_run(FX / "tto_synth.dat"), "tto_zt_t", PlotSpec(), GlobalStyle()).axes[0]
    fmt = ax.yaxis.get_major_formatter()
    assert isinstance(fmt, ScalarFormatter)
    assert fmt.get_useMathText()
    ax.figure.canvas.draw()
    # Behavioural, not attribute-deep: matplotlib's DEFAULT powerlimits (-5, 6) leave a 1e-4
    # axis as "0.0000 … 0.0004" with an EMPTY offset, so a rendered mathtext offset proves
    # set_powerlimits((-2, 2)) took effect.
    off = ax.yaxis.get_offset_text().get_text()
    assert off.startswith("$") and "10" in off, off
    # ... and no other kind's y axis is touched. BEHAVIOURAL (was: a
    # `inspect.getsource(R).count("ScalarFormatter(") == 1` grep, which constrained module
    # SOURCE TEXT as a proxy — a property the byte-identity gate already covers, and one that
    # blocked legitimate edits elsewhere in the module): a non-TTO kind keeps matplotlib's
    # default formatter, i.e. plain ScalarFormatter, no mathtext, no rendered offset.
    from cryosweep_core.analyzers.hc import HCAnalyzer
    hc = HCAnalyzer().analyze(load_dat(str(FX / "hc_synth.dat")), RunConfig())
    for other in ("cp_over_t", "cp_vs_t"):
        fig2 = render_kind(hc, other, PlotSpec(), GlobalStyle())
        ax2 = fig2.axes[0]
        fig2.canvas.draw()
        fmt2 = ax2.yaxis.get_major_formatter()
        assert isinstance(fmt2, ScalarFormatter) and not fmt2.get_useMathText(), other
        assert ax2.yaxis.get_offset_text().get_text() == "", other
    import inspect
    import cryosweep_core.plotting.render as R
    assert "ScalarFormatter(" in inspect.getsource(R._tto_single)


def test_empty_figure_explains_itself_with_the_analyzer_s_own_reason():
    # V2: a blank 0-1 axes reads as a broken plot. The note reuses the capability `reason`
    # string the analyzer already produced — no new copy invented in the renderer.
    for fixture, kind, cap in (("tto_gap_synth.dat", "tto_seebeck_t", "seebeck"),
                               ("tto_norho_synth.dat", "tto_wf_t", "wiedemann_franz")):
        r = _run(FX / fixture)
        reason = next(c["reason"] for c in r.data["capabilities"] if c["name"] == cap)
        assert not c_applicable(r, cap)
        ax = render_kind(r, kind, PlotSpec(), GlobalStyle()).axes[0]
        assert [t.get_text() for t in ax.texts] == [reason]


def test_a_populated_figure_carries_no_empty_note():
    ax = render_kind(_run(FX / "tto_synth.dat"), "tto_kappa_t", PlotSpec(),
                     GlobalStyle()).axes[0]
    assert list(ax.texts) == []


def c_applicable(result, name):
    return next(c["applicable"] for c in result.data["capabilities"] if c["name"] == name)


# ---- final-review hardening: unpinned renderer contracts ----

def _two_direction_result():
    """A synth result rewritten to carry one WARMING and one COOLING curve at the same field.
    No fixture .dat is touched (byte-identity guard); the two curves are told apart on the
    canvas by their (constant, different) kappa values."""
    r = _run(FX / "tto_synth.dat")
    c = r.data["curves"][0]
    n = len(c["t"])
    up = {**c, "direction": "up", "kappa": [1.0] * n}
    down = {**c, "direction": "down", "kappa": [2.0] * n}
    d = dict(r.data)
    d["curves"] = [up, down]
    return r.model_copy(update={"data": d})


def test_direction_legend_key_names_the_marker_it_is_drawn_with():
    # I4 mutation pin: swapping "o" and "^" in `_TTO_DIR_NAMES` survived the whole suite,
    # because the key is only drawn when >1 direction is present and NO fixture is
    # mixed-direction. A warming ramp would then be captioned "↓ cooling" in a paper figure.
    fig = render_kind(_two_direction_result(), "tto_kappa_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    drawn = {float(ln.get_ydata()[0]): ln.get_marker()
             for ln in ax.lines if ln.get_gid() is None}
    assert len(drawn) == 2
    warming_marker, cooling_marker = drawn[1.0], drawn[2.0]
    assert warming_marker != cooling_marker
    leg = ax.get_legend()
    key = {t.get_text(): h.get_marker()
           for t, h in zip(leg.get_texts(), leg.legend_handles)}
    assert key["↑ warming"] == warming_marker
    assert key["↓ cooling"] == cooling_marker


def test_empty_note_says_curve_selection_when_the_capability_IS_applicable():
    # I6 mutation pin: dropping `and not c.get("applicable")` survived the suite, and the
    # figure would then print a POSITIVE capability reason ("kappa(T) curves present") as the
    # explanation for a blank panel. Reachable in the GUI by unchecking every curve.
    r = _run(FX / "tto_synth.dat")
    assert c_applicable(r, "thermal_conductivity") is True
    ax = render_kind(r, "tto_kappa_t", PlotSpec(curves=[]), GlobalStyle()).axes[0]
    assert [t.get_text() for t in ax.texts] == ["no curves selected"]


@pytest.mark.parametrize("kind_key", ("tto_wf_t", "tto_summary_t"))
@pytest.mark.parametrize("width_mm", (90.0, 60.0, 55.0, 50.0))
def test_empty_note_is_fitted_inside_its_axes_at_gui_font_size(width_mm, kind_key):
    # I2: the note was drawn at the full label size with no width fitting, so at a GUI card
    # width of <=60 mm and font_pt=14 it ran past both spines and printed straight through
    # the rotated y-axis label (measured on tto_norho_synth: contained at 90/70 mm, escaping
    # and overprinting at 60/50 mm).
    style = GlobalStyle(font_pt=14.0, width_mm=width_mm)
    fig = render_kind(_run(FX / "tto_norho_synth.dat"), kind_key, PlotSpec(), style)
    fig.canvas.draw()
    # Both kinds must be covered: the headline was the live evidence for I2 (its panel-(c)
    # note escaped the axes and overprinted the rho ylabel at 50-60 mm), yet pinning only
    # tto_wf_t left `_fit_tto_notes(fig)` deletable from render_tto_summary_t with a green
    # suite. Pick the axes that actually carries the note.
    # Select by gid, NOT by "any text": the headline's (a)/(b)/(c) panel tags are also
    # ax.texts, so a naive scan measures a tag and the mutant survives.
    from cryosweep_core.plotting.render import _TTO_NOTE_GID
    ax, note = next((a, t) for a in fig.axes for t in a.texts
                    if t.get_gid() == _TTO_NOTE_GID and t.get_text())
    nb = note.get_window_extent(fig.canvas.get_renderer())
    ab = ax.get_window_extent()
    assert nb.x0 >= ab.x0 and nb.x1 <= ab.x1, (width_mm, nb.x0, nb.x1, ab.x0, ab.x1)
    # ...and clear of the y-axis label, which is what it used to overprint.
    yb = ax.yaxis.label.get_window_extent(fig.canvas.get_renderer())
    assert nb.x0 > yb.x1


def test_overlay_mode_tolerates_an_empty_selection_like_the_single_file_view():
    # M9: `_plot_data`'s overlay branch raises on an empty selection, so "Add to compare…" on
    # a Seebeck-less file blew up exactly where the single-file view renders an explanatory
    # figure (spec §6 tolerance).
    from cryosweep_core.plotting.catalog import OverlayFile
    r = _run(FX / "tto_gap_synth.dat")
    assert series_tto_seebeck_t(r) == []
    fig = render_kind([r, r], "tto_seebeck_t", PlotSpec(), GlobalStyle(),
                      overlay=[OverlayFile(0, "A"), OverlayFile(1, "B")])
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.texts] == ["no Seebeck data"]


def test_a_two_entry_legend_stays_inside_when_the_panel_has_room():
    # M2 -> KNOWN-ISSUES 5: two entries used to be relocated outside unconditionally. The
    # occupancy chooser keeps them inside on this figure (clear upper region), full canvas
    # width preserved, and the legend must not sit on the plotted points.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_kappa_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    leg = ax.get_legend()
    assert len(leg.get_texts()) == 2
    fig.canvas.draw()
    assert leg.get_window_extent().x0 < ax.get_window_extent().x1   # inside
    assert not getattr(fig, "_cryosweep_legend_grown", False)


def test_zt_full_view_pads_the_data_range_by_five_percent():
    # M2: the _tto_full_view pad (0.05) was unconstrained — 0.5 survived, which would waste
    # half the axis on whitespace around the headline peak.
    r = _run(FX / "tto_synth.dat")
    ax = render_kind(r, "tto_zt_t", PlotSpec(), GlobalStyle()).axes[0]
    z = [v for c in r.data["curves"] for v in (c["zt"] or []) if v is not None]
    lo, hi = min(z), max(z)
    bottom, top = ax.get_ylim()
    assert bottom == pytest.approx(lo - 0.05 * (hi - lo), rel=1e-9)
    assert top == pytest.approx(hi + 0.05 * (hi - lo), rel=1e-9)


def test_label_shows_a_field_of_one_thousand_oersted():
    # M2: the |H| >= 50 Oe label threshold was pinned only by 0.077 Oe and 90 kOe, so raising
    # it to 5000 survived — a 1000 Oe curve would silently lose its field from the legend.
    assert _tto_label({"field_oe": 1000.0, "direction": "up"}) == "1000 Oe, warming"
    assert _tto_label({"field_oe": 60.0, "direction": "up"}) == "60 Oe, warming"
    assert _tto_label({"field_oe": 49.0, "direction": "up"}) == "warming"
