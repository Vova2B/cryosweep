"""tto_summary_t: the stacked kappa/S/rho headline (spec §4)."""
import matplotlib; matplotlib.use("Agg")       # noqa: E702
import itertools
import pathlib

import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.plotting.catalog import (BUILTIN_PLOTKINDS, OverlayFile, overlay_series,
                                        series_tto_summary_t)
from cryosweep_core.plotting.render import default_kind_for, render_for, render_kind
from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec

FX = pathlib.Path("tests/core/fixtures")


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def test_headline_is_the_default_kind_for_tto():
    assert default_kind_for("tto") == "tto_summary_t"


def test_kind_registered_with_its_label():
    k = {x.key: x for x in BUILTIN_PLOTKINDS}["tto_summary_t"]
    assert k.probe == "tto" and k.label == "κ / S / ρ vs T" and k.group_colored is True


def test_series_carry_all_three_panel_prefixes():
    s = series_tto_summary_t(_run(FX / "tto_synth.dat"))
    prefixes = {x.key.split(":")[0] for x in s}
    assert prefixes == {"kappa", "seebeck", "rho"}


def _panel_y(ax):
    """Every DATA line's y-data on a panel (gid None excludes reflines)."""
    ys = [np.asarray(ln.get_ydata(), float) for ln in ax.lines if ln.get_gid() is None]
    assert ys, "panel has no data lines"
    return np.concatenate(ys)


def _curve_values(result, key, factor=1.0):
    """The analyzer's own per-point array for `key`, over every curve, times `factor`."""
    vals = [np.nan if v is None else float(v)
            for c in result.data["curves"] for v in (c.get(key) or [])]
    a = np.asarray(vals, float) * factor
    return a[np.isfinite(a)]


# Panel index -> (analyzer array key, plot-space factor). rho: Ohm*m -> Ohm*cm (x100) ->
# microOhm*cm (x1e6) = 1e8.
_PANELS = ((0, "kappa", 1.0), (1, "seebeck", 1.0), (2, "rho", 1e8))


def _assert_panels_carry_their_own_quantity(result):
    """C1: pin each panel to ITS array on the ARTISTS, not on the labels. Reordering
    `_TTO_PANEL_PREFIX` used to pass the whole suite while Seebeck data was drawn under the
    kappa label."""
    fig = render_kind(result, "tto_summary_t", PlotSpec(), GlobalStyle())
    assert fig.axes[2].get_ylabel() == "ρ (µΩ·cm)"
    for i, key, factor in _PANELS:
        drawn = _panel_y(fig.axes[i])
        want = _curve_values(result, key, factor)
        assert float(np.nanmin(drawn)) == pytest.approx(float(want.min()), rel=1e-9)
        assert float(np.nanmax(drawn)) == pytest.approx(float(want.max()), rel=1e-9)


def test_each_panel_draws_the_quantity_its_label_claims_synth():
    _assert_panels_carry_their_own_quantity(_run(FX / "tto_synth.dat"))


def test_each_panel_draws_the_quantity_its_label_claims_real(tto_real_path):
    _assert_panels_carry_their_own_quantity(_run(tto_real_path))


def test_headline_renders_three_stacked_shared_x_panels():
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t", PlotSpec(), GlobalStyle())
    axes = fig.axes
    assert len(axes) == 3
    # SHARED x is the whole point of the layout: subplots(3, 1) WITHOUT sharex used to pass.
    sibs = axes[0].get_shared_x_axes().get_siblings(axes[0])
    assert axes[1] in sibs and axes[2] in sibs
    assert axes[0].get_xlim() == axes[1].get_xlim() == axes[2].get_xlim()
    assert axes[0].get_xlabel() == "" and axes[1].get_xlabel() == ""
    assert axes[2].get_xlabel() == "Temperature (K)"
    assert axes[0].get_ylabel() == "κ (W K⁻¹ m⁻¹)"
    assert axes[1].get_ylabel() == "S (µV/K)"
    assert axes[2].get_ylabel().startswith("ρ (")


def test_panels_carry_a_b_c_tags_and_abut():
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t", PlotSpec(), GlobalStyle())
    fig.canvas.draw()
    for tag, ax in zip(("(a)", "(b)", "(c)"), fig.axes):
        assert tag in [t.get_text() for t in ax.texts]
    boxes = [ax.get_position() for ax in fig.axes]
    assert boxes[0].y0 == pytest.approx(boxes[1].y1, abs=1e-4)   # stacked shared-x -> no gutter
    assert boxes[1].y0 == pytest.approx(boxes[2].y1, abs=1e-4)


def test_rho_panel_uses_an_engineering_prefix_from_ohm_cm():
    # rho ~1e-8..1e-7 Ohm*m -> x100 = 1e-6..1e-5 Ohm*cm -> median < 1e-3 -> microOhm*cm.
    r = _run(FX / "tto_synth.dat")
    fig = render_kind(r, "tto_summary_t", PlotSpec(), GlobalStyle())
    ax_rho = fig.axes[2]
    assert ax_rho.get_ylabel() == "ρ (µΩ·cm)"
    y = _panel_y(ax_rho)
    # Pinned to the ANALYZER's array: the old `1.0 <= nanmax <= 100.0` window sat exactly on
    # this fixture's boundary (max = 10), so a x1000 conversion survived it.
    want = _curve_values(r, "rho", 1e8)
    assert float(np.nanmax(y)) == pytest.approx(float(want.max()), rel=1e-9)


def test_rho_panel_lands_in_the_real_files_microohm_cm_range(tto_real_path):
    r = _run(tto_real_path)
    fig = render_kind(r, "tto_summary_t", PlotSpec(), GlobalStyle())
    assert fig.axes[2].get_ylabel() == "ρ (µΩ·cm)"
    y = _panel_y(fig.axes[2])
    assert float(np.nanmin(y)) == pytest.approx(251.90, abs=0.05)
    assert float(np.nanmax(y)) == pytest.approx(371.96, abs=0.05)


def test_seebeck_panel_gets_a_zero_refline_when_s_crosses_zero(tto_real_path):
    fig = render_kind(_run(tto_real_path), "tto_summary_t", PlotSpec(), GlobalStyle())
    assert any(ln.get_gid() == "refline" for ln in fig.axes[1].lines)


def test_no_zero_refline_when_seebeck_never_crosses_zero():
    # Deliberately the OPPOSITE expectation from test_seebeck_kind_draws_the_zero_reference_line
    # (Task 6, tests/core/test_tto_plots.py) on the SAME fixture, and both are correct: spec §4
    # gives `tto_seebeck_t` a hard unconditional zero line and this stacked panel a
    # crossing-conditional one, to keep the cramped middle panel clean. Not an inconsistency
    # to reconcile.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t", PlotSpec(), GlobalStyle())
    assert not any(ln.get_gid() == "refline" for ln in fig.axes[1].lines)


def test_legend_drawn_once_on_the_top_panel_for_a_single_curve(tto_real_path):
    fig = render_kind(_run(tto_real_path), "tto_summary_t", PlotSpec(), GlobalStyle())
    legs = [ax.get_legend() for ax in fig.axes]
    assert legs[0] is not None and legs[1] is None and legs[2] is None
    assert [t.get_text() for t in legs[0].get_texts()] == ["cooling"]


def test_render_for_routes_a_tto_result_to_the_headline():
    fig = render_for(_run(FX / "tto_synth.dat"), PlotSpec(), GlobalStyle())
    assert len(fig.axes) == 3


def test_full_kappa_range_stays_inside_the_headline_kappa_panel(tto_real_path):
    ax = render_kind(_run(tto_real_path), "tto_summary_t", PlotSpec(), GlobalStyle()).axes[0]
    bottom, top = ax.get_ylim()
    assert bottom <= 0.0368108 and top >= 4.28858


@pytest.mark.parametrize("fixture,cap,panel", [("tto_gap_synth.dat", "seebeck", 1),
                                               ("tto_norho_synth.dat", "wiedemann_franz", 2)])
def test_degenerate_file_still_renders_three_panels_and_explains_the_empty_one(
        fixture, cap, panel):
    # I2: "must not raise" and "the note is drawn" were verified only in prose — replacing
    # `_tto_empty_note` with `raise RuntimeError`, or deleting the call, survived the suite.
    # The note text is the ANALYZER's own capability reason (mirrors test_tto_plots.py).
    r = _run(FX / fixture)
    reason = next(c["reason"] for c in r.data["capabilities"] if c["name"] == cap)
    fig = render_kind(r, "tto_summary_t", PlotSpec(), GlobalStyle())
    assert len(fig.axes) == 3
    ax = fig.axes[panel]
    assert reason in [t.get_text() for t in ax.texts]
    assert [ln for ln in ax.lines if ln.get_gid() is None] == []
    # An empty panel keeps no y ticks: default 0-1 ticks under a real unit label read as a
    # populated axis whose points are hiding somewhere.
    assert list(ax.get_yticks()) == []


def test_overlay_selects_file_qualified_keys_and_colours_by_file():
    # I3: the GUI checklist commits effective keys ('0::kappa:0:down'); matching raw keys
    # emptied the selection -> ValueError -> "no plottable data" on the probe's DEFAULT kind.
    rs = [_run(FX / "tto_synth.dat"), _run(FX / "tto_norho_synth.dat")]
    ov = [OverlayFile(0, "file A"), OverlayFile(1, "file B")]
    kind = {k.key: k for k in BUILTIN_PLOTKINDS}["tto_summary_t"]
    keys = [s.key for s in overlay_series(kind, rs, ov)]
    assert any(k.startswith("0::kappa:") for k in keys)
    fig = render_kind(rs, "tto_summary_t", PlotSpec(curves=keys), GlobalStyle(), overlay=ov)
    assert len(fig.axes) == 3
    leg = fig.axes[0].get_legend()
    assert [t.get_text() for t in leg.get_texts()] == ["file A", "file B"]
    # File identity is the colour: both files' curves used to share one colour.
    colours = {ln.get_color() for ln in fig.axes[0].lines if ln.get_gid() is None}
    assert len(colours) == 2
    # ... and the panel router still sees the RAW key prefix, so kappa lands on the kappa panel.
    want = np.concatenate([_curve_values(r, "kappa") for r in rs])
    assert float(np.nanmax(_panel_y(fig.axes[0]))) == pytest.approx(float(want.max()), rel=1e-9)


def test_overlay_default_selection_also_renders():
    rs = [_run(FX / "tto_synth.dat"), _run(FX / "tto_synth.dat")]
    ov = [OverlayFile(0, "A"), OverlayFile(1, "B")]
    fig = render_kind(rs, "tto_summary_t", PlotSpec(), GlobalStyle(), overlay=ov)
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["A", "B"]


def test_legend_proxy_carries_the_marker_the_curves_are_drawn_with(tto_real_path):
    # m2: a hard-coded 'o' proxy matched nothing on the canvas — the real file is a cooling
    # ramp drawn with '^'.
    fig = render_kind(_run(tto_real_path), "tto_summary_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    drawn = {ln.get_marker() for ln in ax.lines if ln.get_gid() is None}
    proxies = {h.get_marker() for h in ax.get_legend().legend_handles}
    assert drawn == {"^"} and proxies == drawn


def test_a_lone_legend_entry_stays_inside_the_axes(tto_real_path):
    # m4: relocating one word outside-right grows the canvas and spends ~20% of the figure
    # width on it. An explicit legend_loc='outside' still forces the relocation.
    r = _run(tto_real_path)
    fig = render_kind(r, "tto_summary_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]; fig.canvas.draw()
    assert ax.get_legend().get_window_extent().x1 <= ax.get_window_extent().x1
    fig2 = render_kind(r, "tto_summary_t", PlotSpec(legend_loc="outside"), GlobalStyle())
    ax2 = fig2.axes[0]; fig2.canvas.draw()
    assert ax2.get_legend().get_window_extent().x0 >= ax2.get_window_extent().x1


def test_panels_keep_matplotlibs_default_y_formatter():
    # I1: the removed `ticklabel_format` loop was inert; nothing should install a formatter
    # here either — all three panels land O(1)-O(100) after the rho engineering prefix.
    from matplotlib.ticker import ScalarFormatter
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t", PlotSpec(), GlobalStyle())
    fig.canvas.draw()
    for ax in fig.axes:
        fmt = ax.yaxis.get_major_formatter()
        assert isinstance(fmt, ScalarFormatter) and not fmt.get_useMathText()
        assert ax.yaxis.get_offset_text().get_text() == ""


@pytest.mark.parametrize("font_pt", [9, 14])
def test_panel_ylabels_stay_inside_the_figure_and_never_overlap(font_pt):
    # FINDING A: three panels share a 70 mm figure, and constrained layout does NOT grow the
    # canvas for a rotated ylabel taller than its own axes — it lets it overflow. At the GUI's
    # 14 pt the kappa label ran 81 px off the top edge (read `W K⁻¹ m`) and its bbox overlapped
    # BOTH neighbours; even at the 9 pt default it lost its closing `)`. Both sizes are pinned:
    # the fix is a font-size-invariant cap, so a size-specific test would not hold it.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t", PlotSpec(),
                      GlobalStyle(font_pt=font_pt))
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    boxes = [ax.yaxis.label.get_window_extent(rend) for ax in fig.axes
             if ax.yaxis.label.get_text()]
    assert len(boxes) == 3
    for bb in boxes:
        assert fig.bbox.containsy(bb.y0) and fig.bbox.containsy(bb.y1), \
            f"ylabel escapes the figure: {bb} vs {fig.bbox}"
    for a, b in itertools.combinations(boxes, 2):
        assert not a.overlaps(b), f"ylabels overlap: {a} / {b}"


def test_the_stacked_panel_ylabels_share_one_font_size():
    # The cap is per-panel, but three labels at three sizes reads as a mistake; they are
    # levelled to the smallest fitted size.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t", PlotSpec(),
                      GlobalStyle(font_pt=14))
    sizes = {ax.yaxis.label.get_fontsize() for ax in fig.axes if ax.yaxis.label.get_text()}
    assert len(sizes) == 1
