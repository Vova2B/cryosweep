# tests/core/test_render_legend_layout.py — Piece 2 (legend/layout robustness)
import warnings
import matplotlib
import numpy as np
import pytest
from cryosweep_core.result import Result, Provenance
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.catalog import (
    BUILTIN_PLOTKINDS, series_hall_raw_vs_asym, series_hall_tdep_stages,
)
from cryosweep_core.plotting.render import (
    render_kind, _legend_ncol, LEGEND_MAX_COLS,
    _merged_legend, _new_fig, _axes_points_in_host_frac,
)

_KIND = {k.key: k for k in BUILTIN_PLOTKINDS}
_NEG = "R_xy(−B)"   # Unicode minus, matches catalog.py


def _prov():
    return Provenance(file="x", sha256="ab", app_version=None)


def _raw_vs_asym_result(n_temps, drop_neg_at=None):
    """Synthetic Hall result: n_temps held temperatures, each with +B/-B raw branches + asym."""
    pts = []
    for k in range(n_temps):
        # rraw chosen ON the asym Stage-B line (y=slope*B=2*B) for both +B/-B branches so the
        # envelope-clipping fit-line drawer (fit-line-envelope fix) never clips these regression
        # fixtures: neg branch's own |B| span is [0.5,1.0] same as pos branch, and its own data
        # must bracket the line's y-values there too.
        braw, rraw = [-1.0, -0.5, 0.5, 1.0], [2.0, 1.0, 1.0, 2.0]
        if drop_neg_at is not None and k == drop_neg_at:
            braw, rraw = [0.5, 1.0], [1.0, 2.0]          # no negative branch this T
        pts.append({"temperature": 10.0 * (k + 1),
                    "field_raw_T": braw, "R_xy_raw": rraw,
                    "field_asym_T": [0.5, 1.0], "R_asym": [1.0, 2.0],
                    "slope_ohm_per_T": 2.0, "asym_intercept_ohm": 0.0})
    return Result(status="ok", data={"probe": "hall", "points": pts}, provenance=_prov())


def _stages_result(n_temps):
    """Synthetic hall_tdep result with n_temps stages (each: raw + asym vs |B|)."""
    stages = [{"temperature": 5.0 * (k + 1), "fields_T": [0.5, 1.0],
               "R_raw": [1.0, 2.0], "R_asym": [0.5, 1.0]} for k in range(n_temps)]
    return Result(status="ok", data={"probe": "hall_tdep", "stages": stages}, provenance=_prov())


def _mr_result(n_temps):
    """Synthetic resistivity result: one bridge, n_temps rho_h (field) loops (each a legend entry)."""
    curves = [{"held_temp_k": 2.0 + 10 * k, "direction": 1,
               "field": [-1.0, 0.0, 1.0], "rho": [1.0, 0.9, 1.0],
               "rho_zero_field": 0.9} for k in range(n_temps)]
    data = {"probe": "resistivity", "rho_source": "instrument_column",
            "bridges": [{"channel": 1, "rho_source": "instrument_column",
                         "classification": "metallic",
                         "rho_t_curves": [], "rho_h_curves": curves}],
            "capabilities": []}
    return Result(status="ok", data=data, provenance=_prov())


def test_raw_vs_asym_series_carry_roles():
    s = series_hall_raw_vs_asym(_raw_vs_asym_result(2))
    roles = {x.role for x in s}
    assert roles == {"R_xy(+B)", _NEG, "R_asym"}
    assert all(x.role is not None for x in s)


def test_stages_series_carry_roles():
    s = series_hall_tdep_stages(_stages_result(2))
    assert {x.role for x in s} == {"R(+|B|) raw", "R_asym"}


def test_two_kinds_are_group_colored():
    assert _KIND["hall_raw_vs_asym"].group_colored is True
    assert _KIND["hall_tdep_stages"].group_colored is True
    assert _KIND["hc_lowt_multifield"].group_colored is True
    # everything else stays default False
    _GROUP_COLORED = {"hall_raw_vs_asym", "hall_tdep_stages", "hc_lowt_multifield",
                      "hc_schottky_multifield", "hc_transition_multifield",
                      "hc_transition_signal", "acms_chi_t", "acms_chi_prime_t",
                      "acms_chi_dprime_t", "acms_mdc_t",
                      "tto_summary_t",
                      "tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_wf_t",
                      "tto_lorenz_t"}
    assert all(not k.group_colored for k in BUILTIN_PLOTKINDS
               if k.key not in _GROUP_COLORED)


def test_legend_ncol_scaling():
    assert _legend_ncol(11) == 1
    assert _legend_ncol(12) == 1
    assert _legend_ncol(18) == 2
    assert _legend_ncol(42) == 3
    assert _legend_ncol(43) == 4      # 4-col cap onset
    assert _legend_ncol(200) == LEGEND_MAX_COLS


def _legend_outside_right(fig):
    """True iff the legend sits to the right of the axes (relocated)."""
    fig.canvas.draw()
    ax = fig.axes[0]
    leg = ax.get_legend()
    return leg.get_window_extent().x0 >= ax.get_window_extent().x1 - 1.0


def test_dense_nongrouped_legend_relocates_outside():
    # resistivity_mr is NOT group_colored; 18 entries on synthetic data -> relocate via _finish
    r = _mr_result(18)
    fig = render_kind(r, "resistivity_mr")
    assert _legend_outside_right(fig)
    assert fig.axes[0].get_position().width > 0     # axes did not collapse


def test_sparse_legend_stays_inside():
    r = _mr_result(4)
    fig = render_kind(r, "resistivity_mr")
    assert not _legend_outside_right(fig)


def test_relocated_legend_grows_canvas_axes_keep_width():
    # Outside-right legend must NOT squish the axes: the figure canvas grows by the
    # legend width instead, so dense-legend axes stay ~as wide as sparse-legend axes.
    sparse = render_kind(_mr_result(4), "resistivity_mr")
    dense = render_kind(_mr_result(18), "resistivity_mr")
    sparse.canvas.draw(); dense.canvas.draw()
    w_sparse = sparse.axes[0].get_window_extent().width
    w_dense = dense.axes[0].get_window_extent().width
    assert w_dense >= 0.85 * w_sparse
    # canvas actually grew (legend paid for by extra width, not by the axes)
    assert dense.get_size_inches()[0] > sparse.get_size_inches()[0]


def test_sparse_legend_canvas_width_unchanged():
    # <=11 entries: inside legend, figure keeps the style's width (byte-identical path)
    from cryosweep_core.plotting.spec import GlobalStyle as _GS
    style = _GS()
    fig = render_kind(_mr_result(4), "resistivity_mr")
    assert fig.get_size_inches()[0] == pytest.approx(style.width_mm / 25.4)


def test_relocated_legend_font_capped_at_7():
    # 18 entries -> relocate; legend_size 9 should be capped to 7 on the relocated legend
    fig = render_kind(_mr_result(18), "resistivity_mr", None, GlobalStyle(legend_size=9.0))
    sizes = {t.get_fontsize() for t in fig.axes[0].get_legend().get_texts()}
    assert sizes == {7.0}


def test_inside_legend_font_not_capped():
    # 4 entries -> inside; full legend_size honoured (cap applies only to relocate branch)
    fig = render_kind(_mr_result(4), "resistivity_mr", None, GlobalStyle(legend_size=9.0))
    sizes = {t.get_fontsize() for t in fig.axes[0].get_legend().get_texts()}
    assert sizes == {9.0}


def _labels(fig):
    leg = fig.axes[0].get_legend()
    return [t.get_text() for t in leg.get_texts()]


def test_raw_vs_asym_legend_folds_groups_plus_roles():
    fig = render_kind(_raw_vs_asym_result(5), "hall_raw_vs_asym")
    labels = _labels(fig)
    assert len(labels) == 5 + 3                       # 5 T proxies + 3 role proxies
    assert {"R_xy(+B)", _NEG, "R_asym"} <= set(labels)


def test_stages_legend_folds_groups_only():
    # PQ-2 Task 2 (sanctioned look-change): hall_tdep_stages is now multi-panel (Raw /
    # Zero-subtracted / Antisymmetrized side-by-side axes); role is encoded by PANEL, not
    # marker, so the single figure-wide legend (drawn on the last axes) carries only the T
    # group proxies -- no per-role marker key is needed any more.
    fig = render_kind(_stages_result(6), "hall_tdep_stages")
    leg = fig.axes[-1].get_legend()
    labels = [t.get_text() for t in leg.get_texts()]
    assert len(labels) == 6
    assert not ({"R(+|B|) raw", "R_asym"} & set(labels))


def test_role_key_lists_only_present_roles():
    # one T has no negative branch -> R_xy(-B) still present for other T's (4 groups + 3 roles)
    fig = render_kind(_raw_vs_asym_result(4, drop_neg_at=0), "hall_raw_vs_asym")
    labels = _labels(fig)
    assert len(labels) == 4 + 3
    assert _NEG in labels


def _non_refline(fig):
    """Data + fit lines only, excluding the PQ-2 H=0 reference axhline (gid='refline'),
    which hall_raw_vs_asym now always draws first (index 0) — see PQ-2 Task 1."""
    return [l for l in fig.axes[0].lines if l.get_gid() != "refline"]


def test_grouped_color_shared_per_T_markers_per_role():
    fig = render_kind(_raw_vs_asym_result(2), "hall_raw_vs_asym")
    lines = _non_refline(fig)                          # first 6 = markers (2T x 3 roles), then fit lines
    c = [matplotlib.colors.to_hex(l.get_color()) for l in lines[:3]]
    m = [lines[i].get_marker() for i in range(3)]
    assert c[0] == c[1] == c[2]                        # same T -> same colour
    assert len(set(m)) == 3                            # distinct role markers
    assert matplotlib.colors.to_hex(lines[3].get_color()) != c[0]   # next T differs


def test_grouped_single_group_uses_style_color():
    # hall_raw_vs_asym now force-colours single-T by role (PQ-2 Task 1: classic blue/red
    # branch figure), overriding GlobalStyle.color for that kind specifically — exercise the
    # underlying _plot_data_grouped style.color mechanism via hall_tdep_stages instead, which
    # takes the same group_colored path without a role_colors override.
    fig = render_kind(_stages_result(1), "hall_tdep_stages", None, GlobalStyle(color="#ff0000"))
    assert matplotlib.colors.to_hex(_non_refline(fig)[0].get_color()) == "#ff0000"


def test_grouped_first_role_marker_is_style_marker():
    fig = render_kind(_raw_vs_asym_result(1), "hall_raw_vs_asym", None, GlobalStyle(marker="D"))
    assert _non_refline(fig)[0].get_marker() == "D"    # first plotted series = rawpos = first role


def test_grouped_empty_selection_raises():
    with pytest.raises(ValueError):
        render_kind(_raw_vs_asym_result(3), "hall_raw_vs_asym", PlotSpec(curves=[]))


def test_fit_line_matches_group_color():
    fig = render_kind(_raw_vs_asym_result(2), "hall_raw_vs_asym")
    lines = _non_refline(fig)
    # 6 markers (2T x rawpos/rawneg/asym), then 4 branch fits (PQ-2 Task 1: pos+neg per T),
    # then 2 asym fits -> T1 asym fit is now at index 10 (was 6 before branch fits existed).
    asym_color = matplotlib.colors.to_hex(lines[2].get_color())    # T1 asym marker
    fit_color = matplotlib.colors.to_hex(lines[10].get_color())   # T1 asym fit line
    assert asym_color == fit_color


def test_grouped_kind_overlay_uses_file_path():
    from cryosweep_core.plotting.catalog import OverlayFile
    r = _raw_vs_asym_result(2)
    ov = [OverlayFile(0, "A", None), OverlayFile(1, "B", None)]
    fig = render_kind([r, r], "hall_raw_vs_asym", PlotSpec(), GlobalStyle(), overlay=ov)
    labels = {ln.get_label() for ln in fig.axes[0].lines}
    assert any(" · " in lbl for lbl in labels)         # overlay (colour-by-file) path, NOT grouped


# Sizes = (default 90x70) and the REALISTIC GUI floor (70x72; just above the 71mm canvas-height
# minimum from output_panel.py setMinimumHeight(280)/_SCREEN_DPI=100). Do NOT test below the GUI
# floor (e.g. 45mm height / sub-55mm width) — the dense legend legitimately can't fit there.
@pytest.mark.parametrize("w,h", [(90.0, 70.0), (70.0, 72.0)])
def test_raw_vs_asym_never_collapses(w, h):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fig = render_kind(_raw_vs_asym_result(9), "hall_raw_vs_asym", None,
                          GlobalStyle(width_mm=w, height_mm=h))   # 9 T -> 12 entries -> relocate
        fig.canvas.draw()
    pos = fig.axes[0].get_position()
    assert not any("collapsed" in str(x.message) for x in rec)
    assert pos.height > 0.3 and pos.width > 0.05


@pytest.mark.parametrize("w,h", [(90.0, 70.0), (70.0, 72.0)])
def test_stages_never_collapses(w, h):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fig = render_kind(_stages_result(16), "hall_tdep_stages", None,
                          GlobalStyle(width_mm=w, height_mm=h))   # 16 T -> 18 entries -> relocate
        fig.canvas.draw()
    pos = fig.axes[0].get_position()
    assert not any("collapsed" in str(x.message) for x in rec)
    assert pos.height > 0.3 and pos.width > 0.05


# ---- Multi-axis "best" legend placement (twin/offset composites) --------------------
# matplotlib's loc="best" only dodges the HOST axis' artists, so a merged legend on a twin
# composite can land on the twin curve. _merged_legend evaluates all axes' data across the
# four inside corners and pins the clearest, else relocates outside.

def _twin_fig(host_curves, twin_curves, style=None):
    """Build a raw twinx figure: host_curves/twin_curves are lists of (y-array) over a shared
    x-span; returns (fig, ax, tax, handles, labels)."""
    style = style or GlobalStyle()
    fig = _new_fig(style)
    ax = fig.add_subplot(111)
    tax = ax.twinx()
    x = np.linspace(0.0, 300.0, 200)
    handles, labels = [], []
    for i, y in enumerate(host_curves):
        ln, = ax.plot(x, y, color="C0"); handles.append(ln); labels.append(f"chi {i}")
    for i, y in enumerate(twin_curves):
        ln, = tax.plot(x, y, color="C3"); handles.append(ln); labels.append(f"inv {i}")
    return fig, ax, tax, handles, labels


def _legend_overlap_frac(fig, ax_host):
    """Fraction of the figure's visible plotted points (union over all axes) that fall under the
    drawn legend bbox, in host axes-fraction coords."""
    fig.canvas.draw()
    leg = ax_host.get_legend()
    assert leg is not None
    bb = leg.get_window_extent()
    inv = ax_host.transAxes.inverted()
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    pts = _axes_points_in_host_frac(ax_host)
    if len(pts) == 0:
        return 0.0
    under = ((pts[:, 0] >= x0) & (pts[:, 0] <= x1) &
             (pts[:, 1] >= y0) & (pts[:, 1] <= y1)).sum()
    return under / len(pts)


def test_merged_legend_avoids_twin_curve_full_span():
    # MPMS-shaped: 3 rising (host χ) + 3 falling (twin 1/χ) full-span crossing curves, 6 entries.
    # loc="best" on the host alone would land on the falling twin curve. The fix must leave the
    # legend clear of BOTH axes' data (here: no clear corner -> relocate outside -> ~0 overlap).
    x = np.linspace(0.0, 300.0, 200)
    fig, ax, tax, handles, labels = _twin_fig(
        [x + 5 * i for i in range(3)], [300.0 - x + 5 * i for i in range(3)])
    _merged_legend(ax, handles, labels, GlobalStyle(), PlotSpec())
    assert _legend_overlap_frac(fig, ax) <= 0.02


def test_merged_legend_picks_clear_corner():
    # Both curves hug the LOWER band (y in [0..40] of a 0..300 view) leaving the top corners open;
    # the merged legend must sit inside at an upper corner (not outside), overlapping ~no points.
    x = np.linspace(0.0, 300.0, 200)
    low = 20.0 + 5.0 * np.sin(x / 30.0)
    fig, ax, tax, handles, labels = _twin_fig(
        [low + 2 * i for i in range(3)], [low + 2 * i for i in range(3)])
    tax.set_ylim(ax.get_ylim())                         # share view so "low band" holds on both
    _merged_legend(ax, handles, labels, GlobalStyle(), PlotSpec())
    leg = ax.get_legend()
    assert leg is not None
    # inside placement (anchored within the axes, not the outside-right (1.02, 0.5) relocation)
    fig.canvas.draw()
    bb = leg.get_window_extent()
    inv = ax.transAxes.inverted()
    (fx0, _), (fx1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    assert max(fx0, fx1) <= 1.05                         # not pushed outside the right spine
    assert _legend_overlap_frac(fig, ax) <= 0.02


def test_merged_legend_explicit_inside_override_untouched():
    # loc="inside" must NOT get the corner treatment -> plain ax.legend(loc="best") inside.
    x = np.linspace(0.0, 300.0, 200)
    fig, ax, tax, handles, labels = _twin_fig(
        [x + 5 * i for i in range(3)], [300.0 - x + 5 * i for i in range(3)])
    _merged_legend(ax, handles, labels, GlobalStyle(), PlotSpec(legend_loc="inside"))
    assert ax.get_legend() is not None


def _rh_n_twin_result(n_temps):
    """Synthetic hall result with rh + n per temperature (drives hall_rh_n_twin)."""
    pts = [{"temperature": 10.0 * (k + 1), "R_H": -1.0e-9 - 1.0e-11 * k,
            "carrier_n": 1.0e25 + 1.0e22 * k} for k in range(n_temps)]
    return Result(status="ok", data={"probe": "hall", "points": pts}, provenance=_prov())


def test_hall_rh_n_twin_legend_still_sane():
    # Regression: the twin R_H/n Hall composite still draws a single merged legend on the host,
    # clear of both axes' data.
    fig = render_kind(_rh_n_twin_result(5), "hall_rh_n_twin")
    ax = fig.axes[0]
    assert ax.get_legend() is not None
    assert _legend_overlap_frac(fig, ax) <= 0.02


def _legend_vs_ydecor_intersections(fig, ax_host):
    """List of axes whose y-axis decoration bbox (tick labels + label) intersects the drawn
    legend's bbox (display px)."""
    fig.canvas.draw()
    leg = ax_host.get_legend()
    assert leg is not None
    lb = leg.get_window_extent()
    renderer = fig.canvas.get_renderer()
    hits = []
    for ax in fig.axes:
        bb = ax.yaxis.get_tightbbox(renderer)
        if bb is not None and lb.overlaps(bb):
            hits.append(ax)
    return hits


def test_outside_relocated_legend_clears_twin_axis_decorations():
    # MPMS-shaped full-span crossing twin (no clear corner -> outside relocation): the legend
    # must land to the RIGHT of the twin axis' tick numbers + rotated ylabel, not on them.
    x = np.linspace(0.0, 300.0, 200)
    fig, ax, tax, handles, labels = _twin_fig(
        [x + 5 * i for i in range(3)], [300.0 - x + 5 * i for i in range(3)])
    tax.set_ylabel("1/χ (mol·Oe/emu)")
    _merged_legend(ax, handles, labels, GlobalStyle(), PlotSpec())
    assert _legend_vs_ydecor_intersections(fig, ax) == []


def test_forced_outside_legend_clears_offset_axis_decorations():
    # Explicit legend_loc="outside" on a twin figure must also clear the right decorations.
    x = np.linspace(0.0, 300.0, 200)
    fig, ax, tax, handles, labels = _twin_fig(
        [x + 5 * i for i in range(3)], [300.0 - x + 5 * i for i in range(3)])
    tax.set_ylabel("n (1/m³)")
    _merged_legend(ax, handles, labels, GlobalStyle(), PlotSpec(legend_loc="outside"))
    assert _legend_vs_ydecor_intersections(fig, ax) == []


# --------------------------------------------------------------------------------------
# 1/chi: the legend must not land on the Curie-Weiss annotation (2026-09-01).
# --------------------------------------------------------------------------------------
def test_inverse_chi_legend_does_not_cover_the_cw_annotation():
    """The CW annotation is pinned at axes upper-left with ax.text, and matplotlib's legend
    "best" placement only considers DATA artists — text is invisible to it. On a single-curve
    file no legend is drawn, so the defect never showed; the real multi-field M(T) example has
    one entry per held field plus two fit lines, and "best" put the legend straight on top of
    the annotation (both unreadable).

    Pins the outcome, not the mechanism: whatever corner (or outside relocation) is chosen, no
    legend text box may intersect the annotation's box.
    """
    import pathlib
    from matplotlib.text import Text
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.config import RunConfig
    from cryosweep_core.plotting.render import render_inverse_chi

    ex = pathlib.Path(__file__).resolve().parents[2] / "examples"
    src = ex / "magnetization_vsm_multifield.dat"
    if not src.exists():                       # skip-not-fail, as for every optional data file
        import pytest
        pytest.skip("multi-field VSM example not present")

    reg = build_default_registry()
    r = analyze_file(load_dat(str(src)), RunConfig(), reg)
    fig = render_inverse_chi([r])
    fig.canvas.draw()
    ax = fig.axes[0]
    leg = ax.get_legend()
    assert leg is not None, "the multi-curve file must draw a legend at all"
    rend = fig.canvas.get_renderer()
    # the annotation is the axes-level Text carrying the theta/C line
    anns = [t for t in ax.texts if isinstance(t, Text) and "θ" in t.get_text()]
    assert anns, "the Curie-Weiss annotation must still be drawn"
    ab = anns[0].get_window_extent(rend)
    for t in leg.get_texts():
        b = t.get_window_extent(rend)
        assert not (b.x0 <= ab.x1 and ab.x0 <= b.x1 and b.y0 <= ab.y1 and ab.y0 <= b.y1), \
            f"legend entry {t.get_text()!r} overlaps the Curie-Weiss annotation"
