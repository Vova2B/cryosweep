# tests/core/test_render_hall_pq2.py — PQ-2 Task 1: ± branch-split antisymmetrization upgrade
#                                        PQ-2 Task 2: hall_tdep_stages -> per-stage panels
#                                        PQ-2 Task 3: composite kinds + multi-axes helpers
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from cryosweep_core.plotting.render import render_kind
from cryosweep_core.plotting.catalog import get_kind
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.export import save_figure
from cryosweep_core.result import Result, Provenance


def _res(hall_synth_path):
    rt = load_dat(hall_synth_path)
    return HallAnalyzer().analyze(rt, RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))


def _res_with_long(hall_synth_path):
    """Hall result WITH a same-file longitudinal channel -> rho_xx + mobility populated
    (needed for hall_two_panel's right panel)."""
    rt = load_dat(hall_synth_path)
    return HallAnalyzer().analyze(rt, RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))


def _tdep_res(hall_tdep_synth_path):
    return HallTempDepAnalyzer().analyze(
        load_dat(hall_tdep_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05, "longitudinal_channel": 2}),
    )


def _fit_lines(ax):
    return [ln for ln in ax.lines if ln.get_gid() == "fit"]


def _refline(ax):
    return [ln for ln in ax.lines if ln.get_gid() == "refline"]


def _single_T_spec(T, extra=None):
    curves = [f"rawpos:{T}K", f"rawneg:{T}K", f"asym:{T}K"]
    kw = {"curves": curves}
    if extra:
        kw.update(extra)
    return PlotSpec(**kw)


def test_branch_fit_lines_present_dashed_and_color_matched(hall_synth_path):
    res = _res(hall_synth_path)
    fig = render_kind(res, "hall_raw_vs_asym")
    ax = fig.axes[0]
    fits = _fit_lines(ax)
    # 3 temperatures x (pos-branch fit + neg-branch fit + asym fit) = 9 fit lines
    assert len(fits) == 9
    dashed = [ln for ln in fits if ln.get_linestyle() == "--"]
    # exactly the 6 branch fits (2 branches x 3 T) are forced dashed; the 3 asym fits
    # keep the ordinary fit-line style (unaffected by this task).
    assert len(dashed) == 6


def test_branch_fit_lines_color_matched_to_branch_markers(hall_synth_path):
    # PAIRWISE 1:1: markers are plotted in series order (rawpos, rawneg, asym per T) and
    # fit lines are appended in the same plotted order (branch fits first: rawpos, rawneg;
    # then the asym fit). Each fit's color must equal ITS OWN branch's marker color.
    res = _res(hall_synth_path)
    T = res.data["points"][0]["temperature"]
    fig = render_kind(res, "hall_raw_vs_asym", spec=_single_T_spec(T))
    ax = fig.axes[0]
    markers = [ln for ln in ax.lines if ln.get_gid() not in ("fit", "refline")]
    fits = _fit_lines(ax)
    assert len(markers) == 3 and len(fits) == 3        # rawpos, rawneg, asym markers + 3 fits
    pos_m, neg_m, asym_m = markers                     # series order from the catalog
    pos_f, neg_f, asym_f = fits                        # branch fits appended first, in order
    assert pos_f.get_color() == pos_m.get_color()
    assert neg_f.get_color() == neg_m.get_color()
    assert asym_f.get_color() == asym_m.get_color()


def test_single_T_fit_lines_are_exact_role_colors(hall_synth_path):
    # single-T role palette must propagate to the fit lines: C0 (+B), C3 (−B), 0.2 (asym)
    res = _res(hall_synth_path)
    T = res.data["points"][0]["temperature"]
    fig = render_kind(res, "hall_raw_vs_asym", spec=_single_T_spec(T))
    fits = _fit_lines(fig.axes[0])
    assert [ln.get_color() for ln in fits] == ["C0", "C3", "0.2"]


def test_branch_fit_lines_gated_by_fit_line_false(hall_synth_path):
    res = _res(hall_synth_path)
    fig = render_kind(res, "hall_raw_vs_asym", spec=PlotSpec(fit_line=False))
    ax = fig.axes[0]
    assert _fit_lines(ax) == []


def test_h_zero_reference_line_present_and_excluded_from_legend(hall_synth_path):
    res = _res(hall_synth_path)
    fig = render_kind(res, "hall_raw_vs_asym")
    ax = fig.axes[0]
    reflines = _refline(ax)
    assert len(reflines) == 1
    rl = reflines[0]
    assert rl.get_color() == "black"
    ydata = rl.get_ydata()
    assert all(y == 0 for y in ydata)
    handles, labels = ax.get_legend_handles_labels()
    assert all(not lbl.startswith("_") for lbl in labels)  # sanity: legend labels are real
    assert rl not in handles


def test_single_T_uses_blue_red_role_colors(hall_synth_path):
    res = _res(hall_synth_path)
    T = res.data["points"][0]["temperature"]
    fig = render_kind(res, "hall_raw_vs_asym", spec=_single_T_spec(T))
    ax = fig.axes[0]
    markers = [ln for ln in ax.lines if ln.get_gid() not in ("fit", "refline")]
    colors = {ln.get_color() for ln in markers}
    assert "C0" in colors   # +H branch blue
    assert "C3" in colors   # -H branch red
    assert "0.2" in colors  # asym near-black


def test_multi_T_keeps_group_colors(hall_synth_path):
    res = _res(hall_synth_path)
    fig = render_kind(res, "hall_raw_vs_asym")
    ax = fig.axes[0]
    markers = [ln for ln in ax.lines if ln.get_gid() not in ("fit", "refline")]
    colors = {ln.get_color() for ln in markers}
    # with >1 temperature plotted, role-based blue/red/near-black must NOT be forced
    assert not ({"C0", "C3", "0.2"} <= colors)


def test_curves_selection_hides_neg_branch_and_its_fit(hall_synth_path):
    res = _res(hall_synth_path)
    T = res.data["points"][0]["temperature"]
    fig = render_kind(res, "hall_raw_vs_asym",
                       spec=PlotSpec(curves=[f"rawpos:{T}K", f"asym:{T}K"]))
    ax = fig.axes[0]
    markers = [ln for ln in ax.lines if ln.get_gid() not in ("fit", "refline")]
    fits = _fit_lines(ax)
    assert len(markers) == 2          # rawpos + asym only
    assert len(fits) == 2             # pos-branch fit + asym fit; no neg-branch fit


def test_no_crash_when_slope_or_intercept_missing(hall_synth_path, monkeypatch):
    res = _res(hall_synth_path)
    for p in res.data["points"]:
        p["slope_ohm_per_T"] = None
        p["asym_intercept_ohm"] = None
    fig = render_kind(res, "hall_raw_vs_asym")
    ax = fig.axes[0]
    assert _fit_lines(ax) == []       # no branch/asym fit lines drawn, but no crash


# ---- fit-line envelope clipping (visual-gate stripe-artifact fix) ----------------------
# Real-data bug: a pathological Stage-B fit at one T (e.g. slope=-1.43, intercept=9.6 against
# a ~+-2.5e-4 Ohm data window) makes both fit-line drawers evaluate the line over the FULL
# |B| span, producing an off-scale near-vertical clipped segment inside the robust-view
# window (a full-height stripe artifact). Fix: clip the drawn fit segment to the paired
# series' own data envelope (+-10% pad); skip entirely if the in-envelope sub-span is empty
# or < 5% of the B span.

def _make_pathological(res, temperature):
    """Overwrite one point's field/signal data + Stage-B fit with a hand-crafted pathological
    case: tiny data envelope (~+-1e-3), but a wildly off-scale fit line (slope=-1.43,
    intercept=9.6) that at the data's own B-span evaluates to ~8-11 -- many envelope-widths
    away, so the clipped intersection with [B_min, B_max] must be empty."""
    for p in res.data["points"]:
        if p["temperature"] != temperature:
            continue
        B = [-1.0, -0.5, 0.5, 1.0]
        y = [-1e-3, -5e-4, 5e-4, 1e-3]
        p["field_asym_T"] = list(B); p["R_asym"] = list(y)
        p["field_raw_T"] = list(B); p["R_xy_raw"] = list(y)
        p["slope_ohm_per_T"] = -1.43
        p["asym_intercept_ohm"] = 9.6
        return p
    raise AssertionError(f"no point at T={temperature}")


def test_pathological_asym_fit_line_is_skipped_entirely(hall_synth_path):
    res = _res(hall_synth_path)
    points = sorted(res.data["points"], key=lambda p: p["temperature"])
    bad_T = points[-1]["temperature"]           # T=300 in hall_synth
    _make_pathological(res, bad_T)
    fig = render_kind(res, "hall_asym_vs_B")
    ax = fig.axes[0]
    fits = _fit_lines(ax)
    # one fewer fit line than temperatures: the pathological T's line is off-scale for its
    # own +-1mOhm-ish envelope over its own B span -> intersection empty -> skipped
    assert len(fits) == len(points) - 1
    for f in fits:
        ydata = f.get_ydata()
        assert max(abs(v) for v in ydata) < 1.0     # nowhere near the pathological 8-11 range


def test_healthy_asym_fit_lines_still_drawn_full_span(hall_synth_path):
    # regression: T's whose fit line lies inside their own data envelope keep the exact
    # 2-point full-B-span line as before this fix.
    res = _res(hall_synth_path)
    points = sorted(res.data["points"], key=lambda p: p["temperature"])
    bad_T = points[-1]["temperature"]
    _make_pathological(res, bad_T)
    healthy = [p for p in points if p["temperature"] != bad_T]
    fig = render_kind(res, "hall_asym_vs_B")
    fits = _fit_lines(fig.axes[0])
    assert len(fits) == len(healthy)
    for f, p in zip(fits, healthy):
        B = p["field_asym_T"]; slope = p["slope_ohm_per_T"]; b0 = p["asym_intercept_ohm"]
        xs = list(f.get_xdata()); ys = list(f.get_ydata())
        assert len(xs) == 2
        assert xs[0] == pytest.approx(min(B)) and xs[1] == pytest.approx(max(B))
        assert ys[0] == pytest.approx(b0 + slope * xs[0])
        assert ys[1] == pytest.approx(b0 + slope * xs[1])


def test_pathological_branch_and_asym_fit_lines_all_skipped_on_raw_vs_asym(hall_synth_path):
    # hall_raw_vs_asym overlays branch fits (rawpos/rawneg) AND the asym fit; the same
    # pathological slope/intercept feeds all three for a given T, so all three must be
    # skipped (none of them survive envelope clipping).
    res = _res(hall_synth_path)
    points = sorted(res.data["points"], key=lambda p: p["temperature"])
    bad_T = points[-1]["temperature"]
    _make_pathological(res, bad_T)
    fig = render_kind(res, "hall_raw_vs_asym", spec=_single_T_spec(bad_T))
    ax = fig.axes[0]
    assert _fit_lines(ax) == []   # rawpos fit, rawneg fit, asym fit all skipped
    # markers themselves are unaffected (display-only fix; no analyzer/data changes)
    markers = [ln for ln in ax.lines if ln.get_gid() not in ("fit", "refline")]
    assert len(markers) == 3


def test_healthy_branch_fit_lines_still_full_span_on_raw_vs_asym(hall_synth_path):
    res = _res(hall_synth_path)
    points = sorted(res.data["points"], key=lambda p: p["temperature"])
    bad_T = points[-1]["temperature"]
    _make_pathological(res, bad_T)
    good_T = points[0]["temperature"]
    fig = render_kind(res, "hall_raw_vs_asym", spec=_single_T_spec(good_T))
    ax = fig.axes[0]
    fits = _fit_lines(ax)
    assert len(fits) == 3            # rawpos, rawneg, asym fits all present (dashed x2 + asym)
    for f in fits:
        xs = f.get_xdata()
        assert len(xs) == 2          # unchanged exact full-span 2-point line


def test_slope_zero_horizontal_line_drawn_when_inside_envelope(hall_synth_path):
    res = _res(hall_synth_path)
    points = sorted(res.data["points"], key=lambda p: p["temperature"])
    T0 = points[0]["temperature"]
    for p in res.data["points"]:
        if p["temperature"] == T0:
            p["field_asym_T"] = [-1.0, -0.5, 0.5, 1.0]
            p["R_asym"] = [-1e-3, -5e-4, 5e-4, 1e-3]     # envelope ~ +-1.1e-3 (10% pad)
            p["slope_ohm_per_T"] = 0.0
            p["asym_intercept_ohm"] = 2e-4               # inside the envelope
    fig = render_kind(res, "hall_asym_vs_B", spec=_single_T_spec(T0, {"curves": [f"asym:{T0}K"]}))
    ax = fig.axes[0]
    fits = _fit_lines(ax)
    assert len(fits) == 1
    ys = fits[0].get_ydata()
    assert all(y == pytest.approx(2e-4) for y in ys)
    xs = fits[0].get_xdata()
    assert min(xs) == pytest.approx(-1.0) and max(xs) == pytest.approx(1.0)


def test_slope_zero_horizontal_line_skipped_when_outside_envelope(hall_synth_path):
    res = _res(hall_synth_path)
    points = sorted(res.data["points"], key=lambda p: p["temperature"])
    T0 = points[0]["temperature"]
    for p in res.data["points"]:
        if p["temperature"] == T0:
            p["field_asym_T"] = [-1.0, -0.5, 0.5, 1.0]
            p["R_asym"] = [-1e-3, -5e-4, 5e-4, 1e-3]     # envelope ~ +-1.1e-3 (10% pad)
            p["slope_ohm_per_T"] = 0.0
            p["asym_intercept_ohm"] = 5.0                 # far outside the envelope
    fig = render_kind(res, "hall_asym_vs_B", spec=_single_T_spec(T0, {"curves": [f"asym:{T0}K"]}))
    ax = fig.axes[0]
    assert _fit_lines(ax) == []


# ---- PQ-2 Task 2: hall_tdep_stages -> per-stage diagnostic panels -----------------------

def _prov():
    return Provenance(file="x", sha256="ab", app_version=None)


def _stages_result_differing_zsub(n_temps=2):
    """Hand-modified stages: R_zero_sub differs from R_raw for every T -> 3 panels expected."""
    stages = []
    for k in range(n_temps):
        T = 5.0 * (k + 1)
        stages.append({"temperature": T, "fields_T": [0.5, 1.0],
                        "R_raw": [1.0, 2.0], "R_zero_sub": [0.9, 1.8], "R_asym": [0.5, 1.0]})
    return Result(status="ok", data={"probe": "hall_tdep", "stages": stages}, provenance=_prov())


def test_identity_zero_sub_real_data_gives_two_panels(hall_tdep_synth_path):
    # real analyzer output: R_zero_sub == R_raw identically (documented identity) -> Raw +
    # Antisymmetrized only, no separate Zero-subtracted panel.
    res = _tdep_res(hall_tdep_synth_path)
    fig = render_kind(res, "hall_tdep_stages")
    assert len(fig.axes) == 2


def test_differing_zero_sub_gives_three_panels():
    res = _stages_result_differing_zsub(2)
    fig = render_kind(res, "hall_tdep_stages")
    assert len(fig.axes) == 3


def test_panel_titles_and_order():
    res = _stages_result_differing_zsub(2)
    fig = render_kind(res, "hall_tdep_stages")
    titles = [ax.get_title() for ax in fig.axes]
    assert titles == ["Raw", "Zero-sub", "Antisym."]


def test_same_temperature_same_color_across_panels():
    res = _stages_result_differing_zsub(3)
    fig = render_kind(res, "hall_tdep_stages")
    raw_ax, zsub_ax, asym_ax = fig.axes
    # one line per T per panel (3 T's) -> first line in each panel is the first T
    raw_color = raw_ax.lines[0].get_color()
    zsub_color = zsub_ax.lines[0].get_color()
    asym_color = asym_ax.lines[0].get_color()
    assert raw_color == zsub_color == asym_color


def test_leftmost_only_y_label():
    res = _stages_result_differing_zsub(2)
    fig = render_kind(res, "hall_tdep_stages")
    raw_ax, zsub_ax, asym_ax = fig.axes
    assert raw_ax.get_ylabel() == "R (Ω)"
    assert zsub_ax.get_ylabel() == ""
    assert asym_ax.get_ylabel() == ""


def test_every_panel_gets_x_label():
    res = _stages_result_differing_zsub(2)
    fig = render_kind(res, "hall_tdep_stages")
    assert all(ax.get_xlabel() == "|B| (T)" for ax in fig.axes)


def test_deselecting_raw_curves_drops_raw_panel(hall_tdep_synth_path):
    res = _tdep_res(hall_tdep_synth_path)
    T = res.data["stages"][0]["temperature"]
    fig = render_kind(res, "hall_tdep_stages", spec=PlotSpec(curves=[f"asym:{T}K"]))
    assert len(fig.axes) == 1
    assert fig.axes[0].get_title() == "Antisym."


def test_empty_selection_raises(hall_tdep_synth_path):
    res = _tdep_res(hall_tdep_synth_path)
    with pytest.raises(ValueError):
        render_kind(res, "hall_tdep_stages", spec=PlotSpec(curves=[]))


def test_overlay_path_stays_single_axes(hall_tdep_synth_path):
    from cryosweep_core.plotting.catalog import OverlayFile
    res = _tdep_res(hall_tdep_synth_path)
    ov = [OverlayFile(0, "A", None)]
    fig = render_kind([res], "hall_tdep_stages", PlotSpec(), None, overlay=ov)
    assert len(fig.axes) == 1


# ==========================================================================
# PQ-2 Task 3 — shared multi-axes helpers + composite kinds
# ==========================================================================

import matplotlib.colors
from cryosweep_core.plotting.render import _twin_axis, _offset_axis, _merged_legend, _new_fig


def _bare_ax():
    fig = _new_fig(GlobalStyle())
    return fig, fig.add_subplot(111)


def test_twin_axis_color_matches_and_shares_x():
    fig, ax = _bare_ax()
    tax = _twin_axis(ax, GlobalStyle(), "C3")
    assert tax in fig.axes
    assert tax.spines["right"].get_edgecolor()[:3] == pytest.approx(
        matplotlib.colors.to_rgb("C3"))
    assert tax.yaxis.label.get_color() == "C3"


def test_offset_axis_spine_position_and_patch_invisible():
    fig, ax = _bare_ax()
    oax = _offset_axis(ax, GlobalStyle(), "C2", pos=1.18)
    pos_type, pos_val = oax.spines["right"].get_position()
    assert pos_type == "axes" and pos_val == pytest.approx(1.18)
    assert oax.patch.get_visible() is False


def test_merged_legend_combines_handles_from_two_axes():
    fig, ax = _bare_ax()
    l1, = ax.plot([1, 2], [1, 2], color="C0")
    tax = _twin_axis(ax, GlobalStyle(), "C3")
    l2, = tax.plot([1, 2], [2, 1], color="C3")
    _merged_legend(ax, [l1, l2], ["R_H", "n"], GlobalStyle(), PlotSpec())
    leg = ax.get_legend()
    assert leg is not None
    assert [t.get_text() for t in leg.get_texts()] == ["R_H", "n"]


# ---- hall_two_panel ------------------------------------------------------

def test_hall_two_panel_gated_when_no_rho_xx(hall_synth_path):
    res = _res(hall_synth_path)                        # no longitudinal channel configured
    assert get_kind("hall_two_panel").series(res) == []
    with pytest.raises(ValueError):
        render_kind(res, "hall_two_panel")


def test_hall_two_panel_two_axes_left_color_by_T(hall_synth_path):
    res = _res_with_long(hall_synth_path)
    fig = render_kind(res, "hall_two_panel")
    assert len(fig.axes) == 2
    ax_left, ax_right = fig.axes
    colors = {ln.get_color() for ln in ax_left.lines}
    assert len(colors) == len(res.data["points"])       # one colour per temperature
    # hall_synth.dat carries a per-T longitudinal sweep -> right panel is R_xx(B) color-by-T,
    # one series per temperature (not the scalar rho_xx(T) fallback).
    assert ax_right.get_xlabel() == "Field B (T)"
    rlines = [ln for ln in ax_right.lines if ln.get_gid() != "refline"]
    assert len(rlines) == len(res.data["points"])


def test_hall_two_panel_empty_curves_raises(hall_synth_path):
    res = _res_with_long(hall_synth_path)
    with pytest.raises(ValueError):
        render_kind(res, "hall_two_panel", spec=PlotSpec(curves=[]))


def test_hall_two_panel_registered_for_hall_probe():
    from cryosweep_core.registry import build_default_registry
    r = build_default_registry()
    keys = [k.key for k in r.plot_kinds_for("hall")]
    assert "hall_two_panel" in keys
    assert get_kind("hall_two_panel").label == "Hall | Longitudinal"


# ---- hall_tdep_summary ----------------------------------------------------

def test_hall_tdep_summary_j_absent_gives_two_axes(hall_tdep_synth_path):
    res = _tdep_res(hall_tdep_synth_path)               # no current_density_J on real fixture
    fig = render_kind(res, "hall_tdep_summary")
    assert len(fig.axes) == 2


def test_hall_tdep_summary_j_present_gives_three_color_matched_axes(hall_tdep_synth_path):
    res = _tdep_res(hall_tdep_synth_path)
    for i, p in enumerate(res.data["points"]):
        p["current_density_J"] = 1.0e4 + i              # hand-add J (never produced by analyzer)
    fig = render_kind(res, "hall_tdep_summary")
    assert len(fig.axes) == 3
    host, tax, oax = fig.axes
    assert host.yaxis.label.get_color() == "C0"
    assert tax.yaxis.label.get_color() == "C3"
    assert oax.yaxis.label.get_color() == "C2"
    pos_type, pos_val = oax.spines["right"].get_position()
    assert pos_type == "axes" and pos_val == pytest.approx(1.18)


def test_hall_tdep_summary_merged_legend_three_entries(hall_tdep_synth_path):
    res = _tdep_res(hall_tdep_synth_path)
    for i, p in enumerate(res.data["points"]):
        p["current_density_J"] = 1.0e4 + i
    fig = render_kind(res, "hall_tdep_summary")
    leg = fig.axes[0].get_legend()
    assert leg is not None
    assert len(leg.get_texts()) == 3


def test_hall_tdep_summary_gated_when_no_mobility():
    res = Result(status="ok", data={
        "probe": "hall_tdep",
        "points": [{"temperature": 5.0, "R_H": -5e-8, "mobility": None},
                   {"temperature": 10.0, "R_H": -4e-8, "mobility": None}],
        "stages": [],
    }, provenance=_prov())
    assert get_kind("hall_tdep_summary").series(res) == []
    with pytest.raises(ValueError):
        render_kind(res, "hall_tdep_summary")


def test_hall_tdep_summary_gated_when_fewer_than_two_rh_points():
    res = Result(status="ok", data={
        "probe": "hall_tdep",
        "points": [{"temperature": 5.0, "R_H": -5e-8, "mobility": 0.05}],
        "stages": [],
    }, provenance=_prov())
    assert get_kind("hall_tdep_summary").series(res) == []


def test_hall_tdep_summary_exact_mm_png_dims(tmp_path, hall_tdep_synth_path):
    from PIL import Image
    res = _tdep_res(hall_tdep_synth_path)
    style = GlobalStyle()
    fig = render_kind(res, "hall_tdep_summary", style=style)
    p = save_figure(fig, tmp_path / "a.png", style)
    with Image.open(p) as im:
        assert im.size == (round(style.width_mm / 25.4 * style.dpi),
                            round(style.height_mm / 25.4 * style.dpi))


def test_hall_tdep_summary_double_save_byte_identical(tmp_path, hall_tdep_synth_path):
    res = _tdep_res(hall_tdep_synth_path)
    style = GlobalStyle()
    outs = []
    for i in range(2):
        fig = render_kind(res, "hall_tdep_summary", style=style)
        outs.append(save_figure(fig, tmp_path / f"r{i}.png", style).read_bytes())
    assert outs[0] == outs[1]


# ---- hall_rh_n_twin (registered for both probes) --------------------------

def test_hall_rh_n_twin_hall_probe_twinx_log_and_marker_s(hall_synth_path):
    res = _res(hall_synth_path)
    fig = render_kind(res, "hall_rh_n_twin")
    assert len(fig.axes) == 2
    host, tax = fig.axes
    assert tax.get_yscale() == "log"
    rh_line = host.lines[0]
    n_line = tax.lines[0]
    assert rh_line.get_marker() == "s"
    assert n_line.get_marker() == "s"


def test_hall_tdep_rh_n_twin_marker_o(hall_tdep_synth_path):
    res = _tdep_res(hall_tdep_synth_path)
    fig = render_kind(res, "hall_tdep_rh_n_twin")
    assert len(fig.axes) == 2
    host, tax = fig.axes
    assert host.lines[0].get_marker() == "o"
    assert tax.lines[0].get_marker() == "o"


def test_hall_rh_n_twin_gated_fewer_than_two_points():
    res = Result(status="ok", data={
        "probe": "hall",
        "points": [{"temperature": 5.0, "R_H": -5e-8, "carrier_n": 1.0e26}],
    }, provenance=_prov())
    assert get_kind("hall_rh_n_twin").series(res) == []
    with pytest.raises(ValueError):
        render_kind(res, "hall_rh_n_twin")


def test_hall_rh_n_twin_registered_for_both_probes():
    from cryosweep_core.registry import build_default_registry
    r = build_default_registry()
    assert "hall_rh_n_twin" in [k.key for k in r.plot_kinds_for("hall")]
    assert "hall_tdep_rh_n_twin" in [k.key for k in r.plot_kinds_for("hall_tdep")]


# ---- robust view on composite axes (real-data fix: pathological R_H point --------------
# crushing hall_rh_n_twin's left axis) -----------------------------------------------------

def _rh_n_result(rh_values, n_values, temps=None, probe="hall"):
    temps = temps or [10.0 * (i + 1) for i in range(len(rh_values))]
    points = [{"temperature": t, "R_H": rh, "carrier_n": n}
              for t, rh, n in zip(temps, rh_values, n_values)]
    return Result(status="ok", data={"probe": probe, "points": points}, provenance=_prov())


def test_pathological_rh_point_excluded_from_left_axis_ylim():
    # 9 healthy R_H points around -1e-9, one pathological point at -2.9e-4 (real-data bug).
    healthy = [-1.0e-9 - 1.0e-11 * i for i in range(9)]
    rh = healthy + [-2.9e-4]
    n = [1.0e25 + 1.0e22 * i for i in range(10)]
    res = _rh_n_result(rh, n)
    fig = render_kind(res, "hall_rh_n_twin")
    host, tax = fig.axes
    ylo, yhi = host.get_ylim()
    # left-axis view must be bounded by the healthy points' envelope, not the pathological point
    assert ylo > -1.0e-7
    assert yhi > -1.0e-7


def test_log_n_axis_unaffected_by_robust_view():
    healthy = [-1.0e-9 - 1.0e-11 * i for i in range(9)]
    rh = healthy + [-2.9e-4]
    n = [1.0e25 + 1.0e22 * i for i in range(10)]
    res = _rh_n_result(rh, n)
    fig = render_kind(res, "hall_rh_n_twin")
    host, tax = fig.axes
    assert tax.get_yscale() == "log"
    ylo, yhi = tax.get_ylim()
    nmin, nmax = min(n), max(n)
    # log axis must still span the full data decades -- robust view is a no-op here
    assert ylo <= nmin
    assert yhi >= nmax


def test_clean_data_rh_n_twin_ylim_unchanged():
    # no pathological point -- robust view must be a genuine no-op (ylim still contains all pts)
    rh = [-1.0e-9 - 1.0e-11 * i for i in range(10)]
    n = [1.0e25 + 1.0e22 * i for i in range(10)]
    res = _rh_n_result(rh, n)
    fig = render_kind(res, "hall_rh_n_twin")
    host, _tax = fig.axes
    ylo, yhi = host.get_ylim()
    assert ylo <= min(rh)
    assert yhi >= max(rh)


# ---- Opus PQ-2 final-review fix: hall_tdep_stages "Antisymmetrized" title overflows the ----
# fixed-mm canvas on the common 2-panel case (Raw + Antisymmetrized) -------------------------

def test_stage_panel_titles_fit_inside_canvas(hall_tdep_synth_path, tmp_path):
    res = _tdep_res(hall_tdep_synth_path)
    fig = render_kind(res, "hall_tdep_stages")
    style = GlobalStyle()
    out = save_figure(fig, tmp_path / "stages.png", style)
    assert out.exists()

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    for ax in fig.axes:
        extent = ax.title.get_window_extent(renderer=renderer)
        assert extent.x1 <= fig_w_px, (
            f"panel title {ax.title.get_text()!r} overflows canvas: "
            f"x1={extent.x1} > fig width={fig_w_px}"
        )
