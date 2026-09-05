"""PQ-3 Task 2 — vsm_mh PlotKind: M(H) hysteresis + low-field zoom companion panel.
Item 1 ("### `vsm_mh` PlotKind").  Task 1's `loops` field on VSMData is the data layer.
"""
from __future__ import annotations

import dataclasses
import pathlib
import types

import matplotlib
import matplotlib.colors
import numpy as np
import pytest

from cryosweep_core.plotting.catalog import (
    OverlayFile, build_default_layout, get_kind, BUILTIN_PLOTKINDS)
from cryosweep_core.plotting.render import render_kind, NON_DATA_GIDS
from cryosweep_core.plotting.export import save_figure
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle

# ---------------- synthetic result builders ------------------------------

def _loop(T, hmax=13000.0, n=41, falling=False):
    H = np.linspace(-hmax, hmax, n)
    if falling:
        H = H[::-1]
    M = np.tanh(H / 3000.0)
    return {"temperature": float(T), "field_oe": H.tolist(),
            "moment": M.tolist(), "n_points": int(n)}


def _mk(loops):
    return types.SimpleNamespace(data={"probe": "vsm", "loops": list(loops)})


def _series(result):
    return get_kind("vsm_mh").series(result)


def _reflines(ax):
    return [ln for ln in ax.lines if ln.get_gid() == "refline"]


def _data_lines(ax):
    return [ln for ln in ax.lines if ln.get_gid() != "refline"]


# ---------------- series: keys / labels / groups -------------------------

def test_series_unique_T_keys_labels_groups():
    r = _mk([_loop(30.0), _loop(5.0)])
    ss = _series(r)
    assert [s.key for s in ss] == ["mh:30.0K:0", "mh:5.0K:0"]
    assert [s.label for s in ss] == ["30.0 K", "5.0 K"]
    assert [s.group for s in ss] == ["30.0K", "5.0K"]
    assert all(s.default_on for s in ss)
    # x = field_oe, y = moment
    assert ss[0].x == _loop(30.0)["field_oe"]
    assert ss[0].y == _loop(30.0)["moment"]


def test_series_dup_T_suffix_rule():
    # two loops at the same rounded T -> :i index in key, " (2)"-style label on the 2nd
    r = _mk([_loop(30.0), _loop(30.0, falling=True)])
    ss = _series(r)
    assert [s.key for s in ss] == ["mh:30.0K:0", "mh:30.0K:1"]
    assert [s.label for s in ss] == ["30.0 K", "30.0 K (2)"]
    assert [s.group for s in ss] == ["30.0K", "30.0K"]   # same group -> shared colour


def test_series_empty_when_no_loops():
    assert _series(_mk([])) == []
    assert _series(types.SimpleNamespace(data={"probe": "vsm"})) == []


# ---------------- gating: checklist / default layout ---------------------

def test_kind_absent_from_default_layout_without_loops():
    vsm_kinds = [k for k in BUILTIN_PLOTKINDS if k.probe == "vsm"]
    layout = build_default_layout(vsm_kinds, _mk([]))
    assert "vsm_mh" not in {e.kind for e in layout.plots}


def test_kind_present_in_default_layout_with_loops():
    vsm_kinds = [k for k in BUILTIN_PLOTKINDS if k.probe == "vsm"]
    layout = build_default_layout(vsm_kinds, _mk([_loop(30.0)]))
    assert "vsm_mh" in {e.kind for e in layout.plots}


# ---------------- renderer: panels + zoom range --------------------------

def test_two_panels_and_zoom_x_range():
    r = _mk([_loop(30.0, hmax=13000.0), _loop(5.0, hmax=13000.0)])
    fig = render_kind(r, "vsm_mh")
    assert len(fig.axes) == 2
    main_ax, zoom_ax = fig.axes
    lo, hi = zoom_ax.get_xlim()
    # zoom half-width = 10% of max|field| = 1300 Oe
    assert lo == pytest.approx(-1300.0)
    assert hi == pytest.approx(1300.0)
    assert zoom_ax.get_title() == "low field"


def test_reference_lines_on_both_panels_excluded_from_legend():
    r = _mk([_loop(30.0)])
    fig = render_kind(r, "vsm_mh")
    for ax in fig.axes:
        # exactly H=0 axvline + M=0 axhline
        assert len(_reflines(ax)) == 2
    leg = fig.axes[0].get_legend()
    labels = {t.get_text() for t in leg.get_texts()}
    assert labels == {"30.0 K"}       # only the T label; no refline entry


def test_categorical_colors_consistent_across_panels():
    r = _mk([_loop(30.0), _loop(5.0)])
    fig = render_kind(r, "vsm_mh")
    main_ax, zoom_ax = fig.axes
    main_colors = [ln.get_color() for ln in _data_lines(main_ax)]
    zoom_colors = [ln.get_color() for ln in _data_lines(zoom_ax)]
    assert main_colors == zoom_colors
    # two distinct T groups -> two distinct colours
    assert len(set(main_colors)) == 2


def test_connected_lines_present():
    r = _mk([_loop(30.0)])
    fig = render_kind(r, "vsm_mh")
    for ax in fig.axes:
        assert any(ln.get_linestyle() == "-" and ln.get_marker() == "o"
                   for ln in _data_lines(ax))


def test_connect_off_gives_markers_only():
    r = _mk([_loop(30.0)])
    fig = render_kind(r, "vsm_mh", spec=PlotSpec(connect_lines=False))
    for ax in fig.axes:
        assert all(ln.get_linestyle() == "None" for ln in _data_lines(ax))


def test_spec_xmin_xmax_main_only():
    r = _mk([_loop(30.0, hmax=13000.0)])
    fig = render_kind(r, "vsm_mh", spec=PlotSpec(xmin=-5000.0, xmax=5000.0))
    main_ax, zoom_ax = fig.axes
    assert main_ax.get_xlim() == pytest.approx((-5000.0, 5000.0))
    # zoom is unaffected by spec.xmin/xmax -> stays +-10% of max|field| = +-1300
    assert zoom_ax.get_xlim() == pytest.approx((-1300.0, 1300.0))


def test_thousands_ticks_when_enabled():
    r = _mk([_loop(30.0, hmax=13000.0)])
    style = GlobalStyle(thousands_sep=True)
    fig = render_kind(r, "vsm_mh", style=style)
    for ax in fig.axes:
        fmt = ax.xaxis.get_major_formatter()
        assert "," in fmt(10000, 0)      # 10000 -> "10,000"


def test_overlay_single_axes_fallback():
    r = _mk([_loop(30.0)])
    ov = [OverlayFile(0, "A", None), OverlayFile(1, "B", None)]
    fig = render_kind([r, r], "vsm_mh", PlotSpec(), GlobalStyle(), overlay=ov)
    assert len(fig.axes) == 1


def test_value_error_on_empty_selection():
    r = _mk([_loop(30.0)])
    with pytest.raises(ValueError):
        render_kind(r, "vsm_mh", spec=PlotSpec(curves=[]))


def test_x_label_is_oe():
    r = _mk([_loop(30.0)])
    fig = render_kind(r, "vsm_mh")
    assert fig.axes[0].get_xlabel() == "Magnetic Field (Oe)"


# ---------------- export: exact mm + determinism -------------------------

def test_exact_mm_png_dims_default_and_override(tmp_path):
    from PIL import Image
    r = _mk([_loop(30.0)])
    style = GlobalStyle()
    fig = render_kind(r, "vsm_mh", style=style)
    p = save_figure(fig, tmp_path / "a.png", style)
    with Image.open(p) as im:
        assert im.size == (round(style.width_mm / 25.4 * style.dpi),
                           round(style.height_mm / 25.4 * style.dpi))
    # per-plot mm override on the spec
    spec = PlotSpec(width_mm=120.0, height_mm=80.0)
    fig2 = render_kind(r, "vsm_mh", spec=spec, style=style)
    p2 = save_figure(fig2, tmp_path / "b.png", style, spec=spec)
    with Image.open(p2) as im:
        assert im.size == (round(120.0 / 25.4 * style.dpi),
                           round(80.0 / 25.4 * style.dpi))


def test_double_save_byte_identical(tmp_path):
    style = GlobalStyle()
    blobs = []
    for i in range(2):
        r = _mk([_loop(30.0), _loop(5.0)])
        fig = render_kind(r, "vsm_mh", style=style)
        blobs.append(save_figure(fig, tmp_path / f"r{i}.png", style).read_bytes())
    assert blobs[0] == blobs[1]


# ---------------- real VSM_N smoke ---------------------------------------

def test_vsm_n_real_file_renders_multiple_loops(vsm_real_path):
    from cryosweep_core.analyzers.mag import VSMAnalyzer
    from cryosweep_core.config import RunConfig
    from cryosweep_core.io.loader import load_dat
    rt = load_dat(str(vsm_real_path))
    rt = dataclasses.replace(
        rt, header=dataclasses.replace(rt.header, molar_mass=300.0, mass_mg=1.1))
    res = VSMAnalyzer().analyze(rt, RunConfig.load(unit_system="CGS"))
    ss = _series(res)
    assert len(ss) >= 3          # >=2 loops at 30 K + >=1 more
    fig = render_kind(res, "vsm_mh")
    assert len(fig.axes) == 2


# =====================================================================
# PQ-3 Task 3 — twin χ/χ⁻¹ (Item 2) + Curie-Weiss journal upgrade (Item 3)
#
# =====================================================================

from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.fitting.models import CurieWeissModel

FIX = pathlib.Path(__file__).parent / "fixtures"


def _cw_data(theta=-10.0, C=0.5, chi0=0.0, n=60, tmin=12.0, tmax=300.0,
             unit="mol*Oe/emu", cunit="emu*K/(mol*Oe)", chi0unit="emu/(mol*Oe)",
             with_mod=True):
    """Synthetic VSM result carrying a clean Curie-Weiss curve + fit / fit_modified blocks."""
    T = np.linspace(tmin, tmax, n)
    chi = chi0 + C / (T - theta)
    inv = 1.0 / chi
    d = {"probe": "vsm", "temperature": T.tolist(),
         "chi_molar_cgs": chi.tolist(), "chi_molar_si": (chi * 1e-3).tolist(),
         "inv_chi": inv.tolist(), "inv_chi_unit": unit,
         "fit": {"params": {"C": C, "theta": theta, "mu_eff": 2.827 * C ** 0.5},
                 "units": {"C": cunit, "theta": "K", "mu_eff": "mu_B"}}}
    if with_mod:
        d["fit_modified"] = {"params": {"C": C, "theta": theta, "chi0": chi0,
                                        "mu_eff": 2.827 * C ** 0.5},
                             "units": {"C": cunit, "theta": "K",
                                       "chi0": chi0unit, "mu_eff": "mu_B"}}
    return types.SimpleNamespace(data=d)


def _fit_lines(ax):
    return [ln for ln in ax.lines if ln.get_gid() == "fit"]


def _annot_texts(ax):
    return [t for t in ax.texts if "θ" in t.get_text()]


def _analyze_cw(unit_system="CGS"):
    return VSMAnalyzer().analyze(load_dat(str(FIX / "vsm_synth.dat")),
                                 RunConfig.load(unit_system=unit_system))


# ---- fitting-unit reconcile (grep target: CurieWeissModel units) --------

def test_model_C_unit_reconciled_cgs_si():
    T = np.linspace(12, 300, 40); inv = (T + 10.0) / 0.5
    assert CurieWeissModel().fit(T, inv, unit_system="CGS").units["C"] == "emu*K/(mol*Oe)"
    assert CurieWeissModel().fit(T, inv, unit_system="SI").units["C"] == "m^3*K/mol"
    # modified fit carries the same reconciled C unit
    fm = CurieWeissModel().fit(T, inv, unit_system="CGS", modified=True)
    assert fm.units["C"] == "emu*K/(mol*Oe)"


# ---- Item 3 analyzer: fit_modified present / failure-safe ----------------

def test_fit_modified_present_and_plausible():
    res = _analyze_cw()
    fm = res.data.get("fit_modified")
    assert fm is not None
    p = fm["params"]
    assert abs(p["C"] - 0.5) < 0.05 and abs(p["theta"] + 10.0) < 0.5
    assert fm["r2"] > 0.99
    # ordinary fit untouched / still present
    assert res.data["fit"] is not None


def test_fit_modified_present_on_mpms(mpms_real_path):
    res = VSMAnalyzer().analyze(
        dataclasses.replace(load_dat(str(mpms_real_path)),
                            header=dataclasses.replace(load_dat(str(mpms_real_path)).header,
                                                       molar_mass=500.0, mass_mg=10.0)),
        RunConfig.load(unit_system="CGS"))
    fm = res.data.get("fit_modified")
    assert fm is not None and "chi0" in fm["params"]


def test_fit_modified_failure_none_plus_warning(monkeypatch):
    import cryosweep_core.analyzers.mag as magmod
    orig = CurieWeissModel.fit

    def flaky(self, T, inv, unit_system="CGS", modified=False):
        if modified:
            raise RuntimeError("forced modified-fit failure")
        return orig(self, T, inv, unit_system=unit_system, modified=False)

    monkeypatch.setattr(magmod.CurieWeissModel, "fit", flaky)
    res = _analyze_cw()
    assert res.data["fit"] is not None            # ordinary fit unaffected
    assert res.data.get("fit_modified") is None    # modified failed -> None
    assert res.status in ("ok", "low_confidence")  # never errored
    assert any("modified Curie-Weiss fit failed" in w for w in res.warnings)


# ---- Item 2: vsm_chi_t twin axis ----------------------------------------

def test_chi_t_twin_axes_color_matched():
    fig = render_kind(_cw_data(), "vsm_chi_t")
    assert len(fig.axes) == 2
    ax, tax = fig.axes
    # left axis χ = C0, right twin χ⁻¹ = C3 (spine + label colour-matched)
    c0 = matplotlib.colors.to_rgba("C0"); c3 = matplotlib.colors.to_rgba("C3")
    assert matplotlib.colors.to_rgba(ax.yaxis.label.get_color()) == c0
    assert matplotlib.colors.to_rgba(tax.yaxis.label.get_color()) == c3
    assert matplotlib.colors.to_rgba(tax.spines["right"].get_edgecolor()) == c3
    chi_ln = [ln for ln in ax.lines if ln.get_gid() != "refline"][0]
    inv_ln = [ln for ln in tax.lines if ln.get_gid() != "refline"][0]
    assert matplotlib.colors.to_rgba(chi_ln.get_color()) == c0
    assert matplotlib.colors.to_rgba(inv_ln.get_color()) == c3


def test_chi_t_inv_toggle_off_single_axes():
    # deselect inv_chi -> plain single-axes χ, no dead right spine
    fig = render_kind(_cw_data(), "vsm_chi_t", spec=PlotSpec(curves=["curve"]))
    assert len(fig.axes) == 1
    assert fig.axes[0].get_ylabel() == "χ (emu/(mol·Oe))"


def test_chi_t_unit_true_labels_cgs_and_si():
    fig = render_kind(_cw_data(unit="mol*Oe/emu"), "vsm_chi_t")
    assert fig.axes[0].get_ylabel() == "χ (emu/(mol·Oe))"
    assert fig.axes[1].get_ylabel() == "1/χ (mol·Oe/emu)"
    fig_si = render_kind(_cw_data(unit="mol/m^3"), "vsm_chi_t")
    assert fig_si.axes[0].get_ylabel() == "χ (m³/mol)"
    assert fig_si.axes[1].get_ylabel() == "1/χ (mol/m³)"


def test_chi_t_overlay_single_axes_chi_only():
    r = _cw_data()
    ov = [OverlayFile(0, "A", None), OverlayFile(1, "B", None)]
    fig = render_kind([r, r], "vsm_chi_t", PlotSpec(), GlobalStyle(), overlay=ov)
    assert len(fig.axes) == 1
    # only χ drawn (one per file), no χ⁻¹ series on the single axis
    data_lines = [ln for ln in fig.axes[0].lines if ln.get_gid() != "refline"]
    assert len(data_lines) == 2


def test_chi_t_double_save_and_exact_mm(tmp_path):
    from PIL import Image
    style = GlobalStyle()
    blobs = [save_figure(render_kind(_cw_data(), "vsm_chi_t", style=style),
                         tmp_path / f"c{i}.png", style).read_bytes() for i in range(2)]
    assert blobs[0] == blobs[1]
    with Image.open(tmp_path / "c0.png") as im:
        assert im.size == (round(style.width_mm / 25.4 * style.dpi),
                           round(style.height_mm / 25.4 * style.dpi))


# ---- Item 3: inverse_chi CW journal upgrade -----------------------------

def test_inverse_chi_two_models_solid_and_dashed_gray():
    fig = render_kind(_cw_data(), "inverse_chi")
    ax = fig.axes[0]
    fits = _fit_lines(ax)
    assert len(fits) == 2
    solid = [ln for ln in fits if ln.get_linestyle() == "-"]
    dashed = [ln for ln in fits if ln.get_linestyle() == "--"]
    assert len(solid) == 1 and len(dashed) == 1
    assert dashed[0].get_color() == "0.45"       # dashed grey modified CW


def test_inverse_chi_fit_lines_subset_cw_only():
    fig = render_kind(_cw_data(), "inverse_chi", spec=PlotSpec(fit_lines=("cw",)))
    fits = _fit_lines(fig.axes[0])
    assert len(fits) == 1 and fits[0].get_linestyle() == "-"
    # annotation present but WITHOUT the χ₀ line (modified not drawn)
    txt = _annot_texts(fig.axes[0])[0].get_text()
    assert "θ" in txt and "χ₀" not in txt


def test_inverse_chi_fit_lines_empty_hides_both():
    fig = render_kind(_cw_data(), "inverse_chi", spec=PlotSpec(fit_lines=()))
    assert len(_fit_lines(fig.axes[0])) == 0
    # box still shown (fit_line True) — θ/C only
    assert _annot_texts(fig.axes[0])


def test_inverse_chi_fit_line_false_hides_annotation_and_lines():
    fig = render_kind(_cw_data(), "inverse_chi", spec=PlotSpec(fit_line=False))
    assert len(_fit_lines(fig.axes[0])) == 0
    assert _annot_texts(fig.axes[0]) == []


def test_inverse_chi_annotation_content_units_and_chi0():
    txt = _annot_texts(render_kind(_cw_data(), "inverse_chi").axes[0])[0].get_text()
    assert "θ =" in txt and "C =" in txt
    assert "emu*K/(mol*Oe)" in txt              # unit from FitResult.units, not hardcoded
    assert "χ₀ =" in txt and "emu/(mol*Oe)" in txt


def test_inverse_chi_unit_true_label_si():
    fig = render_kind(_cw_data(unit="mol/m^3"), "inverse_chi")
    assert fig.axes[0].get_ylabel() == "1/χ (mol/m³)"


def test_inverse_chi_reference_line_tn_vertical_renders():
    from cryosweep_core.plotting.spec import ReferenceLine
    spec = PlotSpec(reference_lines=[ReferenceLine(axis="v", value=50.0,
                                                   linestyle="--", label="T_N")])
    fig = render_kind(_cw_data(), "inverse_chi", spec=spec)
    vlines = [ln for ln in fig.axes[0].lines
              if ln.get_gid() == "refline" and ln.get_xdata()[0] == 50.0]
    assert len(vlines) == 1 and vlines[0].get_linestyle() == "--"


def test_inverse_chi_double_save_identity(tmp_path):
    style = GlobalStyle()
    blobs = [save_figure(render_kind(_cw_data(), "inverse_chi", style=style),
                         tmp_path / f"i{i}.png", style).read_bytes() for i in range(2)]
    assert blobs[0] == blobs[1]


def test_inverse_chi_annotation_no_legend_overlap_mpms(mpms_real_path):
    res = VSMAnalyzer().analyze(
        dataclasses.replace(load_dat(str(mpms_real_path)),
                            header=dataclasses.replace(load_dat(str(mpms_real_path)).header,
                                                       molar_mass=500.0, mass_mg=10.0)),
        RunConfig.load(unit_system="CGS"))
    fig = render_kind(res, "inverse_chi")
    ax = fig.axes[0]
    fig.canvas.draw()
    leg = ax.get_legend()
    assert leg is not None
    ann = _annot_texts(ax)[0]
    lb = leg.get_window_extent(); ab = ann.get_window_extent()
    # bboxes must not overlap (Bbox.overlaps is True even on edge-touch -> require strict gap)
    assert not (ab.x0 < lb.x1 and lb.x0 < ab.x1 and ab.y0 < lb.y1 and lb.y0 < ab.y1)


# =====================================================================
# PQ-3 Task 4 — warming/cooling ramp-split series (ZFC/FC arrows) on the
# M(T)-family kinds.  Spec Item 4.
#
# REALITY (verified against the shipped analyzer): the analyzer exports the
# single WIDEST temperature segment, which is monotone by construction, so
# `ramps` has length 1 for every current real fixture (vsm_synth, VSM_N,
# MPMS ZFC/FC). The multi-ramp split is therefore driven by synthetic 2-ramp
# results here (forward-looking for when the analyzer is grown to export both
# ZFC+FC ramps — recognized-deferred). Single-ramp behaviour stays byte-
# identical (the vsm_synth PNG goldens in test_mag_loops enforce it).
# =====================================================================

from cryosweep_core.plotting.catalog import get_kind as _get_kind      # noqa: E402  (local alias)


def _ramp_data(with_fit=False, n=12, unit="mol*Oe/emu"):
    """Synthetic 2-ramp (warming then cooling) VSM M(T) result carrying every
    M(T)-family array + a `ramps` list splitting the flat arrays in half."""
    T_up = np.linspace(5.0, 100.0, n)
    T_dn = np.linspace(100.0, 5.0, n)
    T = np.concatenate([T_up, T_dn])
    chi = 0.5 / (T + 10.0)
    inv = 1.0 / chi
    mom = np.tanh(T / 50.0)
    d = {"probe": "vsm", "temperature": T.tolist(),
         "moment_per_fu": mom.tolist(),
         "chi_molar_cgs": chi.tolist(), "chi_molar_si": (chi * 1e-3).tolist(),
         "inv_chi": inv.tolist(), "inv_chi_unit": unit,
         "ramps": [{"direction": "warming", "i0": 0, "i1": n - 1},
                   {"direction": "cooling", "i0": n, "i1": 2 * n - 1}]}
    if with_fit:
        d["fit"] = {"params": {"C": 0.5, "theta": -10.0},
                    "units": {"C": "emu*K/(mol*Oe)", "theta": "K"}}
        d["fit_modified"] = {"params": {"C": 0.5, "theta": -10.0, "chi0": 0.0},
                             "units": {"C": "emu*K/(mol*Oe)", "theta": "K",
                                       "chi0": "emu/(mol*Oe)"}}
    return types.SimpleNamespace(data=d)


def _single_ramp_data(ykey="moment_per_fu"):
    return types.SimpleNamespace(data={
        "probe": "vsm", "temperature": [1.0, 2.0, 3.0], ykey: [1.0, 2.0, 3.0],
        "chi_molar_cgs": [1.0, 2.0, 3.0], "inv_chi": [1.0, 0.5, 0.33],
        "inv_chi_unit": "mol*Oe/emu",
        "ramps": [{"direction": "warming", "i0": 0, "i1": 2}]})


def _nonfit_data_lines(ax):
    return [ln for ln in ax.lines if ln.get_gid() not in NON_DATA_GIDS]


# ---- series layer: keys / labels / linestyles / groups -------------------

def test_moment_t_ramp_split_keys_labels_linestyles():
    ss = _get_kind("vsm_moment_t").series(_ramp_data())
    assert [s.key for s in ss] == ["curve:r0", "curve:r1"]
    assert [s.label for s in ss] == ["Moment ↑", "Moment ↓"]
    assert [s.linestyle for s in ss] == ["-", "--"]
    assert [s.group for s in ss] == ["curve", "curve"]          # shared colour
    # halves cover the flat array with no overlap
    assert ss[0].x == _ramp_data().data["temperature"][:12]
    assert ss[1].x == _ramp_data().data["temperature"][12:]


def test_chi_t_product_ramp_split():
    ss = _get_kind("vsm_chi_t_product").series(_ramp_data())
    assert [s.key for s in ss] == ["curve:r0", "curve:r1"]
    assert [s.label for s in ss] == ["χT ↑", "χT ↓"]
    assert [s.linestyle for s in ss] == ["-", "--"]


def test_inverse_chi_ramp_split_series():
    ss = _get_kind("inverse_chi").series(_ramp_data())
    assert [s.key for s in ss] == ["curve:r0", "curve:r1"]
    assert [s.label for s in ss] == ["1/χ ↑", "1/χ ↓"]


def test_chi_t_ramp_split_both_quantities():
    ss = _get_kind("vsm_chi_t").series(_ramp_data())
    keys = [s.key for s in ss]
    assert keys == ["curve:r0", "curve:r1", "inv_chi:r0", "inv_chi:r1"]
    labels = [s.label for s in ss]
    assert labels == ["χ ↑", "χ ↓", "1/χ ↑", "1/χ ↓"]
    # χ ramps share group "curve"; inv_chi ramps share group "inv_chi"
    assert [s.group for s in ss] == ["curve", "curve", "inv_chi", "inv_chi"]
    inv = [s for s in ss if s.key.startswith("inv_chi")]
    assert all(s.role == "inv_chi" for s in inv)


# ---- single-ramp: exactly today's single series (byte-identity) ----------

def test_moment_t_single_ramp_no_suffix():
    ss = _get_kind("vsm_moment_t").series(_single_ramp_data())
    assert len(ss) == 1
    s = ss[0]
    assert s.key == "curve" and s.label == "Moment"
    assert s.group is None and s.linestyle is None


def test_chi_t_single_ramp_unchanged():
    ss = _get_kind("vsm_chi_t").series(_single_ramp_data("chi_molar_cgs"))
    # one χ + one inv_chi series, original keys, no :r suffix, no linestyle
    assert [s.key for s in ss] == ["curve", "inv_chi"]
    assert all(s.linestyle is None for s in ss)


def test_no_ramps_key_treated_as_single():
    r = types.SimpleNamespace(data={"probe": "vsm", "temperature": [1.0, 2.0],
                                     "moment_per_fu": [1.0, 2.0]})   # no ramps key
    ss = _get_kind("vsm_moment_t").series(r)
    assert len(ss) == 1 and ss[0].key == "curve" and ss[0].linestyle is None


# ---- render: shared colour + solid/dashed linestyle ----------------------

def test_moment_t_render_shared_colour_solid_dashed():
    fig = render_kind(_ramp_data(), "vsm_moment_t")
    lines = _data_lines(fig.axes[0])
    assert len(lines) == 2
    assert lines[0].get_color() == lines[1].get_color()          # one colour per quantity
    assert {ln.get_linestyle() for ln in lines} == {"-", "--"}    # warming solid, cooling dashed
    labels = {ln.get_label() for ln in lines}
    assert labels == {"Moment ↑", "Moment ↓"}


def test_moment_t_connect_off_markers_identical():
    fig = render_kind(_ramp_data(), "vsm_moment_t", spec=PlotSpec(connect_lines=False))
    lines = _data_lines(fig.axes[0])
    assert len(lines) == 2
    assert all(ln.get_linestyle() == "None" for ln in lines)      # markers only
    assert all(ln.get_marker() == "o" for ln in lines)
    assert lines[0].get_color() == lines[1].get_color()


def test_chi_t_render_ramp_split_colours_and_styles():
    fig = render_kind(_ramp_data(), "vsm_chi_t")
    assert len(fig.axes) == 2
    ax, tax = fig.axes
    c0 = matplotlib.colors.to_rgba("C0"); c3 = matplotlib.colors.to_rgba("C3")
    chi_lines = _data_lines(ax); inv_lines = _data_lines(tax)
    assert len(chi_lines) == 2 and len(inv_lines) == 2
    assert all(matplotlib.colors.to_rgba(ln.get_color()) == c0 for ln in chi_lines)
    assert all(matplotlib.colors.to_rgba(ln.get_color()) == c3 for ln in inv_lines)
    assert {ln.get_linestyle() for ln in chi_lines} == {"-", "--"}
    assert {ln.get_linestyle() for ln in inv_lines} == {"-", "--"}
    assert {ln.get_label() for ln in chi_lines} == {"χ ↑", "χ ↓"}


def test_inverse_chi_ramp_split_fit_lines_from_full_arrays():
    # data series split into 2 ramps, but the 2 CW fit lines + annotation are
    # computed from the FULL flat arrays and are NOT per-ramp.
    fig = render_kind(_ramp_data(with_fit=True), "inverse_chi")
    ax = fig.axes[0]
    assert len(_nonfit_data_lines(ax)) == 2          # two ramp data series
    assert len(_fit_lines(ax)) == 2                   # CW + modified CW, unchanged
    assert _annot_texts(ax)                            # θ/C box present


def test_inverse_chi_branch_toggle_keeps_fit_lines():
    # toggle one ZFC/FC branch off -> one data series, fit lines/annotation untouched
    fig = render_kind(_ramp_data(with_fit=True), "inverse_chi",
                      spec=PlotSpec(curves=["curve:r0"]))
    ax = fig.axes[0]
    assert len(_nonfit_data_lines(ax)) == 1
    assert len(_fit_lines(ax)) == 2
    assert _annot_texts(ax)


def test_ramp_split_double_save_identity(tmp_path):
    style = GlobalStyle()
    blobs = [save_figure(render_kind(_ramp_data(), "vsm_moment_t", style=style),
                         tmp_path / f"m{i}.png", style).read_bytes() for i in range(2)]
    assert blobs[0] == blobs[1]


def test_ramp_split_determinism_two_renders():
    a = _get_kind("vsm_chi_t").series(_ramp_data())
    b = _get_kind("vsm_chi_t").series(_ramp_data())
    assert [(s.key, s.label, s.linestyle, s.x, s.y) for s in a] == \
           [(s.key, s.label, s.linestyle, s.x, s.y) for s in b]


# =====================================================================
# PQ-3 t_blocks-driven ramp split — the M(T)-family kinds now source the
# warming/cooling split from `t_blocks` (per-temperature-block arrays), so a
# real ZFC/FC file shows BOTH branches even though the flat `ramps` array only
# holds the single widest monotone segment (length 1). Fallback (t_blocks
# absent/singleton): exactly today's single series -> vsm_synth goldens hold.
# =====================================================================

def _tblk(direction, field_oe, T, *, unit="mol*Oe/emu"):
    """One synthetic t_block dict (warming/cooling ramp at a held field)."""
    T = np.asarray(T, float)
    chi = 0.5 / (T + 10.0)
    return {"direction": direction, "field_oe": float(field_oe),
            "temperature": T.tolist(), "moment": np.tanh(T / 50.0).tolist(),
            "chi": chi.tolist(), "inv_chi": (1.0 / chi).tolist()}


def _tblocks_same_field(n=10, field_oe=500.0, with_fit=False):
    """ZFC/FC at ONE field: two t_blocks (warming then cooling), single setpoint."""
    T_up = np.linspace(5.0, 100.0, n); T_dn = np.linspace(100.0, 5.0, n)
    T = np.concatenate([T_up, T_dn]); chi = 0.5 / (T + 10.0)
    d = {"probe": "vsm", "temperature": T.tolist(),
         "moment_per_fu": np.tanh(T / 50.0).tolist(),
         "chi_molar_cgs": chi.tolist(), "chi_molar_si": (chi * 1e-3).tolist(),
         "inv_chi": (1.0 / chi).tolist(), "inv_chi_unit": "mol*Oe/emu",
         "ramps": [{"direction": "warming", "i0": 0, "i1": 2 * n - 1}],   # flat = 1 monotone-ish
         "t_blocks": [_tblk("warming", field_oe, T_up),
                      _tblk("cooling", field_oe, T_dn)]}
    if with_fit:
        d["fit"] = {"params": {"C": 0.5, "theta": -10.0},
                    "units": {"C": "emu*K/(mol*Oe)", "theta": "K"}}
        d["fit_modified"] = {"params": {"C": 0.5, "theta": -10.0, "chi0": 0.0},
                             "units": {"C": "emu*K/(mol*Oe)", "theta": "K",
                                       "chi0": "emu/(mol*Oe)"}}
    return types.SimpleNamespace(data=d)


def _tblocks_multi_field():
    """500 Oe ZFC/FC (2 dirs) + 40000 Oe single cooling -> multiple field setpoints."""
    T_up = np.linspace(5.0, 100.0, 8); T_dn = np.linspace(100.0, 5.0, 8)
    T_hi = np.linspace(100.0, 5.0, 8)
    allT = np.concatenate([T_up, T_dn, T_hi]); chi = 0.5 / (allT + 10.0)
    d = {"probe": "vsm", "temperature": allT.tolist(),
         "moment_per_fu": np.tanh(allT / 50.0).tolist(),
         "chi_molar_cgs": chi.tolist(), "chi_molar_si": (chi * 1e-3).tolist(),
         "inv_chi": (1.0 / chi).tolist(), "inv_chi_unit": "mol*Oe/emu",
         "t_blocks": [_tblk("warming", 500.0, T_up),
                      _tblk("cooling", 500.0, T_dn),
                      _tblk("cooling", 40000.0, T_hi)]}
    return types.SimpleNamespace(data=d)


def _tblocks_singleton():
    T = np.linspace(5.0, 100.0, 12)
    d = {"probe": "vsm", "temperature": T.tolist(),
         "moment_per_fu": np.tanh(T / 50.0).tolist(),
         "chi_molar_cgs": (0.5 / (T + 10.0)).tolist(),
         "inv_chi": ((T + 10.0) / 0.5).tolist(), "inv_chi_unit": "mol*Oe/emu",
         "t_blocks": [_tblk("warming", 500.0, T)]}
    return types.SimpleNamespace(data=d)


# ---- same-field ZFC/FC: {base}:r{j} keys, {base} ↑/↓ labels, shared group ----

def test_tblocks_same_field_moment_split():
    ss = _get_kind("vsm_moment_t").series(_tblocks_same_field())
    assert [s.key for s in ss] == ["curve:r0", "curve:r1"]
    assert [s.label for s in ss] == ["Moment ↑", "Moment ↓"]
    assert [s.linestyle for s in ss] == ["-", "--"]
    assert [s.group for s in ss] == ["curve", "curve"]        # shared colour per quantity
    # x/y sourced from the t_blocks (not the flat arrays)
    assert ss[0].x == _tblocks_same_field().data["t_blocks"][0]["temperature"]
    assert ss[1].y == _tblocks_same_field().data["t_blocks"][1]["moment"]


def test_tblocks_same_field_inverse_chi_and_product():
    ss = _get_kind("inverse_chi").series(_tblocks_same_field())
    assert [s.label for s in ss] == ["1/χ ↑", "1/χ ↓"]
    pp = _get_kind("vsm_chi_t_product").series(_tblocks_same_field())
    assert [s.label for s in pp] == ["χT ↑", "χT ↓"]
    # χT = chi * T, taken per t_block
    b0 = _tblocks_same_field().data["t_blocks"][0]
    assert pp[0].y == [c * t for c, t in zip(b0["chi"], b0["temperature"])]


def test_tblocks_same_field_chi_t_both_quantities():
    ss = _get_kind("vsm_chi_t").series(_tblocks_same_field())
    assert [s.key for s in ss] == ["curve:r0", "curve:r1", "inv_chi:r0", "inv_chi:r1"]
    assert [s.label for s in ss] == ["χ ↑", "χ ↓", "1/χ ↑", "1/χ ↓"]
    assert [s.group for s in ss] == ["curve", "curve", "inv_chi", "inv_chi"]
    assert all(s.role == "inv_chi" for s in ss if s.key.startswith("inv_chi"))


# ---- multiple field setpoints: per-(field, direction), "{field} Oe ↑" labels ----

def test_tblocks_multi_field_labels_and_groups():
    ss = _get_kind("vsm_moment_t").series(_tblocks_multi_field())
    assert [s.label for s in ss] == ["500 Oe ↑", "500 Oe ↓", "40000 Oe ↓"]
    assert [s.linestyle for s in ss] == ["-", "--", "--"]
    # each field is one colour group; ↑/↓ within a field share the field group
    assert [s.group for s in ss] == ["500Oe", "500Oe", "40000Oe"]
    keys = [s.key for s in ss]
    assert keys == ["curve:500:warming", "curve:500:cooling", "curve:40000:cooling"]


def test_tblocks_multi_field_render_colours_by_field():
    fig = render_kind(_tblocks_multi_field(), "vsm_moment_t")
    lines = _data_lines(fig.axes[0])
    assert len(lines) == 3
    c = [ln.get_color() for ln in lines]
    assert c[0] == c[1]           # 500 Oe ↑/↓ share a colour
    assert c[0] != c[2]           # 40000 Oe distinct colour
    assert {lines[0].get_linestyle(), lines[1].get_linestyle()} == {"-", "--"}


# ---- fallback: singleton/absent t_blocks -> exactly today's single series ----

def test_tblocks_singleton_single_series_no_suffix():
    for kind in ("vsm_moment_t", "vsm_chi_t_product", "inverse_chi"):
        ss = _get_kind(kind).series(_tblocks_singleton())
        assert [s.key for s in ss] == ["curve"], kind
        assert ss[0].linestyle is None and ss[0].group is None, kind


def test_tblocks_render_shared_colour_solid_dashed():
    fig = render_kind(_tblocks_same_field(), "vsm_moment_t")
    lines = _data_lines(fig.axes[0])
    assert len(lines) == 2
    assert lines[0].get_color() == lines[1].get_color()
    assert {ln.get_linestyle() for ln in lines} == {"-", "--"}


def test_tblocks_inverse_chi_fit_lines_from_flat_arrays():
    fig = render_kind(_tblocks_same_field(with_fit=True), "inverse_chi")
    ax = fig.axes[0]
    assert len(_nonfit_data_lines(ax)) == 2       # two ramp data series (from t_blocks)
    assert len(_fit_lines(ax)) == 2                # CW + modified CW from flat arrays, unchanged
    assert _annot_texts(ax)


def test_tblocks_determinism_two_calls():
    a = _get_kind("vsm_chi_t").series(_tblocks_multi_field())
    b = _get_kind("vsm_chi_t").series(_tblocks_multi_field())
    assert [(s.key, s.label, s.linestyle, s.group, s.x, s.y) for s in a] == \
           [(s.key, s.label, s.linestyle, s.group, s.x, s.y) for s in b]


def test_tblocks_double_save_identity(tmp_path):
    style = GlobalStyle()
    blobs = [save_figure(render_kind(_tblocks_same_field(), "vsm_moment_t", style=style),
                         tmp_path / f"t{i}.png", style).read_bytes() for i in range(2)]
    assert blobs[0] == blobs[1]


# ---- real fixtures: the gap this fix closes -----------------------------

def test_mpms_tblocks_render_zfc_fc_split(mpms_real_path):
    res = VSMAnalyzer().analyze(
        dataclasses.replace(load_dat(str(mpms_real_path)),
                            header=dataclasses.replace(load_dat(str(mpms_real_path)).header,
                                                       molar_mass=683.22, mass_mg=12.0)),
        RunConfig.load(unit_system="CGS"))
    # flat ramps is still length-1 (widest monotone segment); the split now comes from t_blocks
    assert len(res.data["ramps"]) == 1
    assert len(res.data["t_blocks"]) >= 2
    ss = _get_kind("vsm_moment_t").series(res)
    labels = [s.label for s in ss]
    # MPMS spans 500 Oe (ZFC/FC) + 40000 Oe -> multi-field labels with arrows, both directions
    assert any(l == "500 Oe ↑" for l in labels)
    assert any(l == "500 Oe ↓" for l in labels)
    assert len(ss) >= 2
    fig = render_kind(res, "vsm_moment_t")
    assert len(_data_lines(fig.axes[0])) == len(ss)


def test_vsm_n_tblocks_multifield_render(vsm_real_path):
    from cryosweep_core.analyzers.mag import VSMAnalyzer as _VA
    res = _VA().analyze(
        dataclasses.replace(load_dat(str(vsm_real_path)),
                            header=dataclasses.replace(load_dat(str(vsm_real_path)).header,
                                                       molar_mass=200.0, mass_mg=1.1)),
        RunConfig.load(unit_system="CGS"))
    ss = _get_kind("vsm_moment_t").series(res)
    labels = [s.label for s in ss]
    # multiple field setpoints -> "{field} Oe ↑/↓" labels, >1 distinct field present
    assert all(("Oe ↑" in l) or ("Oe ↓" in l) for l in labels)
    fields = {l.split(" Oe")[0] for l in labels}
    assert len(fields) >= 2


# =====================================================================
# PQ-3 VSM real-data VISUAL GATE fixes (5 defects). TDD failing-first.
# =====================================================================
from matplotlib.ticker import MaxNLocator, FuncFormatter   # noqa: E402


def _tblk_scaled(direction, field_oe, T, scale):
    """One t_block whose moment/chi magnitude is `scale` (multi-magnitude family)."""
    T = np.asarray(T, float)
    chi = scale / (T + 10.0)
    return {"direction": direction, "field_oe": float(field_oe),
            "temperature": T.tolist(), "moment": (scale * np.tanh(T / 50.0)).tolist(),
            "chi": chi.tolist(), "inv_chi": (1.0 / chi).tolist()}


def _tblocks_two_magnitudes():
    """Low-field bulk ~1e-3, high-field bulk ~1.0 -> pooled robust view would hide the
    high-field series; per-line union must keep BOTH visible."""
    T_lo = np.linspace(5.0, 100.0, 40)
    T_hi = np.linspace(5.0, 100.0, 10)
    allT = np.concatenate([T_lo, T_hi])
    d = {"probe": "vsm", "temperature": allT.tolist(),
         "moment_per_fu": np.concatenate([1e-3 * np.tanh(T_lo / 50.0),
                                          1.0 * np.tanh(T_hi / 50.0)]).tolist(),
         "chi_molar_cgs": (0.5 / (allT + 10.0)).tolist(),
         "inv_chi": ((allT + 10.0) / 0.5).tolist(), "inv_chi_unit": "mol*Oe/emu",
         "t_blocks": [_tblk_scaled("warming", 500.0, T_lo, 1e-3),
                      _tblk_scaled("warming", 40000.0, T_hi, 1.0)]}
    return types.SimpleNamespace(data=d)


# ---- A+D: robust view = union of per-line robust ranges -----------------

def test_robust_view_union_keeps_every_series_bulk_visible():
    fig = render_kind(_tblocks_two_magnitudes(), "vsm_moment_t",
                      PlotSpec(yscale="linear"), GlobalStyle(robust_view=True))
    ax = fig.axes[0]
    lo, hi = ax.get_ylim()
    for ln in _data_lines(ax):
        y = np.asarray(ln.get_ydata(), float)
        y = y[np.isfinite(y)]
        med = float(np.median(y))
        assert lo <= med <= hi, f"series bulk (median {med:g}) outside view [{lo:g},{hi:g}]"


def test_robust_view_single_series_unchanged():
    """Union of one == today: the single-series heavy-tail narrowing is byte-identical."""
    from cryosweep_core.plotting.render import _apply_robust_view
    import matplotlib
    from matplotlib.figure import Figure
    def _ylim_for(ys):
        fig = Figure(); ax = fig.subplots()
        ax.plot(np.arange(len(ys)), ys)
        spec = PlotSpec(yscale="linear"); style = GlobalStyle(robust_view=True)
        _apply_robust_view(ax, spec, style)
        return ax.get_ylim()
    ys = np.linspace(3e-5, 5e-5, 200); ys[-7:] = 1.46e-2
    lim = _ylim_for(ys)
    assert lim[1] < 1e-3           # heavy tail narrowed exactly as before


# ---- E: NaN break between concatenated same-(field,direction) blocks -----

def _tblocks_gap_same_ramp():
    """Two warming blocks at the SAME field/direction with a big T gap between them."""
    Ta = np.linspace(5.0, 40.0, 8)
    Tb = np.linspace(150.0, 300.0, 8)      # gap 40 -> 150
    allT = np.concatenate([Ta, Tb])
    d = {"probe": "vsm", "temperature": allT.tolist(),
         "moment_per_fu": np.tanh(allT / 50.0).tolist(),
         "chi_molar_cgs": (0.5 / (allT + 10.0)).tolist(),
         "inv_chi": ((allT + 10.0) / 0.5).tolist(), "inv_chi_unit": "mol*Oe/emu",
         "t_blocks": [_tblk("warming", 5000.0, Ta),
                      _tblk("warming", 5000.0, Tb)]}
    return types.SimpleNamespace(data=d)


def test_concatenated_blocks_carry_nan_break_in_series():
    ss = _get_kind("vsm_moment_t").series(_tblocks_gap_same_ramp())
    assert len(ss) == 1                         # one warming series (blocks concatenated)
    x = np.asarray(ss[0].x, float); y = np.asarray(ss[0].y, float)
    assert np.isnan(x).sum() == 1 and np.isnan(y).sum() == 1
    assert np.isfinite(x).sum() == 16           # all 16 real points preserved


def test_nan_break_survives_connect_sort_in_render():
    fig = render_kind(_tblocks_gap_same_ramp(), "vsm_moment_t",
                      PlotSpec(connect_lines=True), GlobalStyle(connect_lines=True))
    ln = _data_lines(fig.axes[0])[0]
    xd = np.asarray(ln.get_xdata(), float)
    assert np.isnan(xd).any()                   # break survives -> line does not bridge the gap
    finite = xd[np.isfinite(xd)]
    # the NaN sits between the two blocks, not shoved to the end by argsort
    nan_idx = np.where(~np.isfinite(xd))[0]
    assert 0 < nan_idx[0] < len(xd) - 1


# ---- B: 1/chi label prefix in the twin, no legend overlap ----------------

def test_multifield_inv_chi_labels_prefixed():
    ss = _get_kind("vsm_chi_t").series(_tblocks_multi_field())
    chi = [s.label for s in ss if s.role != "inv_chi"]
    inv = [s.label for s in ss if s.role == "inv_chi"]
    assert chi == ["500 Oe ↑", "500 Oe ↓", "40000 Oe ↓"]        # χ side unchanged
    assert inv == ["1/χ 500 Oe ↑", "1/χ 500 Oe ↓", "1/χ 40000 Oe ↓"]


def test_singlefield_inv_chi_labels_not_double_prefixed():
    ss = _get_kind("vsm_chi_t").series(_tblocks_same_field())
    inv = [s.label for s in ss if s.role == "inv_chi"]
    assert inv == ["1/χ ↑", "1/χ ↓"]            # single-field path already carried "1/χ"


def test_chi_t_twin_legend_no_overlap_and_within_canvas():
    fig = render_kind(_tblocks_multi_field(), "vsm_chi_t")
    fig.canvas.draw()
    ax = fig.axes[0]
    leg = ax.get_legend()
    assert leg is not None
    rends = fig.canvas.get_renderer()
    boxes = [t.get_window_extent(rends) for t in leg.get_texts()]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            overlap = not (a.x1 <= b.x0 or b.x1 <= a.x0 or a.y1 <= b.y0 or b.y1 <= a.y0)
            assert not overlap, f"legend texts {i},{j} overlap"
    fw = fig.get_window_extent()
    lb = leg.get_window_extent(rends)
    assert lb.x0 >= fw.x0 - 1 and lb.x1 <= fw.x1 + 1


# ---- C: vsm_mh main-panel x tick locator caps tick count ----------------

def _inview_xlabel_overlaps(fig, ax):
    fig.canvas.draw()
    rends = fig.canvas.get_renderer()
    lo, hi = ax.get_xlim()
    boxes = [t.get_window_extent(rends)
             for t, loc in zip(ax.get_xticklabels(), ax.get_xticks())
             if t.get_text() and lo <= loc <= hi]
    ov = 0
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            if not (a.x1 <= b.x0 or b.x1 <= a.x0):
                ov += 1
    return ov


def test_vsm_mh_main_panel_maxnlocator():
    r = _mk([_loop(5.0, hmax=100000.0), _loop(30.0, hmax=100000.0)])
    fig = render_kind(r, "vsm_mh")
    main_ax, zoom_ax = fig.axes[0], fig.axes[1]
    loc = main_ax.xaxis.get_major_locator()
    # explicit tight cap (not AutoLocator): the ~1.24in panel fits ~2 six-digit labels.
    assert isinstance(loc, MaxNLocator) and loc._nbins == 2
    # zoom panel keeps its default auto locator (already fine)
    assert getattr(zoom_ax.xaxis.get_major_locator(), "_nbins", None) != 2


def test_vsm_mh_maxnlocator_composes_with_thousands():
    r = _mk([_loop(5.0, hmax=100000.0)])
    fig = render_kind(r, "vsm_mh", PlotSpec(), GlobalStyle(thousands_sep=True))
    main_ax = fig.axes[0]
    loc = main_ax.xaxis.get_major_locator()
    assert isinstance(loc, MaxNLocator) and loc._nbins == 2
    assert isinstance(main_ax.xaxis.get_major_formatter(), FuncFormatter)


def test_vsm_mh_no_xtick_collision_on_real_fixture(vsm_real_path):
    res = VSMAnalyzer().analyze(
        dataclasses.replace(load_dat(str(vsm_real_path)),
                            header=dataclasses.replace(load_dat(str(vsm_real_path)).header,
                                                       molar_mass=300.0, mass_mg=1.1)),
        RunConfig.load(unit_system="CGS"))
    for thousands in (False, True):
        fig = render_kind(res, "vsm_mh", PlotSpec(), GlobalStyle(thousands_sep=thousands))
        assert _inview_xlabel_overlaps(fig, fig.axes[0]) == 0, f"thousands={thousands}"
