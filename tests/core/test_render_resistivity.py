import matplotlib; matplotlib.use("Agg")   # set at import time (late .use() breaks under py3.14)
import pathlib, pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.spec import PlotSpec
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
from cryosweep_core.plotting.render import render_for, render_kind

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}

def _act():        # temperature ramps (rho_t): 2 bridges x 1 curve each
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def _hall_res():   # field sweeps (rho_h): 2 bridges x 5 curves each
    return analyze_file(load_dat(str(FIX / "hall_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_rho_t_default_yscale_is_linear_and_forceable_log():
    # D3: ρ(T) headline is linear by default (robust-view is a no-op on log); still forceable to log.
    assert render_for(_act(), PlotSpec()).axes[0].get_yscale() == "linear"
    assert render_kind(_act(), "resistivity_rho_t", PlotSpec(yscale="log")).axes[0].get_yscale() == "log"


def test_rho_t_auto_unit_prefix_scales_data():
    import numpy as np
    ax = render_for(_act(), PlotSpec()).axes[0]
    unit = ax.get_ylabel()
    assert unit in ("ρ (µΩ·cm)", "ρ (mΩ·cm)", "ρ (Ω·cm)")
    factor = {"ρ (µΩ·cm)": 1e6, "ρ (mΩ·cm)": 1e3, "ρ (Ω·cm)": 1.0}[unit]
    ys = np.concatenate([l.get_ydata() for l in ax.lines if l.get_gid() is None])
    scaled_med = float(np.median(np.abs(ys[np.isfinite(ys)])))
    raw_med = scaled_med / factor          # undo the applied factor -> reconstruct raw |ρ| median
    # the ladder must have selected the prefix consistent with the raw median (this ties the
    # label to the actually-applied scale factor, per controller adjudication):
    if unit == "ρ (µΩ·cm)":
        assert raw_med < 1e-3
    elif unit == "ρ (mΩ·cm)":
        assert 1e-3 <= raw_med < 1.0
    else:
        assert raw_med >= 1.0


def test_rho_axis_autoscale_ladder():
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_axis_autoscale
    for med, want_unit, want_f in ((5e-5, "µΩ·cm", 1e6), (5e-2, "mΩ·cm", 1e3), (5.0, "Ω·cm", 1.0)):
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [med, med, med])
        f, unit = _rho_axis_autoscale(ax)
        assert (f, unit) == (want_f, want_unit)
        assert ax.lines[0].get_ydata()[0] == med * want_f
        plt.close(fig)


def test_rho_axis_autoscale_leaves_reflines_untouched():
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_axis_autoscale
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [5e-5, 5e-5, 5e-5])           # data -> µΩ·cm, factor 1e6
    ref = ax.axhline(0.0, gid="refline")             # refline in axes-space, must not scale
    f, unit = _rho_axis_autoscale(ax)
    assert (f, unit) == (1e6, "µΩ·cm")
    assert list(ref.get_ydata()) == [0.0, 0.0]       # untouched
    plt.close(fig)

def test_rho_t_default_matches_default_on_count():
    res = _act()
    n_default_on = sum(1 for s in KINDS["resistivity_rho_t"].series(res) if s.default_on)
    # count DATA series only (gid is None) — fit overlays carry gid="fit" (Task 6)
    n_lines = len([l for l in render_for(res, PlotSpec()).axes[0].lines if l.get_gid() is None])
    assert n_lines == n_default_on            # default render plots exactly the default_on series

def test_rho_t_explicit_curve_subset():
    res = _act()
    keys = [s.key for s in KINDS["resistivity_rho_t"].series(res)]
    fig = render_kind(res, "resistivity_rho_t", PlotSpec(curves=keys[:1]))
    assert len([l for l in fig.axes[0].lines if l.get_gid() is None]) == 1

def test_mr_renders_field_sweeps():
    res = _hall_res()
    series = KINDS["resistivity_mr"].series(res)
    assert series, "expected rho_h (MR) series in hall_synth-as-resistivity"
    fig = render_kind(res, "resistivity_mr", PlotSpec())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Field (Oe)"
    assert len(ax.lines) == sum(1 for s in series if s.default_on)


# ---- Task 8: MR craft — colormap-by-T + black edges + markers + direction arrows ----

def _fake_mr_result(nch=2, directions=(1, -1)):
    class R:                                            # duck-typed: builders only touch .data
        data = {"probe": "resistivity", "bridges": []}
    for ch in range(1, nch + 1):
        curves = [{"held_temp_k": t, "direction": 0, "n_points": 5,
                   "rho_zero_field": 1e-4, "mr_percent_at_max_field": 5.0,
                   "max_abs_field_oe": 9e4, "low_confidence": False,
                   "directions": list(directions),
                   "field": [-9e4, 0.0, 9e4], "rho": [1.1e-4, 1e-4, 1.2e-4]}
                  for t in (10.0, 2.0)]
        R.data["bridges"].append({"channel": ch, "rho_t_curves": [], "rho_h_curves": curves})
    return R()


def test_mr_render_default_colormap_and_edges():
    from cryosweep_core.plotting.render import GlobalStyle
    fig = render_kind(_hall_res(), "resistivity_mr", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    data = [l for l in ax.lines if l.get_gid() is None]
    assert data
    assert all(l.get_markeredgecolor() in ("black", "k", (0.0, 0.0, 0.0, 1.0))
               for l in data)


def test_mr_render_colormap_opt_out_via_palette():
    # Setting palette (or colormap) is the off-switch for the kind-scoped MR default;
    # then no black edge default is forced.
    from cryosweep_core.plotting.render import GlobalStyle
    fig = render_kind(_hall_res(), "resistivity_mr", PlotSpec(),
                      GlobalStyle(palette=["#123456", "#654321"]))
    ax = fig.axes[0]
    data = [l for l in ax.lines if l.get_gid() is None]
    assert all(l.get_markeredgecolor() not in ("black", "k", (0.0, 0.0, 0.0, 1.0))
               for l in data)


def test_mr_render_arrows_and_markers_off_via_spec():
    from cryosweep_core.plotting.render import GlobalStyle
    from cryosweep_core.plotting.catalog import _CH_MARKERS
    res = _fake_mr_result(nch=2, directions=(1, -1))
    fig = render_kind(res, "resistivity_mr",
                      PlotSpec(direction_arrows=False, channel_markers=False), GlobalStyle())
    ax = fig.axes[0]
    assert all("↑" not in l.get_label() for l in ax.lines)
    # channel markers suppressed -> series fall back to the global marker (o)
    assert all(l.get_marker() == "o" for l in ax.lines if l.get_gid() is None)


def test_mr_render_arrows_and_markers_on_by_default():
    from cryosweep_core.plotting.render import GlobalStyle
    from cryosweep_core.plotting.catalog import _CH_MARKERS
    res = _fake_mr_result(nch=2, directions=(1, -1))
    fig = render_kind(res, "resistivity_mr", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    data = [l for l in ax.lines if l.get_gid() is None]
    assert any("↑↓" in l.get_label() for l in data)
    assert {l.get_marker() for l in data} == {_CH_MARKERS[1], _CH_MARKERS[2]}


def test_mr_pct_t_render_zero_line_and_labels():
    from cryosweep_core.plotting.render import GlobalStyle
    fig = render_kind(_fake_mr_result(), "resistivity_mr_pct_t", PlotSpec(), GlobalStyle())
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Temperature (K)"
    assert "%" in ax.get_ylabel()
    assert any(l.get_gid() == "refline" and set(l.get_ydata()) == {0.0} for l in ax.lines)


# ---- Task 6: rho(T) headline — fit overlay + shade + annotation + Tc marker ----

def _sc():         # SC drop fixture: ch1 carries a detected Tc (mid 8 K), ch2 featureless
    return analyze_file(load_dat(str(FIX / "rho_sc_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())


def test_rho_t_fit_overlay_and_shade():
    # act_synth has clean metallic power_law fits on both bridges (no rho0_unresolved flag).
    fig = render_kind(_act(), "resistivity_rho_t", PlotSpec())
    ax = fig.axes[0]
    fits = [l for l in ax.lines if l.get_gid() == "fit"]
    assert fits and all(l.get_linestyle() == "--" for l in fits)
    # the fit-window shade is opt-in since 2026-09-05 (owner: "useful, but switched off by
    # default") -- absent by default, recoverable via the spec flag
    assert not [p for p in ax.patches if p.get_gid() == "refline"]
    shaded = render_kind(_act(), "resistivity_rho_t", PlotSpec(fit_window_shade=True)).axes[0]
    assert any(p.get_gid() == "refline" for p in shaded.patches)   # axvspan shade


def test_rho_t_fit_off_via_spec():
    fig = render_kind(_act(), "resistivity_rho_t", PlotSpec(fit_line=False))
    assert not [l for l in fig.axes[0].lines if l.get_gid() == "fit"]   # fit knob kills fit only


def test_rho_t_annotation_off_via_spec():
    fig = render_kind(_act(), "resistivity_rho_t", PlotSpec(annotation=False))
    assert not fig.axes[0].texts                                        # independent knob


def test_rho_t_annotation_contents():
    fig = render_kind(_act(), "resistivity_rho_t", PlotSpec())
    txt = "\n".join(t.get_text() for t in fig.axes[0].texts)
    assert "µΩ·cm" in txt or "RRR" in txt                           # rho0 and/or RRR present


def test_rho_t_tc_marker_on_sc_synth():
    import numpy as np
    res = _sc()
    fig = render_kind(res, "resistivity_rho_t", PlotSpec())
    ax = fig.axes[0]
    vlines = [l for l in ax.lines if l.get_gid() == "refline"]
    assert any(np.isclose(l.get_xdata()[0], 8.0, atol=1e-6) for l in vlines)
    fig2 = render_kind(res, "resistivity_rho_t", PlotSpec(tc_marker=False))
    assert not [l for l in fig2.axes[0].lines
                if l.get_gid() == "refline" and np.isclose(l.get_xdata()[0], 8.0, atol=1e-6)]


# ---- Task 7: rho(T) headline — low-T inset ----

def test_rho_t_lowt_inset_present_and_suppressible():
    # inset presence/window/suppression, on a file with a clear spot for it (the synthetic
    # rho_sc_synth now DROPS its inset by measurement — pinned below).
    from cryosweep_core.io.loader import load_dat as _ld
    EX = pathlib.Path(__file__).resolve().parents[2] / "examples"
    res = analyze_file(_ld(str(EX / "resistivity_superconductor.dat")),
                       RunConfig.load(), build_default_registry())
    fig = render_kind([res], "resistivity_rho_t", PlotSpec())
    assert len(fig.axes) == 2                                       # main + inset
    inset = next(a for a in fig.axes if a.get_label() == "inset")
    assert inset.get_xlim()[1] >= 13.8                             # window reaches Tc+5 K
    fig2 = render_kind([res], "resistivity_rho_t", PlotSpec(lowt_inset=False))
    assert len(fig2.axes) == 1


def test_rho_t_sc_synth_drops_inset_by_measurement():
    # rho_sc_synth has no clear corner (KNOWN-ISSUES 1 chooser): its ch2 curve ENDS a marker
    # width past the least-bad box, so the endpoint veto drops the inset, with the note.
    fig = render_kind(_sc(), "resistivity_rho_t", PlotSpec())
    assert not [a for a in fig.axes if a.get_label() == "inset"]
    assert [t for ax in fig.axes for t in ax.texts if t.get_gid() == "inset_note"]


def test_rho_t_inset_suppressed_when_no_lowt_data():
    import types
    # synthetic result whose data starts above 30 K -> no inset
    d = {"probe": "resistivity", "bridges": [{"channel": 1, "rho_t_curves": [
        {"held_field_oe": 0.0, "direction": 0, "n_points": 4, "classification": "metallic",
         "temperature": [50.0, 100.0, 200.0, 300.0], "rho": [1e-4, 2e-4, 3e-4, 4e-4]}],
        "rho_h_curves": []}], "capabilities": []}
    res = types.SimpleNamespace(data=d)
    fig = render_kind(res, "resistivity_rho_t", PlotSpec())
    assert len(fig.axes) == 1


# ---- PQ-4 visual-gate fix wave (Task 10) ----

def test_fixwave_d1_strip_spikes_frames_bulk_and_breaks_line():
    # D1: an extreme upper-tail spike on a rho(T) ramp must be stripped from the plotted line
    # (set to NaN -> no full-height connect-line stripe) so the y-view frames the bulk, not the
    # spike. Only the upper tail is touched (a low-T drop to ~0 must survive).
    import numpy as np, matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_strip_spikes
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    bulk = np.linspace(5.0, 90.0, 200)
    y = bulk.copy(); y[-3:] = [3000.0, 8000.0, 14000.0]      # clustered end-of-ramp spikes
    y[0] = 0.0                                                # a genuine low point (must survive)
    fig, ax = plt.subplots()
    ax.plot(np.arange(y.size), y)
    _rho_strip_spikes(ax, PlotSpec(), GlobalStyle())
    out = np.asarray(ax.lines[0].get_ydata(), float)
    assert np.isnan(out[-3:]).all()                          # spikes stripped -> NaN (line breaks)
    assert out[0] == 0.0                                     # low outlier NOT stripped
    finite_max = np.nanmax(out)
    assert finite_max < 200.0                                # remaining data frames the bulk
    plt.close(fig)


def test_fixwave_d1_strip_spikes_noop_when_robust_view_off():
    import numpy as np, matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_strip_spikes
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    y = np.linspace(5.0, 90.0, 200); y[-1] = 14000.0
    fig, ax = plt.subplots(); ax.plot(np.arange(y.size), y)
    _rho_strip_spikes(ax, PlotSpec(robust_view=False), GlobalStyle())
    assert float(np.nanmax(ax.lines[0].get_ydata())) == 14000.0   # untouched
    plt.close(fig)


def test_fixwave_d1_sc_drop_preserved_in_render():
    # D1 regression guard: the superconducting drop headline (ch1 -> ~0) must remain plotted.
    import numpy as np
    ax = render_kind(_sc(), "resistivity_rho_t", PlotSpec()).axes[0]
    ys = np.concatenate([l.get_ydata() for l in ax.lines if l.get_gid() is None])
    ys = ys[np.isfinite(ys)]
    assert float(ys.min()) < 1.0                             # zero-resistance state still shown


def test_fixwave_d1_sc_fixture_no_points_stripped():
    # Controller re-gate regression pin: on rho_sc_synth BOTH channels must plot fully intact —
    # the earlier median+MAD criterion stripped 51/134 points of ch2's smooth monotone ramp
    # (dense low-T sampling put the median at the low-T plateau -> the legitimate upper half
    # read as a "tail"). The quantile-multiple criterion must introduce ZERO NaNs here.
    import numpy as np
    ax = render_kind(_sc(), "resistivity_rho_t", PlotSpec()).axes[0]
    data_lines = [l for l in ax.lines if l.get_gid() is None]
    assert len(data_lines) >= 2                              # ch1 (SC) + ch2 (featureless ramp)
    for l in data_lines:
        y = np.asarray(l.get_ydata(), float)
        assert np.isfinite(y).all(), f"stripped {np.isnan(y).sum()} pts on {l.get_label()!r}"


def test_fixwave_d1_monotone_dense_lowt_ramp_untouched():
    # A smooth monotone rise sampled densely at low T (median pinned at the plateau) must NOT
    # be treated as spiky: max/p90 stays ~1, far below the 8x multiple.
    import numpy as np, matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_strip_spikes
    from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
    T = np.concatenate([np.linspace(2, 10, 79), np.linspace(10.5, 300, 55)])   # 79/134 below 10 K
    y = 50.0 + 30.0 * (T / 300.0)                            # 50 -> 80, rho_sc_synth-ch2-like
    fig, ax = plt.subplots(); ax.plot(T, y)
    _rho_strip_spikes(ax, PlotSpec(), GlobalStyle())
    assert np.isfinite(np.asarray(ax.lines[0].get_ydata(), float)).all()
    plt.close(fig)


def test_fixwave_d2_legend_not_over_inset():
    # D2: with the lower-right inset present, the legend must not sit in the inset's region.
    fig = render_kind(_act(), "resistivity_rho_t", PlotSpec())
    fig.canvas.draw()
    main = fig.axes[0]
    inset = next(a for a in fig.axes[1:] if a.get_position().width < 0.6)
    leg = main.get_legend()
    assert leg is not None
    lb = leg.get_window_extent(); ib = inset.get_window_extent()
    overlap = not (lb.x1 <= ib.x0 or lb.x0 >= ib.x1 or lb.y1 <= ib.y0 or lb.y0 >= ib.y1)
    assert not overlap, "legend overlaps the low-T inset"


def test_fixwave_d3_dedup_fit_label_and_no_edge_clip():
    # D3: repeated 'power-law fit' entries collapse to one; the legend never overflows the figure.
    fig = render_kind(_act(), "resistivity_rho_t", PlotSpec())
    fig.canvas.draw()
    leg = fig.axes[0].get_legend()
    labels = [t.get_text() for t in leg.get_texts()]
    assert labels.count("power-law fit") <= 1
    lb = leg.get_window_extent()
    assert lb.x1 <= fig.bbox.width + 1.0 and lb.x0 >= -1.0    # inside the figure horizontally


def test_fixwave_d4_inset_has_no_xlabel():
    # D4: the low-T inset drops its (clipped) x-axis label; the y-label stays.
    # (act_synth keeps its inset under the occupancy chooser; rho_sc_synth no longer does.)
    fig = render_kind(_act(), "resistivity_rho_t", PlotSpec())
    inset = next(a for a in fig.axes[1:] if a.get_position().width < 0.6)
    assert inset.get_xlabel() == ""
    assert inset.get_ylabel().startswith("ρ (")


def test_fixwave_d5_annotation_unit_matches_axis():
    # D5: ρ₀ in the annotation is stated in the SAME engineering unit as the autoscaled y-axis,
    # and stays non-scientific when compact.
    import types
    from cryosweep_core.plotting.render import _rho_annotation, _rho_axis_autoscale
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.spec import GlobalStyle
    # residual ~2.86e-3 Ω·cm with data in the mΩ·cm band -> "2.86 mΩ·cm", not "2.86e+03 µΩ·cm"
    d = {"bridges": [{"channel": 1, "residual_rho": 2.86e-3,
                      "power_law": {"params": {"n": 0.9}}, "rrr": 1.37}]}
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [3.0e-3, 3.5e-3, 4.0e-3], gid=None)
    f, unit = _rho_axis_autoscale(ax)
    _rho_annotation(ax, d, GlobalStyle(), f, unit)
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert unit == "mΩ·cm"
    assert "2.86 mΩ·cm" in txt
    assert "µΩ·cm" not in txt and "e+0" not in txt
    plt.close(fig)


def _arrhenius_insulator_result(classification="insulating"):
    # dense-at-high-T / sparse-at-low-T Arrhenius ramp rho = 1e-3 * e^{30/T}; the diverging low-T
    # points sit ~600x above the p90 -> the 8x-p90 spike rule fires on them unless the curve is
    # protected as insulating. Field=0, direction=0 -> Series.key "b1:T:0:0".
    import numpy as np, types
    T = np.concatenate([np.linspace(4, 10, 6), np.linspace(12, 300, 120)])
    rho = 1e-3 * np.exp(30.0 / T)
    data = {"bridges": [{"channel": 1, "rho_source": "instrument_column",
                         "rho_t_curves": [{"held_field_oe": 0.0, "direction": 0,
                                           "classification": classification,
                                           "n_points": int(T.size),
                                           "temperature": T.tolist(), "rho": rho.tolist()}],
                         "rho_h_curves": []}]}
    return types.SimpleNamespace(data=data)


def test_fixwave_i2_arrhenius_insulator_keeps_all_points():
    # I2: a genuine diverging low-T insulator must NOT have its real low-T points spike-stripped.
    # robust_view default-True, linear-y default -> the strip pass runs, but the curve is
    # classified 'insulating' so it is protected: ZERO NaNs introduced.
    import numpy as np
    ax = render_kind(_arrhenius_insulator_result("insulating"),
                     "resistivity_rho_t", PlotSpec()).axes[0]
    data_lines = [l for l in ax.lines if l.get_gid() is None]
    assert data_lines
    for l in data_lines:
        y = np.asarray(l.get_ydata(), float)
        assert np.isfinite(y).all(), f"stripped {np.isnan(y).sum()} genuine insulator points"


def test_fixwave_i2_protection_is_what_saves_it():
    # Control: the SAME curve classified 'metallic' (not protected) DOES get its low-T points
    # stripped -> proves the insulating protection, not some other effect, is what preserves them.
    import numpy as np
    ax = render_kind(_arrhenius_insulator_result("metallic"),
                     "resistivity_rho_t", PlotSpec()).axes[0]
    y = np.asarray([l for l in ax.lines if l.get_gid() is None][0].get_ydata(), float)
    assert np.isnan(y).any()


def test_fixwave_m1_annotation_includes_tc_from_second_bridge():
    # M1: ρ0/n/RRR come from bridge 1, but a Tc lives only on bridge 2 -> the annotation must
    # still show that Tc (it is marked by _rho_tc_markers), not drop it after the first bridge.
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_annotation
    from cryosweep_core.plotting.spec import GlobalStyle
    d = {"bridges": [
        {"channel": 1, "residual_rho": 2.0e-6, "rrr": 5.0, "power_law": {"params": {"n": 2.0}},
         "rho_t_curves": [{"tc_mid_k": None, "n_points": 10}]},
        {"channel": 2, "rho_t_curves": [{"tc_mid_k": 8.0, "tc_onset_k": 9.0, "tc_zero_k": 7.0,
                                         "n_points": 20}]}]}
    fig, ax = plt.subplots()
    _rho_annotation(ax, d, GlobalStyle(), 1e6, "µΩ·cm")
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert "RRR = 5" in txt                       # bridge-1 scalar present
    assert "8.00 K" in txt                        # bridge-2 Tc present, not dropped
    plt.close(fig)


def test_fixwave_n1_low_confidence_tc_visible_in_render():
    # N1: a low-confidence Tc (e.g. noisy-plateau BG metal bypassing the narrowness gate) must
    # SHOW its doubt on the figure: "(low confidence)" in the annotation and a visually weaker
    # (dotted, semi-transparent) marker line — not only a flag buried in the data/CSV.
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_annotation, _rho_tc_markers
    from cryosweep_core.plotting.spec import GlobalStyle
    d = {"bridges": [{"channel": 1, "rrr": 3.0,
                      "rho_t_curves": [{"tc_mid_k": 91.2, "tc_onset_k": 140.7, "tc_zero_k": 58.8,
                                        "tc_low_confidence": True, "n_points": 50}]}]}
    fig, ax = plt.subplots()
    ax.plot([0, 300], [1e-6, 1e-4], label="Ch1 0 Oe")
    _rho_tc_markers(ax, d, PlotSpec(), GlobalStyle())
    _rho_annotation(ax, d, GlobalStyle(), 1e6, "µΩ·cm")
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert "(low confidence)" in txt
    marker = next(l for l in ax.lines if l.get_gid() == "refline")
    assert marker.get_linestyle() == ":" and marker.get_alpha() == 0.5   # weaker than confident "--"
    plt.close(fig)


def test_fixwave_n1_high_confidence_tc_render_unchanged():
    # Control pin: the high-confidence path stays byte-identical ("--", lw 0.8, no suffix).
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_annotation, _rho_tc_markers
    from cryosweep_core.plotting.spec import GlobalStyle
    d = {"bridges": [{"channel": 1,
                      "rho_t_curves": [{"tc_mid_k": 8.0, "tc_onset_k": 9.0, "tc_zero_k": 7.0,
                                        "tc_low_confidence": False, "n_points": 50}]}]}
    fig, ax = plt.subplots()
    ax.plot([0, 300], [1e-6, 1e-4], label="Ch1 0 Oe")
    _rho_tc_markers(ax, d, PlotSpec(), GlobalStyle())
    _rho_annotation(ax, d, GlobalStyle(), 1e6, "µΩ·cm")
    txt = "\n".join(t.get_text() for t in ax.texts)
    assert "(low confidence)" not in txt and "K?" not in txt
    marker = next(l for l in ax.lines if l.get_gid() == "refline")
    assert marker.get_linestyle() == "--" and marker.get_alpha() is None
    plt.close(fig)


def test_fixwave_m3_bridge_color_none_when_prefix_absent():
    # M3: on a multi-channel plot, a bridge whose 'Ch{n} ' line is absent must NOT borrow the
    # first line's colour -> return None (style layer picks).
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import _rho_bridge_color
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], label="Ch1 0 Oe")     # only channel 1 drawn
    assert _rho_bridge_color(ax, {"channel": 2}) is None
    assert _rho_bridge_color(ax, {"channel": 1}) is not None
    plt.close(fig)


def _rho_sc():     # SC transition on bridge 1 -> annotation carries the long Tc line
    return analyze_file(load_dat(str(FIX / "rho_sc_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())


def _ann_and_legend_text_hits(fig):
    fig.canvas.draw()
    ax = fig.axes[0]
    ren = fig.canvas.get_renderer()
    leg = ax.get_legend()
    ann = next(t for t in ax.texts if t.get_position() == (0.02, 0.98))
    ab = ann.get_window_extent(ren)
    return [not (b.x1 < ab.x0 or ab.x1 < b.x0 or b.y1 < ab.y0 or ab.y1 < b.y0)
            for b in (t.get_window_extent(ren) for t in leg.get_texts())]


def test_example_d4_legend_dodges_annotation_at_gui_canvas_size():
    """Example-defect D4: on a GUI-card-sized canvas the upper-left rho(T) annotation's Tc line
    ran THROUGH the legend text (screenshot-verified; placement is decided at the 90x70 mm
    creation size and matplotlib's resize re-anchor is blind to text artists). The draw-event
    dodge must relocate the legend once its TEXT glyph boxes are hit at the realized size."""
    from cryosweep_core.plotting.spec import GlobalStyle
    fig = render_kind(_rho_sc(), "resistivity_rho_t", PlotSpec(),
                      GlobalStyle().model_copy(update={"dpi": 100}))
    fig.set_size_inches(3.0, 2.6)             # a Grid-mode card; collision is measured here pre-fix
    fig.canvas.draw()                          # draw 1: dodge callback sees the hit, moves the legend
    assert not any(_ann_and_legend_text_hits(fig)), \
        "legend text overlaps the rho(T) annotation at GUI canvas size"


def test_example_d4_dodge_is_noop_at_creation_size():
    """At the 90x70 mm creation size annotation and legend clear each other; the dodge must not
    move anything (gallery byte-identity depends on this)."""
    fig = render_kind(_rho_sc(), "resistivity_rho_t", PlotSpec())
    fig.canvas.draw()
    leg = fig.axes[0].get_legend()
    assert not getattr(leg, "_cryosweep_dodged", False)
    assert not any(_ann_and_legend_text_hits(fig))


def test_example_d4_inset_hidden_when_canvas_too_small():
    """Follow-up D4: on the GUI-card canvas the 42%x40% low-T inset covered the very curves it
    supplements and crowded the relocated legend (screenshot-verified). The same too-small
    signal that dodges the legend must also hide the inset; at creation size it stays."""
    from cryosweep_core.plotting.spec import GlobalStyle

    def inset_axes_of(fig):
        return [a for a in fig.axes if a.get_label() == "inset"]

    fig = render_kind(_rho_sc(), "resistivity_rho_t", PlotSpec(),
                      GlobalStyle().model_copy(update={"dpi": 100}))
    fig.set_size_inches(3.0, 2.6)
    fig.canvas.draw()
    assert all(not a.get_visible() for a in inset_axes_of(fig)), \
        "inset must hide on a too-small canvas"
    fig2 = render_kind(_rho_sc(), "resistivity_rho_t", PlotSpec())
    fig2.canvas.draw()
    assert all(a.get_visible() for a in inset_axes_of(fig2)), \
        "inset must stay at creation/export size"
