"""Opt-in ±1σ error bands on TTO kinds (spec §3, E2/E6/I2/I3/I5/M2/M8).

DEFAULT-OFF IS THE BYTE-IDENTITY CONTRACT: with PlotSpec.error_band at False every existing
render must be unchanged, so the first block here is a regression pin, not a feature test."""
import matplotlib; matplotlib.use("Agg")       # noqa: E702
import pathlib

import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.plotting.catalog import (BUILTIN_PLOTKINDS, OverlayFile,
                                        series_tto_kappa_t, series_tto_seebeck_t,
                                        series_tto_wf_t, series_tto_zt_t)
from cryosweep_core.plotting.render import _TTO_BAND_KINDS, render_kind
from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec
from tests.core.conftest import real_data

FX = pathlib.Path("tests/core/fixtures")
TTO_KINDS = ("tto_summary_t", "tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_wf_t")


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def _bands(fig):
    return [c for ax in fig.axes for c in ax.collections if c.get_gid() == "errband"]


def _containers(fig):
    return [c for ax in fig.axes for c in ax.containers]


# ---- the default-off contract -----------------------------------------------------------

def test_error_band_defaults_to_false():
    assert PlotSpec().error_band is False


def test_band_kind_allow_list_is_the_pinned_set():
    assert _TTO_BAND_KINDS == {"tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_summary_t"}


@pytest.mark.parametrize("kind", TTO_KINDS)
def test_default_spec_draws_no_bands_and_no_errorbars_single_mode(kind):
    fig = render_kind(_run(FX / "tto_synth.dat"), kind, PlotSpec(), GlobalStyle())
    assert _bands(fig) == []
    assert _containers(fig) == []          # I3: an errorbar would add a container


@pytest.mark.parametrize("kind", TTO_KINDS)
def test_default_spec_draws_no_bands_and_no_errorbars_overlay_mode(kind):
    # I3: TTO is safe in overlay mode only by ROUTING LUCK -- _plot_data's overlay branch uses
    # plain ax.plot. Populating Series.yerr makes that luck load-bearing, so pin it.
    rs = [_run(FX / "tto_synth.dat"), _run(FX / "tto_synth.dat")]
    ov = [OverlayFile(0, "A"), OverlayFile(1, "B")]
    fig = render_kind(rs, kind, PlotSpec(), GlobalStyle(), overlay=ov)
    assert _bands(fig) == []
    assert _containers(fig) == []


# ---- yerr wiring -------------------------------------------------------------------------

def test_measured_series_carry_yerr_and_derived_series_do_not():
    r = _run(FX / "tto_synth.dat")
    for fn in (series_tto_kappa_t, series_tto_seebeck_t, series_tto_zt_t):
        for s in fn(r):
            assert s.yerr is not None, s.key
            assert len(s.yerr) == len(s.y), s.key
    # kappa_e / kappa_ph are derived: no *_std array exists for them.
    for s in series_tto_wf_t(r):
        if s.key.startswith(("kappa_e:", "kappa_ph:")):
            assert s.yerr is None, s.key


def test_yerr_values_are_the_std_column_with_none_holes_as_nan():
    r = _run(FX / "tto_synth.dat")
    s = [x for x in series_tto_kappa_t(r) if x.key == "kappa:0:down"][0]
    curve = [c for c in r.data["curves"] if c["field_oe"] == 0.0][0]
    want = np.array([np.nan if v is None else v for v in curve["kappa_std"]], float)
    assert np.allclose(np.asarray(s.yerr, float), want, equal_nan=True)
    # the synth generator writes every *_std as 1 % of its value
    assert np.allclose(np.asarray(s.yerr, float), 0.01 * np.abs(np.asarray(s.y, float)),
                       rtol=1e-9)


def test_ndarray_std_is_accepted_instead_of_raising_on_array_truthiness():
    # `if std and ...` raises "truth value of an array with more than one element is ambiguous"
    # the day an analyzer emits a *_std as an ndarray (all four are lists today).
    import types
    from cryosweep_core.plotting.catalog import _tto_curve_series
    r = types.SimpleNamespace(data={"curves": [dict(
        t=[1.0, 2.0, 3.0], kappa=[1.0, 2.0, 3.0], kappa_std=np.array([0.1, 0.2, 0.3]),
        field_oe=0.0, direction="down")]})
    s = _tto_curve_series(r, "kappa", "kappa")[0]
    assert np.allclose(np.asarray(s.yerr, float), [0.1, 0.2, 0.3])


def test_series_key_scheme_is_unchanged_by_the_yerr_wiring():
    # The key scheme is a PERSISTED CONTRACT (presets validate saved layouts against it).
    keys = sorted(s.key for s in series_tto_kappa_t(_run(FX / "tto_synth.dat")))
    assert keys == ["kappa:0:down", "kappa:90000:down"]


# ---- bands on ------------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["tto_kappa_t", "tto_seebeck_t", "tto_zt_t"])
def test_error_band_on_draws_one_band_per_series(kind):
    r = _run(FX / "tto_synth.dat")
    fig = render_kind(r, kind, PlotSpec(error_band=True), GlobalStyle())
    n_series = len([ln for ln in fig.axes[0].lines if ln.get_gid() is None])
    # ON THE MAIN AXES: one band per series. `_bands(fig)` walks every axes, and since the
    # final review's I1 fix `tto_kappa_t`'s low-T inset carries its own band too (pinned by
    # test_the_lowt_inset_draws_its_own_band_because_that_is_the_high_sigma_region below).
    main = [c for c in fig.axes[0].collections if c.get_gid() == "errband"]
    assert len(main) == n_series >= 1
    assert _containers(fig) == []          # E6: fill_between, never errorbar


def test_band_styling_is_the_pinned_contract():
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_kappa_t",
                      PlotSpec(error_band=True), GlobalStyle())
    for coll in _bands(fig):
        assert coll.get_alpha() == pytest.approx(0.20)
        assert coll.get_zorder() == pytest.approx(1.5)
        assert coll.get_gid() == "errband"
    assert _bands(fig)


def test_band_brackets_its_own_line_on_a_single_kind():
    # Non-vacuous: a band drawn at the wrong scale (or with e = 0) would NOT bracket.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_kappa_t",
                      PlotSpec(error_band=True, curves=["kappa:0:down"]), GlobalStyle())
    ax = fig.axes[0]
    line = [ln for ln in ax.lines if ln.get_gid() is None][0]
    ly = np.asarray(line.get_ydata(), float)
    ly = ly[np.isfinite(ly)]
    pts = np.concatenate([p.vertices[:, 1] for p in _bands(fig)[0].get_paths()])
    pts = pts[np.isfinite(pts)]
    assert pts.min() < ly.min()
    assert pts.max() > ly.max()


@pytest.mark.parametrize("kind", ["tto_wf_t"])
def test_excluded_kinds_never_draw_a_band_even_when_asked(kind):
    # I2: tto_wf_t's kappa component comes from the SAME builder with a kappa_std array and
    # renders through the SAME _tto_draw, so without the allow-list the decomposition figure
    # would show a band on kappa and none on kappa_e/kappa_ph.
    fig = render_kind(_run(FX / "tto_synth.dat"), kind, PlotSpec(error_band=True),
                      GlobalStyle())
    assert _bands(fig) == []


def test_band_is_reordered_with_the_line_not_left_in_input_order():
    # M2/_connect_sort: the band must be sorted by x the SAME way the line is. A band left in
    # input order on a descending sweep produces a self-crossing polygon whose vertex x order
    # disagrees with the line's.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_kappa_t",
                      PlotSpec(error_band=True, curves=["kappa:0:down"]), GlobalStyle())
    ax = fig.axes[0]
    lx = np.asarray([ln for ln in ax.lines if ln.get_gid() is None][0].get_xdata(), float)
    assert np.all(np.diff(lx) >= 0)                        # the line is x-sorted
    verts = _bands(fig)[0].get_paths()[0].vertices
    upper = verts[:len(lx), 0]
    assert np.all(np.diff(upper) >= 0)                     # ... and so is the band


def _draw_one(series, kind_key="tto_kappa_t", spec=None):
    """Drive `_tto_draw_bands` in plot space (no fixture) and return (ax, band, extent)."""
    from cryosweep_core.plotting.render import _tto_draw_bands
    from matplotlib.figure import Figure
    kind = {k.key: k for k in BUILTIN_PLOTKINDS}[kind_key]
    ax = Figure().add_subplot(111)
    ext = _tto_draw_bands(ax, [(None, series)], kind, spec or PlotSpec(error_band=True),
                          GlobalStyle(), {series.group: "C0"})
    return ax, [c for c in ax.collections if c.get_gid() == "errband"][0], ext


def _half_widths(band):
    """Per-x half-width (hi-lo)/2 of a fill_between polygon, keyed by ascending x."""
    v = np.concatenate([p.vertices for p in band.get_paths()])
    v = v[np.isfinite(v).all(axis=1)]
    return [(x, (v[v[:, 0] == x, 1].max() - v[v[:, 0] == x, 1].min()) / 2.0)
            for x in sorted(set(v[:, 0].tolist()))]


def test_band_sigma_follows_x_through_the_sort_on_a_descending_sweep():
    # THE non-vacuous half of M2/_connect_sort. Every fixture sweep here happens to be
    # ascending, so the fixture test above can only ever see an already-sorted band -- deleting
    # the `_, e = _connect_sort(x_raw, e)` alignment passes it. On a DESCENDING x the sigma must
    # be permuted with the line: sigma 0.3 belongs to x=3, not to x=1. Without the alignment the
    # half-widths come out mirrored about the sweep, [0.3, 0.2, 0.1].
    from cryosweep_core.plotting.catalog import Series
    s = Series(key="kappa:0:down", label="cooling", x=[3.0, 2.0, 1.0], y=[1.0, 1.0, 1.0],
               group="cooling", yerr=[0.3, 0.2, 0.1])
    _, band, _ = _draw_one(s)
    xs = [x for x, _ in _half_widths(band)]
    hw = [h for _, h in _half_widths(band)]
    assert xs == [1.0, 2.0, 3.0]
    assert hw == [pytest.approx(0.1), pytest.approx(0.2), pytest.approx(0.3)]


def test_non_finite_sigma_shrinks_the_band_locally_instead_of_blanking_it():
    # M2: zeroing (not dropping) a bad sigma. Done in plot space so no fixture is needed.
    from cryosweep_core.plotting.catalog import Series
    s = Series(key="kappa:0:down", label="cooling", x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0],
               group="cooling", yerr=[0.1, float("nan"), 0.1])
    ax, band, ext = _draw_one(s)
    ys = band.get_paths()[0].vertices[:, 1]
    assert np.isfinite(ys).all()                           # never blanked
    assert ext == (pytest.approx(0.9), pytest.approx(3.1))
    # ...and the polygon stays ONE unbroken path. get_paths()[0] alone is vacuous here: with the
    # NaN left in place matplotlib splits the fill into 2 paths of 5 vertices each, whose first
    # path is still finite and whose extent is unchanged -- i.e. exactly the "blank a hole in
    # the middle of the ribbon" failure, invisible to the assertions above.
    assert len(band.get_paths()) == 1
    assert len(band.get_paths()[0].vertices) == 2 * len(s.x) + 3


def test_expand_ylim_grows_the_view_for_a_band_wider_than_the_data():
    # The CONSTRUCTED pin for _tto_expand_ylim_for_bands. At the DEFAULT robust_k=8 the
    # fill_between collection already drives autoscale on every current fixture, so stubbing
    # the helper leaves those ylims unchanged and the fixture test below passes by equality.
    # (The helper is NOT merely defensive: at robust_k=1.5 it is load-bearing on the real
    # summary rho panel -- see test_rho_band_survives_a_biting_robust_view.) Construct the case
    # directly: a view set by a helper that reads ax.lines only, with sigma wider than the data.
    from cryosweep_core.plotting.render import _tto_expand_ylim_for_bands
    from cryosweep_core.plotting.catalog import Series
    s = Series(key="kappa:0:down", label="cooling", x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0],
               group="cooling", yerr=[5.0, 5.0, 5.0])
    ax, _, ext = _draw_one(s)
    assert ext == (pytest.approx(-4.0), pytest.approx(8.0))
    ax.set_ylim(0.0, 4.0)                                  # "the lines-only view"
    _tto_expand_ylim_for_bands(ax, PlotSpec(error_band=True), GlobalStyle())
    assert ax.get_ylim() == (pytest.approx(-4.0), pytest.approx(8.0))
    _tto_expand_ylim_for_bands(ax, PlotSpec(error_band=True), GlobalStyle())
    assert ax.get_ylim() == (pytest.approx(-4.0), pytest.approx(8.0))   # idempotent
    ax.set_ylim(-100.0, 100.0)
    _tto_expand_ylim_for_bands(ax, PlotSpec(error_band=True), GlobalStyle())
    assert ax.get_ylim() == (pytest.approx(-100.0), pytest.approx(100.0))  # never contracts


def test_expand_ylim_is_bypassed_on_log_y_and_by_explicit_limits():
    from cryosweep_core.plotting.render import _tto_expand_ylim_for_bands
    from cryosweep_core.plotting.catalog import Series
    s = Series(key="kappa:0:down", label="cooling", x=[1.0, 2.0, 3.0], y=[1.0, 2.0, 3.0],
               group="cooling", yerr=[5.0, 5.0, 5.0])
    ax, _, _ = _draw_one(s)                                # extent (-4, 8), bottom <= 0
    ax.set_yscale("log")
    ax.set_ylim(1.0, 4.0)
    _tto_expand_ylim_for_bands(ax, PlotSpec(error_band=True), GlobalStyle())
    assert ax.get_ylim() == (pytest.approx(1.0), pytest.approx(4.0))
    ax2, _, _ = _draw_one(s)
    ax2.set_ylim(0.0, 4.0)
    _tto_expand_ylim_for_bands(ax2, PlotSpec(error_band=True, ymin=0.0, ymax=4.0), GlobalStyle())
    assert ax2.get_ylim() == (pytest.approx(0.0), pytest.approx(4.0))


def test_band_extent_expands_the_y_view_and_never_contracts_it():
    # I5: _apply_robust_view / _tto_full_view read ax.lines ONLY, so the band would be cropped
    # exactly where the uncertainty is largest. NOTE: at the DEFAULT robust_k this passes by
    # EQUALITY -- the collection already drives autoscale and the ZT ribbon fits inside the
    # full view unaided -- so it observes the end state, it does not pin
    # _tto_expand_ylim_for_bands. The real-render pin is
    # test_rho_band_survives_a_biting_robust_view; the constructed one is the test above.
    r = _run(FX / "tto_synth.dat")
    off = render_kind(r, "tto_zt_t", PlotSpec(), GlobalStyle()).axes[0].get_ylim()
    on = render_kind(r, "tto_zt_t", PlotSpec(error_band=True), GlobalStyle()).axes[0].get_ylim()
    assert on[0] <= off[0] and on[1] >= off[1]
    pts = np.concatenate([p.vertices[:, 1] for p in
                          _bands(render_kind(r, "tto_zt_t", PlotSpec(error_band=True),
                                             GlobalStyle()))[0].get_paths()])
    pts = pts[np.isfinite(pts)]
    assert on[0] <= pts.min() and on[1] >= pts.max()       # the whole band is visible


def test_band_expansion_is_bypassed_on_a_log_y_axis():
    # I2: `set_ylim(bottom<=0)` on a log axis is silently ignored WITH a UserWarning, so
    # without the bypass a user-selected log y half-applies the expansion and spams a warning
    # (-p no:warnings only suppresses the report). Same bypass _apply_robust_view and
    # _tto_full_view carry. Measured: this render emits no warning at all today.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        render_kind(_run(FX / "tto_synth.dat"), "tto_kappa_t",
                    PlotSpec(error_band=True, yscale="log"), GlobalStyle())


def test_explicit_y_limits_still_win_over_the_band_expansion():
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_zt_t",
                      PlotSpec(error_band=True, ymin=0.0, ymax=1e-9), GlobalStyle())
    assert fig.axes[0].get_ylim() == (pytest.approx(0.0), pytest.approx(1e-9))


def test_all_three_summary_panels_get_bands():
    # Task 6: the rho-panel band is drawn AFTER _rho_axis_autoscale (C2) — see the seam block
    # at the end of this module for the assertions that pin its SCALE.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t",
                      PlotSpec(error_band=True), GlobalStyle())
    ax_k, ax_s, ax_r = fig.axes[0], fig.axes[1], fig.axes[2]
    assert [c for c in ax_k.collections if c.get_gid() == "errband"]
    assert [c for c in ax_s.collections if c.get_gid() == "errband"]
    assert [c for c in ax_r.collections if c.get_gid() == "errband"]


def test_bands_render_on_the_real_subset_without_raising():
    for kind in ("tto_kappa_t", "tto_seebeck_t", "tto_zt_t", "tto_summary_t"):
        render_kind(_run(FX / "tto_real_subset.dat"), kind, PlotSpec(error_band=True),
                    GlobalStyle())


# ---- Task 6: the rho-panel seam ----------------------------------------------------------

def _rho_panel(fig):
    return fig.axes[2]


def test_rho_panel_draws_a_rho_band_at_all():
    # Existence only, and named to say so: a stub fill_between satisfies it. It says nothing
    # about the engineering prefix -- the two SCALE assertions below carry that weight.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t",
                      PlotSpec(error_band=True), GlobalStyle())
    assert [c for c in _rho_panel(fig).collections if c.get_gid() == "errband"]


def test_rho_panel_band_brackets_the_rho_line_on_the_RENDERED_axis():
    # C2: _rho_axis_autoscale multiplies `for ln in ax.lines` ONLY. A PolyCollection would
    # never receive the factor and ax.relim() ignores collections, so nothing else would
    # betray a band drawn 1e6x too small. This is a pure RATIO assertion: it cannot be
    # satisfied by a coincidence of units.
    fig = render_kind(_run(FX / "tto_synth.dat"), "tto_summary_t",
                      PlotSpec(error_band=True, curves=["kappa:0:down", "seebeck:0:down",
                                                        "rho:0:down"]),
                      GlobalStyle())
    ax = _rho_panel(fig)
    ly = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in ax.lines
                         if ln.get_gid() is None])
    ly = ly[np.isfinite(ly)]
    band = [c for c in ax.collections if c.get_gid() == "errband"][0]
    by = np.concatenate([p.vertices[:, 1] for p in band.get_paths()])
    by = by[np.isfinite(by)]
    assert by.min() < ly.min()
    assert by.max() > ly.max()
    # ... and by a sane MARGIN, not by 1e-6 of the line value: the synth *_std columns are
    # 1 % of their value, so a band drawn without the 1e6 factor would clear the two
    # assertions above by ~1e-8 relative and this one would catch it.
    assert (by.max() - ly.max()) / ly.max() == pytest.approx(0.01, rel=0.5)


def test_rho_panel_band_survives_the_microohm_cm_ladder_rung_on_real_data():
    # The real file's rho lands in the 1e6 / "µΩ·cm" rung -- the exact case the trap needs.
    from cryosweep_core.plotting.render import _RHO_UNIT_LADDER
    assert _RHO_UNIT_LADDER[0][1] == 1e6
    fig = render_kind(_run(FX / "tto_real_subset.dat"), "tto_summary_t",
                      PlotSpec(error_band=True), GlobalStyle())
    ax = _rho_panel(fig)
    assert "µΩ·cm" in ax.get_ylabel()
    band = [c for c in ax.collections if c.get_gid() == "errband"][0]
    by = np.concatenate([p.vertices[:, 1] for p in band.get_paths()])
    by = by[np.isfinite(by)]
    ly = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in ax.lines
                         if ln.get_gid() is None])
    ly = ly[np.isfinite(ly)]
    assert by.min() < ly.min() and by.max() > ly.max()
    # the whole band is inside the view (I5): a line-derived y-view would crop it at T_min,
    # where rho carries 8.15 % sigma on the real file.
    bottom, top = ax.get_ylim()
    assert bottom <= by.min() and top >= by.max()


def test_rho_band_survives_a_biting_robust_view():
    # THE real-render pin for _tto_expand_ylim_for_bands. At the default robust_k=8 the helper
    # is inert on today's fixtures (autoscale already covers the ribbon), so every other test
    # here survives stubbing it. `robust_k` is a user-settable GlobalStyle knob exposed in the
    # GUI: at 1.5 `_apply_robust_view` -- which runs INSIDE _finish, AFTER the band is drawn,
    # and reads ax.lines ONLY -- re-clips the rho panel to a line-derived envelope. Measured
    # with the helper stubbed: ylim (246.602, 365.656) against a band spanning
    # (220.514, 374.039), i.e. clipped at BOTH ends, including the low-T point where rho
    # carries its largest uncertainty (I5).
    fig = render_kind(_run(FX / "tto_real_subset.dat"), "tto_summary_t",
                      PlotSpec(error_band=True), GlobalStyle(robust_k=1.5))
    ax = _rho_panel(fig)
    band = [c for c in ax.collections if c.get_gid() == "errband"][0]
    by = np.concatenate([p.vertices[:, 1] for p in band.get_paths()])
    by = by[np.isfinite(by)]
    bottom, top = ax.get_ylim()
    assert bottom <= by.min() and top >= by.max()


def test_band_extent_merges_across_repeated_draws_on_one_axes():
    # `_tto_draw_bands` is called at most once per axes TODAY, so the `prev is not None` merge
    # branch is unreachable from every render path and overwriting `ax._tto_band_extent`
    # survives the whole suite. Pin the branch directly, in plot space: two draws with disjoint
    # extents must leave the UNION recorded, not the last one.
    from cryosweep_core.plotting.catalog import Series
    from cryosweep_core.plotting.render import _tto_draw_bands
    from matplotlib.figure import Figure
    kind = {k.key: k for k in BUILTIN_PLOTKINDS}["tto_kappa_t"]
    lo_s = Series(key="kappa:0:down", label="cooling", x=[1.0, 2.0], y=[0.0, 0.0],
                  group="cooling", yerr=[1.0, 1.0])          # extent (-1, 1)
    hi_s = Series(key="kappa:1:up", label="warming", x=[1.0, 2.0], y=[10.0, 10.0],
                  group="warming", yerr=[1.0, 1.0])          # extent (9, 11)
    ax = Figure().add_subplot(111)
    spec, style = PlotSpec(error_band=True), GlobalStyle()
    assert _tto_draw_bands(ax, [(None, lo_s)], kind, spec, style, {"cooling": "C0"}) \
        == (pytest.approx(-1.0), pytest.approx(1.0))
    assert _tto_draw_bands(ax, [(None, hi_s)], kind, spec, style, {"warming": "C1"}) \
        == (pytest.approx(9.0), pytest.approx(11.0))         # the RETURN is this call's own...
    # ...but the axes-level record, which _tto_expand_ylim_for_bands reads, is the union.
    assert ax._tto_band_extent == (pytest.approx(-1.0), pytest.approx(11.0))


def test_rho_axis_autoscale_is_not_taught_to_walk_collections():
    # The alternative fix is a SECOND way to get the same number wrong; pinned at source level
    # because the outputs of the two designs agree on every file that does not break.
    import inspect
    import cryosweep_core.plotting.render as R
    src = inspect.getsource(R._rho_axis_autoscale)
    # M5: match the ATTRIBUTE ACCESS, not the bare word — a future comment mentioning
    # collections (this plan's own C2 rationale is full of the word) must not fail the pin.
    assert ".collections" not in src
    assert "fill_between" not in src
    assert "for ln in ax.lines" in src


def test_draw_bands_scales_sigma_by_yscale():
    # Task 5 left `e = np.asarray(s.yerr, float) * yscale` UNTESTED: the summary rho panel was
    # the only yscale != 1 caller and its band was still deferred. Task 6 turns it on, so pin
    # it directly -- dropping the `* yscale` leaves the band at 1/yscale of its true width
    # while the LINE (y * yscale) still lands correctly, i.e. exactly the 1e6 collapse.
    from matplotlib.figure import Figure
    from cryosweep_core.plotting.catalog import Series
    from cryosweep_core.plotting.render import _tto_draw_bands
    kind = {k.key: k for k in BUILTIN_PLOTKINDS}["tto_summary_t"]
    s = Series(key="rho:0:down", label="cooling", x=[1.0, 2.0, 3.0],
               y=[1.0, 2.0, 3.0], group="cooling", yerr=[0.1, 0.2, 0.3])
    ax = Figure().add_subplot(111)
    ext = _tto_draw_bands(ax, [(None, s)], kind, PlotSpec(error_band=True), GlobalStyle(),
                          {"cooling": "C0"}, yscale=1e8)
    band = [c for c in ax.collections if c.get_gid() == "errband"][0]
    assert [h for _, h in _half_widths(band)] == [pytest.approx(0.1e8), pytest.approx(0.2e8),
                                                  pytest.approx(0.3e8)]
    assert ext == (pytest.approx(0.9e8), pytest.approx(3.3e8))


# ---- I1 (final review): the low-T inset is the HIGH-sigma region ---------------------------

def _inset_bands(fig):
    return [c for c in fig.axes[1].collections if c.get_gid() == "errband"]


def test_the_lowt_inset_draws_its_own_band_because_that_is_the_high_sigma_region():
    """The inset magnifies [0, 30] K -- exactly where kappa's relative sigma is LARGEST
    (measured on the gate file: max 8.60 % at 4.033 K, median 1.20 %, against ~0.1 % at room
    temperature). Before this fix the main axes showed a band and the inset beside it showed
    the same data as exact, which is the "only one of these curves has an error estimate"
    defect the _TTO_BAND_KINDS allow-list exists to prevent, applied within one figure."""
    real = real_data("tto")
    r = _run(real) if real is not None else _run(FX / "tto_real_subset.dat")
    fig = render_kind(r, "tto_kappa_t", PlotSpec(error_band=True), GlobalStyle())
    assert len(fig.axes) == 2                       # host + inset
    assert len(_inset_bands(fig)) == 1
    # the band is REAL, not a zero-width polygon: its half-width must be a visible fraction
    band = _inset_bands(fig)[0]
    ys = band.get_paths()[0].vertices[:, 1]
    assert float(ys.max() - ys.min()) > 0.0
    # and it shares the pinned artist contract with every other band
    assert band.get_alpha() == pytest.approx(0.20) and band.get_zorder() == pytest.approx(1.5)


def test_the_inset_band_is_off_when_the_toggle_is_off():
    # the byte-identity half: nothing at all is added on the default spec.
    fig = render_kind(_run(FX / "tto_real_subset.dat"), "tto_kappa_t",
                      PlotSpec(), GlobalStyle())
    assert len(fig.axes) == 2 and _inset_bands(fig) == []


def test_the_inset_band_is_skipped_when_kappa_std_is_missing_or_mismatched():
    """Structural safety, same shape as the series builder's length guard: a `kappa_std`
    shorter than `t` would raise a bare numpy broadcast error inside the inset."""
    r = _run(FX / "tto_real_subset.dat")
    curve = max(r.data["curves"], key=lambda c: len(c["t"]))
    curve["kappa_std"] = curve["kappa_std"][:-3]
    fig = render_kind(r, "tto_kappa_t", PlotSpec(error_band=True), GlobalStyle())
    assert _inset_bands(fig) == []                  # skipped, not crashed on
    curve["kappa_std"] = None
    fig = render_kind(r, "tto_kappa_t", PlotSpec(error_band=True), GlobalStyle())
    assert _inset_bands(fig) == []


# ---- m1 (final review): the series builder's length guard ---------------------------------

def test_a_short_std_array_yields_no_yerr_and_still_renders():
    """m1. Mutation `if std is not None and len(std) == len(t):` -> `if std is not None:`
    SURVIVED. Without the guard a mismatched `*_std` reaches `_tto_draw_bands`, where `y - e`
    raises a bare numpy broadcast ValueError inside `render_kind` — a crash, not a bad plot."""
    import types
    from cryosweep_core.plotting.catalog import _tto_curve_series
    r = types.SimpleNamespace(data={"curves": [dict(
        t=[1.0, 2.0, 3.0], kappa=[1.0, 2.0, 3.0], kappa_std=[0.1, 0.2],
        field_oe=0.0, direction="down")]})
    s = _tto_curve_series(r, "kappa", "kappa")[0]
    assert s.yerr is None
    # and the render survives with the band asked for
    r2 = _run(FX / "tto_real_subset.dat")
    curve = max(r2.data["curves"], key=lambda c: len(c["t"]))
    curve["kappa_std"] = curve["kappa_std"][:5]
    fig = render_kind(r2, "tto_kappa_t", PlotSpec(error_band=True), GlobalStyle())
    assert _bands(fig) == []                      # nothing drawn, nothing raised


# ---- I3 (final review): the band must be VISIBLE, and the lever is the marker footprint ----

def _marker_sizes(fig):
    return [ln.get_markersize() for ln in fig.axes[0].lines if ln.get_gid() is None]


@pytest.mark.parametrize("kind", sorted(_TTO_BAND_KINDS))
def test_markers_shrink_while_a_band_is_drawn_so_the_ribbon_shows_through(kind):
    """Measured on the gate file by rendering each kind twice at IDENTICAL y-limits and
    diffing pixel-by-pixel: turning the band on changed 14 px on tto_seebeck_t (0.014 % of the
    figure), 297 on tto_zt_t, 747 on tto_kappa_t, 1255 on tto_summary_t. The cause is
    FOOTPRINT -- 976 filled markers at the default size cover the ribbon wherever it is narrow,
    in the same colour. Shrinking them to 0.55x lifts the band-only pixel delta to 88 / 578 /
    1060 / 1763 (band isolated by stubbing `_tto_draw_bands`, so the marker change itself is
    not counted).

    NOT alpha and NOT zorder: those are a pinned artist contract, and raising either paints the
    ribbon over the data it qualifies."""
    r = _run(FX / "tto_real_subset.dat")
    off = render_kind(r, kind, PlotSpec(), GlobalStyle())
    on = render_kind(r, kind, PlotSpec(error_band=True), GlobalStyle())
    assert _marker_sizes(off) and len(_marker_sizes(off)) == len(_marker_sizes(on))
    for a, b in zip(_marker_sizes(off), _marker_sizes(on)):
        assert b == pytest.approx(a * 0.55) and b < a


@pytest.mark.parametrize("kind", ["tto_wf_t", "tto_lorenz_t"])
def test_a_non_band_kind_keeps_its_marker_size_even_with_the_toggle_on(kind):
    # The shrink is tied to a band actually being drawn. On the excluded kinds error_band=True
    # draws nothing, so shrinking there would be a gratuitous style change.
    r = _run(FX / "tto_real_subset.dat")
    off = render_kind(r, kind, PlotSpec(), GlobalStyle())
    on = render_kind(r, kind, PlotSpec(error_band=True), GlobalStyle())
    assert _marker_sizes(on) == _marker_sizes(off) and _marker_sizes(off)


def test_the_inset_markers_shrink_with_the_band_too():
    # Looked at: at 976 points the unshrunk inset is a solid black worm that buries the ribbon.
    r = _run(FX / "tto_real_subset.dat")
    off = render_kind(r, "tto_kappa_t", PlotSpec(), GlobalStyle())
    on = render_kind(r, "tto_kappa_t", PlotSpec(error_band=True), GlobalStyle())
    a = off.axes[1].lines[0].get_markersize()
    b = on.axes[1].lines[0].get_markersize()
    assert b < a and b == pytest.approx(max(a * 0.55, 1.0))


def test_the_band_expansion_on_the_rho_panel_is_the_BAND_not_dead_space():
    """The review called the rho panel's expansion "~40 % dead space". Re-measured: rho's
    relative sigma reaches 20.5 % at T = 2.457 K (rho 260.29 +- 53.42 uOhm.cm), so the band
    really does reach 206.87 while the line stops at 251.90. Of the 47.5 units added below,
    ~39 are band and the rest is the standard 5 % pad -- so the expansion is NOT skipped and
    NOT thresholded: clipping it would hide a genuine +-20 % error bar at the lowest T, which
    is the single thing this slice exists to prevent."""
    real = real_data("tto")
    r = _run(real) if real is not None else None
    if r is None:
        pytest.skip("real Thermal Transport file absent")
    fig = render_kind(r, "tto_summary_t", PlotSpec(error_band=True), GlobalStyle())
    ax = fig.axes[2]
    lo, hi = getattr(ax, "_tto_band_extent")
    assert lo == pytest.approx(206.8741656111626)
    assert hi == pytest.approx(376.3687101756302)
    assert ax.get_ylim()[0] <= lo and ax.get_ylim()[1] >= hi
