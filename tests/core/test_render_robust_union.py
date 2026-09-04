"""Robust-view UNION discriminator regression tests (PQ-3 fix).

Pins the two non-negotiable real-data acceptance behaviors plus their synthetic
mirrors (so CI without the gitignored real files still guards the rule):

  (1) A legitimate multi-magnitude same-quantity family (MPMS M(T): 500 Oe vs
      40000 Oe, ratio ~80x) keeps EVERY series' bulk in view.
  (2) A single garbage line whose scale sits orders beyond the family (a non-Hall
      channel segment folded into hall_asym_vs_B) is EXCLUDED from the union so the
      real antisym signal is not flattened to invisibility.

Plus: fit/refline lines never steer the view; single-series is a union-of-one no-op.
"""
import dataclasses
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from cryosweep_core.plotting.render import _apply_robust_view, render_kind
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.robust import robust_range

from tests.core.conftest import FIX, real_data

VSM_N_REAL = real_data("vsm")          # None when the local-only file is unavailable
RES_DAT = real_data("res")
MPMS_REAL = real_data("mpms")

STYLE = GlobalStyle()


# ---------------------------------------------------------------- helpers
def _axes(series, gids=None):
    fig, ax = plt.subplots()
    for i, y in enumerate(series):
        y = np.asarray(y, float)
        ln, = ax.plot(np.arange(y.size), y)
        if gids is not None and gids[i] is not None:
            ln.set_gid(gids[i])
    return fig, ax


def _tailed(bulk_lo, bulk_hi, n=200, tail=None, ntail=7):
    """A clean ramp bulk with a short heavy tail so robust-view has a tail to narrow."""
    y = np.linspace(bulk_lo, bulk_hi, n)
    if tail is not None:
        y[-ntail:] = tail
    return y


def _scale(y):
    lo, hi = robust_range(np.asarray(y, float), k=STYLE.robust_k)
    return abs((lo + hi) / 2) + (hi - lo) / 2


# ---------------------------------------------------------------- synthetic rule guards
def test_multi_magnitude_family_all_bulk_visible():
    # Two small series (~1e-3) + one legit big series (~0.08); scale ratio ~80x < K=100.
    # The big series' bulk must stay in view (NOT excluded as garbage).
    small1 = _tailed(6e-4, 3e-3, tail=8e-3)
    small2 = _tailed(6e-4, 3.3e-3, tail=9e-3)
    big = _tailed(5e-2, 0.10, tail=0.30)
    fig, ax = _axes([small1, small2, big])
    _apply_robust_view(ax, PlotSpec(), STYLE)
    top = ax.get_ylim()[1]
    big_hi = robust_range(big, k=STYLE.robust_k)[1]
    assert top >= big_hi, f"big series clipped: top={top} < robust_hi={big_hi}"
    plt.close(fig)


def test_garbage_line_excluded_from_union():
    # Five signal lines (bulk ~1e-5) + one garbage line ~5e-2 (scale ~hundreds x the
    # median line scale, fully disjoint from the pooled bulk). Garbage must be excluded
    # so the view stays at the signal scale, not blown up ~thousandfold.
    signal = [_tailed(0.0, 2e-5, tail=8e-4) for _ in range(5)]
    garbage = _tailed(4.8e-2, 5.4e-2, tail=6e-2)
    fig, ax = _axes(signal + [garbage])
    _apply_robust_view(ax, PlotSpec(), STYLE)
    top = ax.get_ylim()[1]
    med_sig = float(np.median([_scale(y) for y in signal]))
    assert top < 100 * med_sig, f"garbage steered the view: top={top}, med_scale={med_sig}"
    # sanity: the garbage really is oversized+disjoint (would dominate a naive union)
    assert _scale(garbage) > 100 * med_sig
    plt.close(fig)


def test_fit_line_does_not_steer_view():
    # One data line (bulk ~1e-5, small tail) + one fit line reaching ~1e-2. The fit line
    # is already envelope-clipped elsewhere; it must not push the view up.
    data = _tailed(0.0, 2e-5, tail=6e-4)
    fit = np.linspace(0.0, 1e-2, 200)
    fig, ax = _axes([data, fit], gids=[None, "fit"])
    _apply_robust_view(ax, PlotSpec(), STYLE)
    top = ax.get_ylim()[1]
    data_hi = robust_range(data, k=STYLE.robust_k)[1]
    assert top < 5 * data_hi, f"fit line steered view: top={top}, data_hi={data_hi}"
    plt.close(fig)


def test_zero_centered_garbage_excluded():
    # THE ch1 failure mode: garbage noise SYMMETRIC ABOUT ZERO. Its robust range overlaps
    # any zero-centered pooled bulk, so a disjointness prong can never fire on it; the
    # discriminator must work on SPAN RATIO alone. 5 signal lines (span ~1e-5) + 1
    # zero-centered noise line (span ~1e-1, ratio ~thousands x the median span).
    rng = np.random.default_rng(42)
    signal = [_tailed(-1e-5, 1e-5, tail=4e-4) for _ in range(5)]
    garbage = rng.uniform(-5e-2, 5e-2, 200)          # symmetric about zero, huge span
    fig, ax = _axes(signal + [garbage])
    _apply_robust_view(ax, PlotSpec(), STYLE)
    lo_v, hi_v = ax.get_ylim()
    med_sig_hi = float(np.median([robust_range(y, k=STYLE.robust_k)[1] for y in signal]))
    assert hi_v < 100 * med_sig_hi, f"zero-centered garbage steered view: top={hi_v}"
    assert lo_v > -100 * med_sig_hi
    plt.close(fig)


def test_span_ladder_family_kept_despite_1000x_span_ratio():
    # VSM_N shape: a LEGIT family whose flat low-field curves (tiny span) and swinging
    # high-field curves (span ~1000x the median) form a quasi-continuous span LADDER
    # (each consecutive sorted span within ~2 decades of the next). A span-vs-median-only
    # rule would exclude the big legit series; the discriminator must cut only at an
    # ISOLATED >=K consecutive gap, which a ladder never has.
    flat1 = _tailed(4.5e-4, 6.7e-4)                      # span ~2e-4
    flat2 = _tailed(4.4e-4, 7.8e-4)                      # span ~3e-4
    mid = _tailed(5e-3, 5.8e-2)                          # span ~5e-2 (~60x gap, < K)
    big = _tailed(4e-2, 4.8e-1)                          # span ~4e-1 (~8x gap)
    biggest = _tailed(1e-1, 9.9e-1, tail=30.0)           # span ~9e-1 (~2x gap), heavy tail so view narrows
    fig, ax = _axes([flat1, flat2, mid, big, biggest])
    _apply_robust_view(ax, PlotSpec(), STYLE)
    top = ax.get_ylim()[1]
    biggest_hi = robust_range(biggest, k=STYLE.robust_k)[1]
    assert top >= biggest_hi, f"legit ladder member clipped: top={top} < {biggest_hi}"
    plt.close(fig)


def test_refline_still_excluded():
    data = _tailed(0.0, 2e-5, tail=6e-4)
    ref = np.full(200, 1e-2)
    fig, ax = _axes([data, ref], gids=[None, "refline"])
    _apply_robust_view(ax, PlotSpec(), STYLE)
    top = ax.get_ylim()[1]
    data_hi = robust_range(data, k=STYLE.robust_k)[1]
    assert top < 5 * data_hi
    plt.close(fig)


def test_single_series_is_union_of_one():
    # Union-of-one must reproduce the pooled single-line robust narrowing exactly.
    y = _tailed(3e-5, 5e-5, tail=1.46e-2)
    fig, ax = _axes([y])
    _apply_robust_view(ax, PlotSpec(), STYLE)
    lo_v, hi_v = ax.get_ylim()
    r_lo, r_hi = robust_range(y, k=STYLE.robust_k)
    span = r_hi - r_lo
    assert hi_v == pytest.approx(r_hi + 0.05 * span)
    assert lo_v == pytest.approx(r_lo - 0.05 * span)
    plt.close(fig)


# ---------------------------------------------------------------- real-file acceptance
def _hall_result(path, hall_cfg=None):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    rt = load_dat(str(path))
    cfg = RunConfig.load(hall=hall_cfg or {"hall_channel": 2, "longitudinal_channel": 1,
                                           "thickness_mm": 0.2}, probe_override="hall")
    return analyze_file(rt, cfg, build_default_registry())


# The exact pq_compare manifest runconfig for the hall_* gallery entries
# (docs/superpowers/pq-reference-gallery/manifest.json): Hall on channel 1.
_MANIFEST_HALL = {"hall_channel": 1, "thickness_mm": 0.5, "longitudinal_channel": 2}


def _pooled_base_hi(ax):
    """The pre-union pooled robust hi (non-refline, non-fit) = the base target."""
    ys = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in ax.lines
                         if ln.get_gid() not in ("refline", "fit")])
    ys = ys[np.isfinite(ys)]
    return robust_range(ys, k=GlobalStyle(width_mm=160, height_mm=120, dpi=110).robust_k)[1]


def test_hall_asym_within_2x_of_base_pooled():
    if RES_DAT is None:
        pytest.skip("local-only measurement file for key 'res' is not available")
    st = GlobalStyle(width_mm=160, height_mm=120, dpi=110)
    res = _hall_result(RES_DAT)
    fig = render_kind(res, "hall_asym_vs_B", PlotSpec(), st)
    ax = fig.axes[0]
    top = ax.get_ylim()[1]
    base_hi = _pooled_base_hi(ax)
    assert top <= 2.0 * base_hi, f"asym top={top} exceeds 2x base pooled hi={base_hi}"
    plt.close(fig)


def test_hall_rxy_within_2x_of_base_pooled():
    if RES_DAT is None:
        pytest.skip("local-only measurement file for key 'res' is not available")
    st = GlobalStyle(width_mm=160, height_mm=120, dpi=110)
    res = _hall_result(RES_DAT)
    fig = render_kind(res, "hall_rxy_vs_B", PlotSpec(), st)
    ax = fig.axes[0]
    top = ax.get_ylim()[1]
    base_hi = _pooled_base_hi(ax)
    assert top <= 2.0 * base_hi, f"rxy top={top} exceeds 2x base pooled hi={base_hi}"
    plt.close(fig)


def test_hall_asym_manifest_ch1_within_2x_of_base_pooled():
    if RES_DAT is None:
        pytest.skip("local-only measurement file for key 'res' is not available")
    # THE visual-gate config (manifest hall_channel=1). Here the antisym 300 K line IS the
    # garbage (non-Hall channel wiring): robust range (-4.9e-2, +4.9e-2), symmetric about
    # zero (never disjoint from a zero-centered bulk), span ~8000x the median line span.
    # The span-ratio discriminator must exclude it; view returns to the ~1e-5 signal scale
    # (base pooled robust hi ~1.16e-5).
    st = GlobalStyle(width_mm=160, height_mm=120, dpi=110)
    res = _hall_result(RES_DAT, _MANIFEST_HALL)
    fig = render_kind(res, "hall_asym_vs_B", PlotSpec(), st)
    ax = fig.axes[0]
    lo_v, hi_v = ax.get_ylim()
    ys = np.concatenate([np.asarray(ln.get_ydata(), float) for ln in ax.lines
                         if ln.get_gid() not in ("refline", "fit")])
    ys = ys[np.isfinite(ys)]
    base_lo, base_hi = robust_range(ys, k=st.robust_k)
    assert hi_v <= 2.0 * base_hi, f"ch1 asym top={hi_v} exceeds 2x base pooled hi={base_hi}"
    assert lo_v >= 2.0 * base_lo, f"ch1 asym bottom={lo_v} below 2x base pooled lo={base_lo}"
    plt.close(fig)


def test_hall_rxy_manifest_ch1_pinned():
    if RES_DAT is None:
        pytest.skip("local-only measurement file for key 'res' is not available")
    # Same manifest config, raw R_xy view. No garbage exclusion fires here (max/median
    # span ratio ~7x, a legit family), so the per-line union hi tracks the 300 K series'
    # robust hi ~1.34e-3 — a plausible-sanctioned widening over the base pooled ~8.9e-4
    # (the union keeps every legit series' bulk visible by design). Pin the value so any
    # future rule change that moves this view is a deliberate decision.
    st = GlobalStyle(width_mm=160, height_mm=120, dpi=110)
    res = _hall_result(RES_DAT, _MANIFEST_HALL)
    fig = render_kind(res, "hall_rxy_vs_B", PlotSpec(), st)
    lo_v, hi_v = fig.axes[0].get_ylim()
    assert hi_v == pytest.approx(1.4016761434074904e-03, rel=1e-6)
    assert lo_v == pytest.approx(-3.078661138251217e-05, rel=1e-6)
    plt.close(fig)


def test_mpms_moment_t_all_series_bulk_visible():
    if MPMS_REAL is None:
        pytest.skip("local-only measurement file for key 'mpms' is not available")
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    st = GlobalStyle(width_mm=160, height_mm=120, dpi=110)
    rt = load_dat(str(MPMS_REAL))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header,
                                                            molar_mass=683.22, mass_mg=12.0))
    res = analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())
    fig = render_kind(res, "vsm_moment_t", PlotSpec(), st)
    ax = fig.axes[0]
    lo_v, hi_v = ax.get_ylim()
    # every legend series' bulk must be inside the view (the 40000 Oe series is ~80x the
    # 500 Oe series; a pooled robust view would clip it to the 500 Oe scale ~0.016).
    per = [np.asarray(ln.get_ydata(), float) for ln in ax.lines
           if ln.get_gid() not in ("refline", "fit")]
    big_max = max(float(a[np.isfinite(a)].max()) for a in per)
    small_min = min(float(a[np.isfinite(a)].min()) for a in per)
    assert hi_v >= big_max, f"40000 Oe bulk clipped: top={hi_v} < {big_max}"
    assert lo_v <= small_min
    # prove the pooled-clip regression did NOT happen: view is far above the pooled hi
    pooled_hi = _pooled_base_hi(ax)
    assert hi_v > 5 * pooled_hi
    plt.close(fig)


def test_vsm_n_moment_t_all_series_bulk_visible():
    if VSM_N_REAL is None:
        pytest.skip("local-only measurement file for key 'vsm' is not available")
    # The hardest legit family (visual-gate render): 100/5000/40000/100000 Oe M(T) —
    # flat low-field curves (robust span ~2e-4) vs a swinging 100000 Oe curve (span
    # ~0.9, ratio ~1000x the median span), but a quasi-continuous span ladder (max
    # consecutive sorted-span gap ~60x). Every series' bulk must stay in view: a
    # span-vs-median-only garbage rule would clip the top to ~0.1 and hide the two
    # biggest legit series.
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    st = GlobalStyle(width_mm=160, height_mm=120, dpi=110)
    rt = load_dat(str(VSM_N_REAL))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header,
                                                            molar_mass=300.0, mass_mg=1.1))
    res = analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())
    fig = render_kind(res, "vsm_moment_t", PlotSpec(), st)
    ax = fig.axes[0]
    hi_v = ax.get_ylim()[1]
    his = []
    for ln in ax.lines:
        if ln.get_gid() in ("refline", "fit"):
            continue
        a = np.asarray(ln.get_ydata(), float)
        a = a[np.isfinite(a)]
        if a.size:
            # a series' visible bulk tops out at its data max even when the robust
            # envelope overshoots it — require the view to cover min(robust_hi, dmax)
            his.append(min(robust_range(a, k=st.robust_k)[1], float(a.max())))
    assert hi_v >= max(his), f"biggest legit series clipped: top={hi_v} < {max(his)}"
    plt.close(fig)
