"""RRR + classification + summary + capabilities (spec §2 steps 6-7, D10)."""
import pathlib

import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat

FX = pathlib.Path("tests/core/fixtures")


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def _caps(d):
    return {c["name"]: c for c in d["capabilities"]}


def test_synth_rrr_matches_the_hand_derived_fraction():
    # 150-point grid, exact 2 K spacing: 5 points nearest T_max are 300/298/296/294/292 K
    # -> median rho = rho(296) = 9.88e-8; 5 nearest T_min are 2/4/6/8/10 K -> rho(6) = 1.18e-8.
    # RRR = 9.88/1.18 = 494/59. Written as the fraction: the spelled decimal is 1 ULP off.
    d = _run(FX / "tto_synth.dat").data
    assert d["rrr"]["rrr"] == pytest.approx(494 / 59)
    assert d["rrr"]["t_high_k"] == pytest.approx(296.0)
    assert d["rrr"]["t_low_k"] == pytest.approx(6.0)
    assert d["rrr"]["classification"] == "metallic"


def test_high_field_group_is_excluded_from_rrr_selection():
    # The 90000 Oe group must not be the RRR-selection curve (|H| < 50 Oe rule). Asserted as
    # an IDENTITY (the selection IS the 0 Oe curve) rather than as a restatement of the filter.
    d = _run(FX / "tto_synth.dat").data
    from cryosweep_core.analyzers.tto import _rrr_curve, TTOCurve
    curves = [TTOCurve(**c) for c in d["curves"]]
    high = [c for c in curves if c.field_oe == pytest.approx(90000.0)]
    assert high, "fixture must contain the 90000 Oe group for this test to mean anything"
    sel = _rrr_curve(curves)
    assert sel.field_oe == pytest.approx(0.0)
    assert all(sel is not c for c in high)


def test_rrr_curve_picks_the_widest_zero_field_span_first_on_tie():
    # Pins BOTH halves of the selection rule that Tasks 6-10 reuse: (a) widest T span wins,
    # (b) ties resolve to the FIRST curve in list order, and (c) |H| >= 50 Oe is out of the
    # running no matter how wide it is.
    from cryosweep_core.analyzers.tto import _rrr_curve, TTOCurve

    def curve(field_oe, t_span, tag):
        t = list(np.linspace(0.0, t_span, 20))
        return TTOCurve(field_oe=field_oe, direction=tag, n_points=20, t=t,
                        kappa=[1.0] * 20)

    narrow = curve(0.0, 50.0, "narrow")
    wide_first = curve(0.0, 200.0, "wide_first")
    wide_second = curve(0.077, 200.0, "wide_second")
    widest_but_fielded = curve(60.0, 1000.0, "fielded")
    sel = _rrr_curve([narrow, wide_first, wide_second, widest_but_fielded])
    assert sel is wide_first
    assert sel is not widest_but_fielded


def test_classify_reports_non_monotonic_and_unknown():
    # The 1.02 / 0.98 flat band and the no-valid-rho fallback, pinned directly.
    from cryosweep_core.analyzers.tto import _classify
    T = np.linspace(2.0, 300.0, 40)
    lo = 1e-8
    assert _classify(T, np.linspace(lo, 1.01 * lo, 40)) == "non_monotonic"
    assert _classify(T, np.linspace(lo, 0.99 * lo, 40)) == "non_monotonic"
    # ...and from below: a ratio of 0.5 is well clear of the band and IS insulating (the
    # fixture's ratio of ~0.02 would pass under any threshold above ~0.021).
    assert _classify(T, np.linspace(lo, 0.5 * lo, 40)) == "insulating"
    assert _classify(T, np.full(40, np.nan)) == "unknown"


def test_rrr_declines_a_quotient_that_overflows_to_infinity():
    # Never emit a non-finite scalar: a subnormal low-T rho overflows rho_hi/rho_lo to inf,
    # which json.dumps(allow_nan=False) would reject downstream.
    from cryosweep_core.analyzers.tto import _rrr
    T = np.arange(1.0, 11.0)
    rho = np.array([5e-324] * 5 + [1e300] * 5)
    assert np.isinf(rho[-1] / rho[0])          # the overflow really happens
    assert _rrr(T, rho) == (None, None, None)


def test_endpoint_selection_is_stable_under_duplicate_temperatures():
    # 20 duplicate T values on each side: an unstable argsort picks a different 5-point set
    # (implementation-defined), a stable one always takes the first five in row order.
    from cryosweep_core.analyzers.tto import _endpoint
    T = np.array([5.0] * 20 + [9.0] * 20)
    rho = np.arange(1, 41, dtype=float) * 1e-8
    assert _endpoint(T, rho, True) == (5.0, pytest.approx(float(np.median(rho[:5]))))
    assert _endpoint(T, rho, False) == (9.0, pytest.approx(float(np.median(rho[-5:]))))


def test_pf_at_thigh_follows_the_rrr_selection_curve_even_when_rrr_is_none():
    # The PF curve is the RRR-SELECTION curve, not curves[0], even when RRR itself declines.
    # Contrived: the widest zero-field curve (40 Oe) has no valid rho -> rrr None and PF None,
    # while the narrower 0 Oe curve (which is curves[0], groups sort by field) has good rho.
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer, TTOCurve, _pf_at_thigh
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    zf = df.index[df["Magnetic Field (Oe)"].astype(float) == 0.0]
    wide, narrow = zf[:120], zf[120:]          # wide = T 300..62 K, narrow = T 60..2 K
    df.loc[wide, "Magnetic Field (Oe)"] = 40.0
    df.loc[wide, "Resistivity (Ohm-m)"] = np.nan
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    curves = r.data["curves"]
    assert curves[0]["field_oe"] == pytest.approx(0.0)
    assert len(curves[0]["t"]) == len(narrow)
    assert r.data["rrr"] is None
    assert r.data["summary"]["pf_at_thigh"] is None      # curves[0] would give a real number
    assert _pf_at_thigh(TTOCurve(**curves[0])) is not None


def test_rrr_is_none_when_no_zero_field_curve_exists():
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    df["Magnetic Field (Oe)"] = 90000.0
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    assert r.data["rrr"] is None
    assert _caps(r.data)["rrr"]["applicable"] is False
    assert _caps(r.data)["rrr"]["reason"] == "no zero-field ρ(T) ramp"


def test_classification_flips_to_insulating_when_rho_falls_with_temperature():
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    T = df["Sample Temp. (K)"].astype(float)
    df["Resistivity (Ohm-m)"] = 1e-6 / T          # falls with T
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    assert r.data["rrr"]["classification"] == "insulating"
    assert r.data["rrr"]["rrr"] < 1.0


def test_classification_thresholds_sit_at_1_02_and_0_98():
    # The 1.02/0.98 band was unconstrained: only a 1.45 metallic and a strongly insulating
    # sample exist in the suite, so widening it to 1.40/0.60 survived — and a real ratio-1.1
    # sample would then be labelled "non_monotonic" (i.e. flat/ambiguous) instead of metallic.
    from cryosweep_core.analyzers.tto import _classify
    T = np.linspace(2.0, 300.0, 40)

    def cls(ratio):
        # rho rising (or falling) linearly from 1e-8 to ratio*1e-8 across the sweep
        return _classify(T, 1e-8 * (1.0 + (ratio - 1.0) * (T - T.min()) / np.ptp(T)))

    assert cls(1.10) == "metallic"          # just above the band, not "ambiguous"
    assert cls(1.03) == "metallic"
    assert cls(1.01) == "non_monotonic"     # inside the band -> flat/ambiguous
    assert cls(0.99) == "non_monotonic"
    assert cls(0.97) == "insulating"
    assert cls(0.90) == "insulating"


def test_two_field_setpoints_two_percent_apart_stay_separate_curves():
    # _FIELD_REL_TOL = 1%. The fixtures sit at 0 and 90 kOe — 50x too far apart to constrain
    # it — so widening the tolerance to 0.5 survived, silently merging two nearby real
    # setpoints into one curve.
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    zf = df.index[df["Magnetic Field (Oe)"].astype(float) == 0.0]
    df.loc[zf[:75], "Magnetic Field (Oe)"] = 10000.0
    df.loc[zf[75:], "Magnetic Field (Oe)"] = 10200.0        # 2% away: a DIFFERENT setpoint
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    fields = sorted(c["field_oe"] for c in r.data["curves"])
    assert fields == [pytest.approx(10000.0), pytest.approx(10200.0),
                      pytest.approx(90000.0)]


def test_a_capability_needs_five_valid_points_not_one():
    # `_any_curve_has(..., n=5)` -> n=1 survived: a single stray finite point would advertise
    # a capability (and a plot kind) the file cannot support.
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))

    def caps_with(n_valid):
        df = rt.df.copy()
        col = "Seebeck Coef. (µV/K)"
        keep = df.index[:n_valid]
        df[col] = np.nan
        df.loc[keep, col] = 1.0
        return _caps(TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig()).data)

    assert caps_with(4)["seebeck"]["applicable"] is False
    assert caps_with(5)["seebeck"]["applicable"] is True


def test_an_absent_field_column_is_assumed_to_be_zero_field():
    # The `np.zeros` fallback was unpinned: filling 1000.0 instead survived, which would have
    # silently pushed every curve out of the |H| < 50 Oe RRR selection.
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.drop(columns=["Magnetic Field (Oe)"])
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    assert r.status == "ok"
    assert [c["field_oe"] for c in r.data["curves"]] == [0.0]
    assert r.data["rrr"] is not None                    # still counts as zero field


def test_cross_probe_rrr_equality_on_a_sentinel_free_array():
    # D10: the TTO _endpoint drops resistivity.py's sentinel guard + MAD exclusion. On a
    # sentinel-free array both implementations must agree exactly.
    from cryosweep_core.analyzers.tto import _rrr as tto_rrr
    from cryosweep_core.analyzers.resistivity import _rrr as res_rrr
    T = np.linspace(2.0, 300.0, 120)
    rho = 1e-8 * (1.0 + 9.0 * T / 300.0)
    a = tto_rrr(T, rho)
    b = res_rrr(T, rho, cfg=None)
    assert a[0] == b[0] and a[1] == b[1] and a[2] == b[2]


def test_summary_power_factor_at_high_temperature():
    # Median PF over the 5 points nearest T_max on the RRR-selection curve. On the synth grid
    # those are 292/294/296/298/300 K; PF is monotone in T so the median is PF(296 K):
    #   S = 2.96 uV/K, rho = 1e-8*(1+9*296/300) = 9.88e-8 -> PF = (2.96e-6)^2/9.88e-8.
    d = _run(FX / "tto_synth.dat").data
    expected = (2.96e-6) ** 2 / (1e-8 * (1 + 9 * 296.0 / 300.0))
    assert d["summary"]["pf_at_thigh"] == pytest.approx(expected, rel=1e-6)


def test_summary_zt_peak_is_the_global_max_with_its_temperature():
    d = _run(FX / "tto_synth.dat").data
    peak = d["summary"]["zt_peak"]
    tpk = d["summary"]["zt_peak_t_k"]
    allz = [(z, t) for c in d["curves"] for z, t in zip(c["zt"] or [], c["t"]) if z is not None]
    best_z = max(z for z, _ in allz)
    # NOT max((z, t)): that breaks ties by the LARGEST t, whereas _zt_peak keeps the FIRST
    # point attaining the max. On a tied fixture the tuple form would fail spuriously.
    assert peak == pytest.approx(best_z)
    assert any(tpk == pytest.approx(t) for z, t in allz if z == best_z)


def _zt_curve(field, t, zt):
    from cryosweep_core.analyzers.tto import TTOCurve
    return TTOCurve(field_oe=field, direction="down", n_points=len(t), t=list(t),
                    kappa=[1.0] * len(t), zt=list(zt))


def test_zt_peak_searches_ACROSS_curves_not_just_the_first():
    # I5 mutation pin: inserting `break` after the first curve in _zt_peak survives the rest
    # of the suite, because on tto_synth.dat the global max happens to sit on curves[0]. The
    # spec says "max valid ZT across all curves" (§2 step 7) — on a multi-field file the
    # headline ZT would otherwise be taken from the wrong curve.
    from cryosweep_core.analyzers.tto import _zt_peak
    first = _zt_curve(0.0, [10.0, 20.0, 30.0], [1e-4, 2e-4, 3e-4])
    later = _zt_curve(90000.0, [11.0, 21.0, 31.0], [4e-4, 9e-4, 5e-4])
    v, t, edge, _std = _zt_peak([first, later])
    assert v == pytest.approx(9e-4)          # the later curve's max, not the first curve's
    assert t == pytest.approx(21.0)
    assert edge is False                      # and it is an interior point of that curve


def test_zt_peak_flags_a_maximum_that_sits_at_a_temperature_range_edge():
    # I3: a maximum at an end of the measured range is not an observed peak — the sweep just
    # stopped there. Flagged so the summary/CSV/GUI cannot claim more than the data supports.
    from cryosweep_core.analyzers.tto import _zt_peak
    at_top = _zt_curve(0.0, [2.0, 3.0, 4.0], [1e-4, 2e-4, 3e-4])
    at_bottom = _zt_curve(0.0, [2.0, 3.0, 4.0], [3e-4, 2e-4, 1e-4])
    interior = _zt_curve(0.0, [2.0, 3.0, 4.0], [1e-4, 3e-4, 2e-4])
    assert _zt_peak([at_top])[1:3] == (4.0, True)
    assert _zt_peak([at_bottom])[1:3] == (2.0, True)
    assert _zt_peak([interior])[1:3] == (3.0, False)
    assert _zt_peak([]) == (None, None, None, None)


def test_zt_peak_ties_keep_the_FIRST_maximum_including_its_sigma():
    """m2 (final review). Mutation `zv > best_v` -> `zv >= best_v` SURVIVED the whole suite.
    The invariant is stated in `_zt_peak`'s docstring and is the WHOLE REASON `std` is tracked
    inside the scan (spec §2.4's I4): the winning row index is not recoverable from the value
    afterwards, precisely because ties keep the first. Under `>=` the last tied row wins and
    `zt_peak_std` silently comes from a different measurement than `zt_peak_t_k` claims."""
    from cryosweep_core.analyzers.tto import TTOCurve, _zt_peak
    tied = TTOCurve(field_oe=0.0, direction="down", n_points=3, t=[10.0, 20.0, 30.0],
                    kappa=[1.0] * 3, zt=[5e-4, 3e-4, 5e-4],
                    zt_std=[1e-6, 9e-9, 7e-6])
    v, t, edge, std = _zt_peak([tied])
    assert v == pytest.approx(5e-4)
    assert t == pytest.approx(10.0)                # the FIRST maximum, not the last
    assert std == pytest.approx(1e-6)              # ... and ITS sigma, not the last one's
    # across curves too: an equal maximum on a later curve must not displace the first
    later = TTOCurve(field_oe=90000.0, direction="down", n_points=1, t=[99.0],
                     kappa=[1.0], zt=[5e-4], zt_std=[4e-5])
    v2, t2, _e2, std2 = _zt_peak([tied, later])
    assert (v2, t2) == (pytest.approx(5e-4), pytest.approx(10.0))
    assert std2 == pytest.approx(1e-6)


def test_capability_set_and_deferred_entries():
    caps = _caps(_run(FX / "tto_synth.dat").data)
    assert caps["thermal_conductivity"]["applicable"] is True
    for name in ("seebeck", "wiedemann_franz", "power_factor", "figure_of_merit", "rrr"):
        assert caps[name]["applicable"] is True, name
    for name in ("callaway_fit", "boundary_scattering_fit", "diffusive_seebeck",
                 "kappa_field_sweep"):
        # m10: boundary_scattering_fit's reason is no longer the bare word — a reader seeing
        # `kappa_ph_power_fit: fitted` beside it must be told they are different claims.
        assert caps[name]["applicable"] is False, name
        assert caps[name]["reason"].startswith("deferred"), name


def test_gap_fixture_declines_seebeck_and_power_factor():
    caps = _caps(_run(FX / "tto_gap_synth.dat").data)
    assert caps["seebeck"]["applicable"] is False
    assert caps["seebeck"]["reason"] == "no Seebeck data"
    assert caps["power_factor"]["applicable"] is False
    assert caps["wiedemann_franz"]["applicable"] is True


def test_norho_fixture_declines_wiedemann_franz_and_figure_of_merit():
    caps = _caps(_run(FX / "tto_norho_synth.dat").data)
    assert caps["wiedemann_franz"]["applicable"] is False
    assert caps["wiedemann_franz"]["reason"] == "requires finite ρ > 0"
    assert caps["figure_of_merit"]["applicable"] is False
    assert caps["seebeck"]["applicable"] is True


def test_real_subset_rrr_and_classification_track_the_full_file():
    # NO _require_real(): the subset is COMMITTED, so the slice keeps a real-SHAPE RRR oracle
    # on machines without the 976-row original. The subset's endpoints are NOT the full file's
    # (body[::4] keeps different extreme rows), so the value is 1.45239, not 1.4555 — both
    # measured at Task 2 Step 6. Tight on purpose: a loose band would hide an
    # endpoint-convention regression, which is the whole point of this oracle.
    d = _run(FX / "tto_real_subset.dat").data
    assert d["rrr"]["rrr"] == pytest.approx(1.452393, abs=1e-5)
    assert d["rrr"]["t_high_k"] == pytest.approx(288.4163, abs=1e-3)
    assert d["rrr"]["t_low_k"] == pytest.approx(2.555393, abs=1e-5)
    assert d["rrr"]["classification"] == "metallic"


def test_real_file_rrr_classification_and_zt_peak(tto_real_path):
    # Tight bands on purpose: [1.45, 1.46] catches an endpoint-convention regression that the
    # loose [1.4, 1.6] band would have hidden (raw-extrema ratio is 1.476, NOT the reported
    # 1.4555).
    d = _run(tto_real_path).data
    assert 1.45 <= d["rrr"]["rrr"] <= 1.46
    assert d["rrr"]["classification"] == "metallic"
    assert 3.92e-4 <= d["summary"]["zt_peak"] <= 3.93e-4
    assert d["summary"]["zt_peak_t_k"] == pytest.approx(301.37, abs=0.01)
    assert d["summary"]["pf_at_thigh"] == pytest.approx(4.645e-6, rel=1e-3)
    # I3: on THIS file the maximum is the last measured point (T_max = 301.370002 K, ZT still
    # rising) — no interior maximum was observed, and the summary must say so.
    assert d["summary"]["zt_peak_at_edge"] is True
    assert d["summary"]["zt_peak_t_k"] == pytest.approx(max(d["curves"][0]["t"]))
