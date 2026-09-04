"""Analyzer wiring of the kappa_ph power-law fit (spec §1, analyzer side + I7).

NOTE (M3): the `_CAP_ADVISORY` / `_CAP_LABELS` assertions for the new capability live in
`tests/gui/test_tto_panel.py` (Step 4b), NOT here — no test under `tests/core` imports
`cryosweep_gui`, and the core suite must not start depending on Qt."""
import json
import math
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


def _analyze_with_selected_curve(monkeypatch, t, kappa_ph):
    """analyze() the real subset, but with `_kappa_ph_fit_curve` forced to return a curve we
    built. Pins the CALL SITE: the analyzer must fit whatever the kappa_ph selector returns,
    so swapping that call for `_rrr_curve` (a rho heuristic that knows nothing about kappa_ph)
    fits the file's own curve instead and every number below moves. Uses no new fixture."""
    from cryosweep_core.analyzers import tto as M
    t = [float(v) for v in t]
    curve = M.TTOCurve(field_oe=0.0, direction="down", n_points=len(t), t=t,
                       kappa=[1.0] * len(t), kappa_ph=[float(v) for v in kappa_ph])
    monkeypatch.setattr(M, "_kappa_ph_fit_curve",
                        lambda curves, primary=M._KAPPA_PH_PRIMARY_K: curve)
    return _run(FX / "tto_real_subset.dat").data


def test_the_three_coarse_synth_fixtures_decline_the_fit_with_a_reason():
    # linspace(300, 2, 150) has EXACTLY 5 points at T <= 10 K (2/4/6/8/10) -> below the
    # >=10 floor. The decline must be reported, not silent, and must not crash analyze().
    for name in ("tto_synth.dat", "tto_gap_synth.dat", "tto_norho_synth.dat"):
        d = _run(FX / name).data
        assert d["kappa_ph_fit"] is None, name
        cap = _caps(d)["kappa_ph_power_fit"]
        assert cap["applicable"] is False, name
        assert cap["reason"] == "needs >=10 finite kappa_ph > 0 points below 10 K", name


def test_the_deferred_stubs_are_untouched():
    caps = _caps(_run(FX / "tto_synth.dat").data)
    for name in ("callaway_fit", "boundary_scattering_fit", "diffusive_seebeck",
                 "kappa_field_sweep"):
        assert caps[name]["applicable"] is False
        assert caps[name]["reason"].startswith("deferred")


def test_real_subset_fits_and_every_emitted_number_is_finite():
    # Runs WITHOUT the real-file guard: tto_real_subset.dat is committed, so the real-shape
    # assertions below are never skipped on a machine lacking the gate file.
    d = _run(FX / "tto_real_subset.dat").data
    kf = d["kappa_ph_fit"]
    assert kf is not None
    assert _caps(d)["kappa_ph_power_fit"]["applicable"] is True
    for key in ("n", "n_sigma", "b", "b_sigma", "r2"):
        assert math.isfinite(kf[key]), key
    assert kf["n_points"] == 40                    # pinned (163 on the full gate file)
    assert kf["window_k"][1] == 10.0
    # window_k[0] is the lowest T actually FITTED on the SELECTED curve. Curve 0 is that curve
    # here only because this subset holds a single curve — reference the selection, not [0].
    from cryosweep_core.analyzers.tto import TTOCurve, _kappa_ph_fit_curve
    sel = _kappa_ph_fit_curve([TTOCurve(**c) for c in d["curves"]])
    fitted_t = [t for t, k in zip(sel.t, sel.kappa_ph)
                if k is not None and k > 0 and 0.0 < t <= 10.0]
    assert len(fitted_t) == kf["n_points"]
    assert kf["window_k"][0] == min(fitted_t)
    json.dumps(d, allow_nan=False)                 # the standing envelope gate


def test_ladder_entries_carry_the_pinned_key_set_and_the_spread_identity():
    kf = _run(FX / "tto_real_subset.dat").data["kappa_ph_fit"]
    for e in kf["ladder"]:
        assert set(e) == {"cutoff_k", "method", "n", "sigma", "r2", "n_points"}
    cf = [e for e in kf["ladder"] if e["method"] == "curve_fit"]
    assert len(cf) == 4                                     # 10 / 15 / 20 / 30 K
    # PINNED values, not the same formula the assertion is meant to constrain (a self-formula
    # cannot detect the spread being computed over the wrong subset).
    assert kf["n"] == pytest.approx(2.0350774348699807, rel=1e-9)
    assert kf["n_spread"] == pytest.approx(0.7207143396761797, rel=1e-9)
    assert kf["n_spread"] == pytest.approx(max(e["n"] for e in cf) - min(e["n"] for e in cf))
    ll = [e for e in kf["ladder"] if e["method"] == "loglog"]
    assert len(ll) == 1 and kf["n_loglog"] == pytest.approx(2.0198581263518447, rel=1e-9)
    assert kf["n_loglog"] == pytest.approx(ll[0]["n"])
    assert kf["n_method_delta"] == pytest.approx(abs(kf["n"] - kf["n_loglog"]))
    assert kf["ladder"][-1]["method"] == "loglog"           # loglog is always LAST
    # On THIS file the log-log n sits INSIDE the curve_fit range, so the method filter is not
    # load-bearing here; the test below constructs the case where it is.


def test_the_spread_excludes_the_loglog_rung_when_it_lies_outside_the_range(monkeypatch):
    # kappa_ph = 1e-3*T^3 with a smooth low-T boundary-scattering rollover: log-log weights the
    # rolled-off low-T decade equally and comes back at n = 4.51, ABOVE every curve_fit rung
    # (3.03 .. 3.32). Dropping the `method == "curve_fit"` filter would report the spread as
    # 1.48 -- a method difference mislabelled as window sensitivity.
    t = np.linspace(2.0, 20.0, 60)
    d = _analyze_with_selected_curve(monkeypatch, t, 1.0e-3 * t ** 3 * (t ** 4 / (t ** 4 + 4.0 ** 4)))
    kf = d["kappa_ph_fit"]
    cf = [e["n"] for e in kf["ladder"] if e["method"] == "curve_fit"]
    ll = [e["n"] for e in kf["ladder"] if e["method"] == "loglog"]
    assert len(cf) == 3 and len(ll) == 1
    assert ll[0] > max(cf)                                   # the filter IS load-bearing here
    assert kf["n_spread"] == pytest.approx(max(cf) - min(cf), rel=1e-9)
    assert kf["n_spread"] == pytest.approx(0.28821829046015646, rel=1e-6)
    assert kf["n_spread"] < 0.5 < max(cf + ll) - min(cf + ll)


def test_analyze_fits_the_curve_the_kappa_ph_selector_returned(monkeypatch):
    # THE headline contract (I7), end to end: analyze() must fit the curve
    # `_kappa_ph_fit_curve` chose. Mutating the call site to `_rrr_curve` fits the file's own
    # curve (n = 2.035, 40 points) instead of this exact T^3 one.
    t = np.linspace(2.0, 30.0, 90)
    d = _analyze_with_selected_curve(monkeypatch, t, 1.0e-3 * t ** 3)
    kf = d["kappa_ph_fit"]
    assert kf is not None
    assert kf["n"] == pytest.approx(3.0, rel=1e-6)
    assert kf["b"] == pytest.approx(1.0e-3, rel=1e-6)
    assert kf["n_points"] == int(np.count_nonzero(t <= 10.0))
    assert kf["window_k"] == [pytest.approx(2.0), 10.0]
    assert _caps(d)["kappa_ph_power_fit"]["applicable"] is True


def test_a_single_fitted_rung_reports_n_spread_none_never_zero(monkeypatch):
    # T tops out at 10 K, so the 15/20/30 K rungs are SKIPPED and one curve_fit rung remains.
    # 0.0 here would read as "n is perfectly window-stable" -- the opposite of the truth.
    t = np.linspace(2.0, 10.0, 40)
    d = _analyze_with_selected_curve(monkeypatch, t, 1.0e-3 * t ** 3)
    kf = d["kappa_ph_fit"]
    assert len([e for e in kf["ladder"] if e["method"] == "curve_fit"]) == 1
    assert kf["n_spread"] is None
    assert "ladder_incomplete" in kf["quality_flags"]
    assert "window_sensitive" not in kf["quality_flags"]
    json.dumps(d, allow_nan=False)


def test_a_non_finite_fit_scalar_is_reported_as_no_fit_with_an_honest_reason(monkeypatch):
    # The five fit scalars are bare floats the D11 sanitiser never walks, so a non-finite one
    # would break the standing json.dumps(allow_nan=False) gate. Defence in depth over Task 1's
    # own decline -- pinned here because nothing else exercises it.
    from cryosweep_core.analyzers import tto as M
    from cryosweep_core.result import FitResult

    def _inf_sigma(T, kappa_ph, kappa_e=None, primary=10.0):
        return (FitResult(model="kappa_ph_power", params={"B": 1.0e-3, "n": 3.0},
                          sigma={"B": 1.0e-5, "n": float("inf")}, covariance=[], r2=0.99,
                          n_points=40, fit_range=[2.0, 10.0],
                          units={"B": "", "n": ""}, quality_flags=[]),
                [{"cutoff_k": 10.0, "method": "curve_fit", "n": 3.0, "sigma": float("inf"),
                  "r2": 0.99, "n_points": 40}])

    monkeypatch.setattr(M, "fit_kappa_ph_ladder", _inf_sigma)
    d = _run(FX / "tto_real_subset.dat").data
    assert d["kappa_ph_fit"] is None
    cap = _caps(d)["kappa_ph_power_fit"]
    assert cap["applicable"] is False
    # NOT the point-count reason: a curve WAS selected here, so that reason would be false.
    assert cap["reason"] == ("kappa_ph power-law fit declined on the selected curve "
                             "(no finite B*T^n solution)")
    json.dumps(d, allow_nan=False)


def test_a_raising_fit_declines_with_the_same_honest_reason(monkeypatch):
    from cryosweep_core.analyzers import tto as M

    def _boom(T, kappa_ph, kappa_e=None, primary=10.0):
        raise ValueError("no")

    monkeypatch.setattr(M, "fit_kappa_ph_ladder", _boom)
    d = _run(FX / "tto_real_subset.dat").data
    assert d["kappa_ph_fit"] is None
    assert _caps(d)["kappa_ph_power_fit"]["reason"] == (
        "kappa_ph power-law fit declined on the selected curve (no finite B*T^n solution)")


def test_fit_curve_selection_prefers_kappa_ph_density_not_the_widest_t_span():
    # I7 (non-vacuous): the WIDEST zero-field curve here carries NO kappa_ph at all, while a
    # narrow high-field curve carries plenty. _rrr_curve would pick the wide one and the fit
    # would decline blaming point count.
    from cryosweep_core.analyzers.tto import TTOCurve, _kappa_ph_fit_curve, _rrr_curve
    wide = TTOCurve(field_oe=0.0, direction="down", n_points=200,
                    t=list(np.linspace(2.0, 300.0, 200)), kappa=[1.0] * 200,
                    kappa_ph=[None] * 200)
    narrow = TTOCurve(field_oe=90000.0, direction="down", n_points=40,
                      t=list(np.linspace(2.0, 10.0, 40)), kappa=[1.0] * 40,
                      kappa_ph=[1.0e-3 * t ** 3 for t in np.linspace(2.0, 10.0, 40)])
    assert _rrr_curve([wide, narrow]) is wide                # the old rule picks the empty one
    assert _kappa_ph_fit_curve([wide, narrow]) is narrow


def test_fit_curve_selection_breaks_ties_towards_zero_field_then_curve_order():
    from cryosweep_core.analyzers.tto import TTOCurve, _kappa_ph_fit_curve
    t = list(np.linspace(2.0, 10.0, 40))
    kph = [1.0e-3 * v ** 3 for v in t]
    hi = TTOCurve(field_oe=90000.0, direction="down", n_points=40, t=t, kappa=[1.0] * 40,
                  kappa_ph=kph)
    zf = TTOCurve(field_oe=0.077, direction="down", n_points=40, t=t, kappa=[1.0] * 40,
                  kappa_ph=kph)
    assert _kappa_ph_fit_curve([hi, zf]) is zf                # zero field wins the tie
    zf2 = TTOCurve(field_oe=1.0, direction="up", n_points=40, t=t, kappa=[1.0] * 40,
                   kappa_ph=kph)
    assert _kappa_ph_fit_curve([zf, zf2]) is zf               # then first by curve order


def test_fit_curve_selection_returns_none_below_the_ten_point_floor():
    from cryosweep_core.analyzers.tto import TTOCurve, _kappa_ph_fit_curve
    t = [2.0, 4.0, 6.0, 8.0, 10.0]
    c = TTOCurve(field_oe=0.0, direction="down", n_points=5, t=t, kappa=[1.0] * 5,
                 kappa_ph=[1.0] * 5)
    assert _kappa_ph_fit_curve([c]) is None


def test_a_length_mismatched_kappa_ph_is_skipped_not_crashed_on():
    # Structural safety: a kappa_ph shorter than t would otherwise raise a bare numpy
    # broadcast ValueError inside the selector — OUTSIDE analyze()'s fit try/except.
    from cryosweep_core.analyzers.tto import TTOCurve, _kappa_ph_fit_curve
    t = list(np.linspace(2.0, 10.0, 40))
    bad = TTOCurve(field_oe=0.0, direction="down", n_points=40, t=t, kappa=[1.0] * 40,
                   kappa_ph=[1.0e-3] * 39)
    good = TTOCurve(field_oe=0.0, direction="down", n_points=40, t=t, kappa=[1.0] * 40,
                    kappa_ph=[1.0e-3 * v ** 3 for v in t])
    assert _kappa_ph_fit_curve([bad]) is None
    assert _kappa_ph_fit_curve([bad, good]) is good


def test_powerlaw_fixture_recovers_n_equals_three_at_every_rung():
    d = _run(FX / "tto_powerlaw_synth.dat").data
    assert len(d["curves"]) == 1
    assert d["curves"][0]["direction"] == "down" and d["curves"][0]["n_points"] == 150
    kf = d["kappa_ph_fit"]
    assert kf is not None
    assert kf["n"] == pytest.approx(3.0, abs=1e-6)
    assert kf["b"] == pytest.approx(1.0e-3, rel=1e-6)
    assert kf["r2"] == pytest.approx(1.0, abs=1e-7)
    assert kf["n_points"] == 43
    assert kf["window_k"] == [pytest.approx(2.0), 10.0]
    assert len(kf["ladder"]) == 5
    assert [e["n_points"] for e in kf["ladder"] if e["method"] == "curve_fit"] == [43, 70, 96, 150]
    for e in kf["ladder"]:
        assert e["n"] == pytest.approx(3.0, abs=1e-6), e
    assert kf["n_loglog"] == pytest.approx(3.0, abs=1e-6)
    assert kf["n_method_delta"] < 1e-8


def test_powerlaw_fixture_does_not_cry_wolf_and_carries_no_quality_flags():
    # THE reason the 0.05 floor exists (C4). Under the earlier ratio-only rule the measured
    # n_spread 7.37e-9 beats 3*sigma_n 3.87e-9 and window_sensitive FIRES on data where n is
    # recovered to 9 decimals. n_spread's digits are convergence noise -> only "< 0.05" is
    # asserted; the digits are recorded in the spec as a measurement, not pinned here.
    kf = _run(FX / "tto_powerlaw_synth.dat").data["kappa_ph_fit"]
    assert kf["quality_flags"] == []
    assert kf["n_spread"] is not None and kf["n_spread"] < 0.05


def test_analyze_is_deterministic_with_the_fit_present():
    a = json.dumps(_run(FX / "tto_real_subset.dat").data, sort_keys=True, allow_nan=False)
    b = json.dumps(_run(FX / "tto_real_subset.dat").data, sort_keys=True, allow_nan=False)
    assert a == b


# ---------------------------------------------------------------------------------------
# Final-review C1: a bound-pinned / degenerate / worse-than-a-constant fit is NOT a
# measurement and must be declined, not reported. `tto_deltat_synth.dat` is this slice's own
# fixture and reproduces the defect exactly: kappa_ph is flat below 10 K (constant to 1e-8),
# so curve_fit parks on the model's lower bound and returns n = 0.5 with r2 = -3.6e13 and
# n_spread = 1.1e-16 -- numbers that READ as a perfectly window-stable exponent.
# ---------------------------------------------------------------------------------------

def test_a_bound_pinned_flat_kappa_ph_declines_instead_of_reporting_n_equals_the_bound():
    d = _run(FX / "tto_deltat_synth.dat").data
    assert d["kappa_ph_fit"] is None
    cap = _caps(d)["kappa_ph_power_fit"]
    assert cap["applicable"] is False
    # The reason must name the defect AND carry its numbers -- a bare "declined" would send
    # the reader looking for a missing input that does not exist on this probe.
    assert cap["reason"] == (
        "kappa_ph is not a power law below 10 K — the exponent pinned at the search bound "
        "(n = 0.5); the power law fits worse than a constant (r2 = -3.57e+13) "
        "— no exponent is reported")
    # NOT the point-count reason and NOT the generic no-solution reason: a curve was selected
    # and a solution WAS returned; it just is not one.
    assert "points" not in cap["reason"] and "no finite" not in cap["reason"]
    json.dumps(d, allow_nan=False)


def test_the_declined_bound_pinned_fit_leaves_every_csv_kappa_ph_cell_blank(tmp_path):
    from cryosweep_core.io.export import export_result
    import csv as _csv
    out = export_result(_run(FX / "tto_deltat_synth.dat"), tmp_path / "d")
    with open(out["tto_summary"], newline="") as f:
        row = next(iter(_csv.DictReader(f)))
    for col in ("kappa_ph_n", "kappa_ph_n_sigma", "kappa_ph_n_spread", "kappa_ph_n_loglog",
                "kappa_ph_n_method_delta", "kappa_ph_b", "kappa_ph_r2",
                "kappa_ph_window_k_max", "kappa_ph_flags"):
        assert row[col] == "", col


def test_a_degenerate_single_temperature_window_declines(monkeypatch):
    # nothing is fitted: n comes back as exactly p0 (2.0) and r2 as 0.0.
    d = _analyze_with_selected_curve(monkeypatch, [5.0] * 40, [1.0e-3] * 40)
    assert d["kappa_ph_fit"] is None
    reason = _caps(d)["kappa_ph_power_fit"]["reason"]
    assert "single distinct T" in reason and "worse than a constant" in reason


def test_r2_non_positive_alone_declines_even_with_no_quality_flags(monkeypatch):
    # The r2 half of the rule is independent of the flags: a power law that describes the data
    # WORSE THAN ITS OWN MEAN is not a measurement whatever the optimizer reports.
    from cryosweep_core.analyzers import tto as M
    from cryosweep_core.result import FitResult

    def _bad_r2(T, kappa_ph, kappa_e=None, primary=10.0):
        return (FitResult(model="kappa_ph_power", params={"B": 1.0e-3, "n": 3.0},
                          sigma={"B": 1.0e-5, "n": 0.01}, covariance=[], r2=-2.5,
                          n_points=40, fit_range=[2.0, 10.0],
                          units={"B": "", "n": ""}, quality_flags=[]),
                [{"cutoff_k": 10.0, "method": "curve_fit", "n": 3.0, "sigma": 0.01,
                  "r2": -2.5, "n_points": 40}])

    monkeypatch.setattr(M, "fit_kappa_ph_ladder", _bad_r2)
    d = _run(FX / "tto_real_subset.dat").data
    assert d["kappa_ph_fit"] is None
    assert _caps(d)["kappa_ph_power_fit"]["reason"] == (
        "kappa_ph is not a power law below 10 K — the power law fits worse than a constant "
        "(r2 = -2.5) — no exponent is reported")


def test_the_integrity_decline_does_not_touch_a_GOOD_fit():
    # The guard must be surgical. Both good fixtures keep their exact oracles, flags and
    # capability line; a rule that also declined these would have removed a real measurement.
    real = _run(FX / "tto_real_subset.dat").data
    assert real["kappa_ph_fit"] is not None
    assert _caps(real)["kappa_ph_power_fit"]["applicable"] is True
    assert real["kappa_ph_fit"]["quality_flags"] == ["window_sensitive"]
    assert real["kappa_ph_fit"]["r2"] > 0.999

    law = _run(FX / "tto_powerlaw_synth.dat").data
    assert law["kappa_ph_fit"] is not None
    assert law["kappa_ph_fit"]["quality_flags"] == []
    assert law["kappa_ph_fit"]["n"] == pytest.approx(3.0, abs=1e-6)


def test_a_kappa_e_dominant_fit_is_reported_not_declined(monkeypatch):
    # kappa_e_dominant describes the CONTEXT of a real fit, not a broken one -- it must stay a
    # word on the surfaces (see the GUI row test) and must NOT decline. Guards against the
    # over-correction of treating every flag as fatal.
    from cryosweep_core.analyzers import tto as M
    from cryosweep_core.result import FitResult

    def _dominant(T, kappa_ph, kappa_e=None, primary=10.0):
        return (FitResult(model="kappa_ph_power", params={"B": 1.0e-3, "n": 3.0},
                          sigma={"B": 1.0e-5, "n": 0.01}, covariance=[], r2=0.99,
                          n_points=40, fit_range=[2.0, 10.0], units={"B": "", "n": ""},
                          quality_flags=["kappa_e_dominant"]),
                [{"cutoff_k": 10.0, "method": "curve_fit", "n": 3.0, "sigma": 0.01,
                  "r2": 0.99, "n_points": 40}])

    monkeypatch.setattr(M, "fit_kappa_ph_ladder", _dominant)
    d = _run(FX / "tto_real_subset.dat").data
    assert d["kappa_ph_fit"] is not None
    assert d["kappa_ph_fit"]["quality_flags"] == ["kappa_e_dominant"]
    assert _caps(d)["kappa_ph_power_fit"]["applicable"] is True


def test_fit_curve_selection_picks_the_DENSER_of_two_qualifying_curves():
    # I4 (final review): the older "density not span" test gives its wide curve kappa_ph=None,
    # so that curve is dropped by the 10-point FLOOR before any count comparison runs -- and
    # `key = (-n, ...)` -> `(n, ...)` (pick the FEWEST points) survived the whole suite. Here
    # BOTH curves clear the floor at the SAME field, so only the count ordering can decide.
    from cryosweep_core.analyzers.tto import TTOCurve, _kappa_ph_fit_curve
    t15 = list(np.linspace(2.0, 10.0, 15))
    t40 = list(np.linspace(2.0, 10.0, 40))
    sparse = TTOCurve(field_oe=0.0, direction="down", n_points=15, t=t15,
                      kappa=[1.0] * 15, kappa_ph=[1.0e-3 * v ** 3 for v in t15])
    dense = TTOCurve(field_oe=0.0, direction="up", n_points=40, t=t40,
                     kappa=[1.0] * 40, kappa_ph=[1.0e-3 * v ** 3 for v in t40])
    assert _kappa_ph_fit_curve([sparse, dense]) is dense     # denser wins from behind
    assert _kappa_ph_fit_curve([dense, sparse]) is dense     # ... and from in front


def test_the_boundary_scattering_stub_says_HOW_it_differs_from_the_free_n_fit():
    # m10: with the free-n fit now a REAL capability, "boundary_scattering_fit: deferred" sitting
    # beside "kappa_ph_power_fit: fitted" in the capabilities CSV reads as a contradiction. It is
    # not: this fit MEASURES n (2.0266 on the gate file, which argues AGAINST n = 3); the stub is
    # the n = 3 hypothesis test.
    cap = _caps(_run(FX / "tto_real_subset.dat").data)["boundary_scattering_fit"]
    assert cap["applicable"] is False
    assert cap["reason"] == ("deferred — the n = 3 boundary-scattering hypothesis test, "
                             "distinct from the free-n kappa_ph fit")


def test_n_method_delta_is_ABSOLUTE_even_when_the_loglog_rung_is_larger(monkeypatch):
    """m3. Dropping the `abs()` SURVIVED: on the gate file n > n_loglog, so signed and absolute
    agree there and nothing else exercises the other sign. A negative "delta" in the CSV's
    `kappa_ph_n_method_delta` would read as a direction, not a disagreement magnitude."""
    from cryosweep_core.analyzers import tto as M
    from cryosweep_core.result import FitResult

    def _loglog_larger(T, kappa_ph, kappa_e=None, primary=10.0):
        return (FitResult(model="kappa_ph_power", params={"B": 1.0e-3, "n": 2.0},
                          sigma={"B": 1.0e-5, "n": 0.01}, covariance=[], r2=0.99,
                          n_points=40, fit_range=[2.0, 10.0], units={"B": "", "n": ""},
                          quality_flags=[]),
                [{"cutoff_k": 10.0, "method": "curve_fit", "n": 2.0, "sigma": 0.01,
                  "r2": 0.99, "n_points": 40},
                 {"cutoff_k": 30.0, "method": "curve_fit", "n": 1.5, "sigma": 0.01,
                  "r2": 0.99, "n_points": 90},
                 {"cutoff_k": 10.0, "method": "loglog", "n": 2.6, "sigma": 0.02,
                  "r2": 0.98, "n_points": 40}])

    monkeypatch.setattr(M, "fit_kappa_ph_ladder", _loglog_larger)
    kf = _run(FX / "tto_real_subset.dat").data["kappa_ph_fit"]
    assert kf["n_loglog"] > kf["n"]                       # the premise, pinned
    assert kf["n_method_delta"] == pytest.approx(0.6)     # NOT -0.6
