"""Always-on TTO integrity signals (spec §2): the DeltaT/T warning, the Seebeck low-T
sign-oscillation warning, rrr_std + the classification guard, and zt_peak_std.

E3: these are UNCONDITIONAL. Error bands are opt-in (PlotSpec.error_band), so the numeric
integrity surface must not depend on a toggle."""
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


def _curve(field_oe=0.0, t=(), **kw):
    from cryosweep_core.analyzers.tto import TTOCurve
    return TTOCurve(field_oe=field_oe, direction="up", n_points=len(t), t=list(t),
                    kappa=[1.0] * len(t), **kw)


def _run_mutated(name, mutate):
    """analyze() over a committed fixture whose RAW dataframe is edited IN MEMORY.

    No fixture `.dat` is written or regenerated — `mutate(df)` receives a copy of the loaded
    frame and edits raw columns in place, and the analyzer runs on a `dataclasses.replace`d
    RawTable. This is how the integration-level cases below reach code paths (the
    classification guard's positive branch, a rho column with no sigma column) that no
    committed file exercises."""
    import dataclasses

    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / name))
    df = rt.df.copy()
    mutate(df)
    return TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())


# ---- 1. Delta T / T ---------------------------------------------------------------------

def test_delta_temp_column_is_matched_case_insensitively_on_the_stripped_raw_name():
    import pandas as pd
    from cryosweep_core.analyzers.tto import _delta_temp_column
    assert _delta_temp_column(pd.DataFrame(columns=["  DELTA TEMP. (K) "])) == "  DELTA TEMP. (K) "
    assert _delta_temp_column(pd.DataFrame(columns=["Sample Temp. (K)"])) is None


def test_delta_t_warning_counts_rows_and_reports_the_worst_one():
    from cryosweep_core.analyzers.tto import _delta_t_warning
    T = np.array([2.0, 10.0, 100.0, 200.0])
    dt = np.array([0.5, 0.6, 1.0, 1.0])             # 25 %, 6 %, 1 %, 0.5 %
    msg = _delta_t_warning(dt, T)
    assert msg == ("2 rows have ΔT/T > 5% (max 25.00% at 2.000 K) — "
                   "kappa there is averaged over a wide T window")


def test_delta_t_warning_uses_the_absolute_value_because_the_sign_is_a_wiring_convention():
    # The gate file's DeltaT is positive everywhere (0.0887-5.2496 K), so it cannot test this.
    from cryosweep_core.analyzers.tto import _delta_t_warning
    assert _delta_t_warning(np.array([-0.5]), np.array([2.0])) is not None


def test_delta_t_warning_is_silent_when_nothing_exceeds_five_percent():
    from cryosweep_core.analyzers.tto import _delta_t_warning
    assert _delta_t_warning(np.array([0.05, 0.09]), np.array([2.0, 10.0])) is None


def test_absent_and_all_empty_delta_temp_columns_behave_identically():
    # M5: make_tto never populates column index 14, so every synthetic fixture carries
    # `Delta Temp. (K)` present-and-EMPTY. Those fixtures ARE the negative-case assertion.
    from cryosweep_core.analyzers.tto import _delta_t_warning
    assert _delta_t_warning(np.array([np.nan, np.nan]), np.array([2.0, 10.0])) is None
    for name in ("tto_synth.dat", "tto_powerlaw_synth.dat"):
        r = _run(FX / name)
        assert not any("ΔT/T" in w for w in r.warnings), name


def test_delta_t_rows_are_the_keep_filtered_rows_not_the_emitted_curve_rows():
    # M4, and NON-VACUOUS only because of tto_deltat_synth (I3): 3 rows are dropped BEFORE the
    # oversized ΔT row, so reading the column WITHOUT [keep] shifts the pairing and names a
    # different temperature (6.308 K, or 10.615 K if the two arrays are filtered the other way
    # round; a length-mismatched read raises). The gate file CANNOT test this — measured,
    # keep.sum() == len(df) == 976 — and no other fixture populates the column at all.
    r = _run(FX / "tto_deltat_synth.dat")
    assert any("3 rows dropped" in w for w in r.warnings)      # the drop really happened
    hits = [w for w in r.warnings if "ΔT/T" in w]
    assert hits == ["1 rows have ΔT/T > 5% (max 10.64% at 8.462 K) — "
                    "kappa there is averaged over a wide T window"]


def test_delta_t_ignores_a_huge_delta_t_that_sits_on_a_DROPPED_row():
    # THE non-vacuous [keep] test. `tto_deltat_synth` alone is NOT enough: its 3 dropped rows
    # carry ΔT = 0.01 K at T ~ 25-26 K (0.04 %), so a [keep]-less read of BOTH arrays produces
    # the identical message and the mutation "drop [keep] from tto.py:542-543" survives.
    # Here a dropped row (kappa = -1, index 5) carries ΔT = 5 K at T = 26.410 K -> 18.9 %:
    # an unfiltered read would report 2 rows and name 26.410 K as the worst. The aligned read
    # must see only the surviving oversized row, unchanged from the untouched fixture.
    import numpy as _np
    def mutate(df):
        d = _np.array(df["Delta Temp. (K)"], float)
        d[5] = 5.0                                   # index 5 has kappa = -1 -> D6-dropped
        df["Delta Temp. (K)"] = d
    r = _run_mutated("tto_deltat_synth.dat", mutate)
    assert any("3 rows dropped" in w for w in r.warnings)
    hits = [w for w in r.warnings if "ΔT/T" in w]
    assert hits == ["1 rows have ΔT/T > 5% (max 10.64% at 8.462 K) — "
                    "kappa there is averaged over a wide T window"]


def test_a_fixture_with_an_empty_delta_temp_column_still_warns_about_nothing():
    # The companion negative case: tto_gap_synth drops the same 3 rows but its ΔT column is
    # present-and-EMPTY, so the analyzer must stay silent rather than raise on the shape.
    r = _run(FX / "tto_gap_synth.dat")
    assert any("3 rows dropped" in w for w in r.warnings)
    assert not any("ΔT/T" in w for w in r.warnings)


def test_real_file_delta_t_warning_oracle(tto_real_path):
    r = _run(tto_real_path)
    hits = [w for w in r.warnings if "ΔT/T" in w]
    assert hits == ["20 rows have ΔT/T > 5% (max 11.72% at 2.025 K) — "
                    "kappa there is averaged over a wide T window"]


# ---- 2. Seebeck sign oscillation --------------------------------------------------------

def test_sign_oscillation_warns_on_a_dense_burst_of_reversals():
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    t = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0]
    s = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]                      # 5 reversals in 1.0 K
    msgs = _seebeck_oscillation_warning([_curve(t=t, seebeck=s)])
    assert msgs == ["S changes sign 5 times between 10.000 K and 11.000 K (a 1.000 K window) — "
                    "the low-T sign structure oscillates from point to point"]


def test_sign_oscillation_is_silent_on_a_single_clean_zero_crossing():
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    t = [2.0, 5.0, 8.0, 11.0, 14.0, 17.0]
    s = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
    assert _seebeck_oscillation_warning([_curve(t=t, seebeck=s)]) == []


def test_sign_oscillation_is_silent_when_the_reversals_are_spread_over_a_wide_window():
    # The trigger is DENSITY: 5 reversals across 19 K is a slow wiggle, not point-to-point noise.
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    t = [1.0, 4.0, 8.0, 12.0, 16.0, 19.5]
    s = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert _seebeck_oscillation_warning([_curve(t=t, seebeck=s)]) == []


def test_sign_oscillation_ignores_reversals_above_twenty_kelvin():
    # 20 K is the bound in BOTH §2.2 and the oracle. Nothing in this slice may say 12 K.
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    t = [20.5, 20.7, 20.9, 21.1, 21.3, 21.5]
    s = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    assert _seebeck_oscillation_warning([_curve(t=t, seebeck=s)]) == []


def test_sign_oscillation_still_fires_between_twelve_and_twenty_kelvin():
    # ENFORCES the "NEVER 12 K" comment on _S_OSC_MAX_T_K. The 20.5-21.5 K silence test above
    # is silent under a 12 K bound too, and the gate file's burst (10.19-11.91 K) sits below
    # 12 K — so without this case the mutation _S_OSC_MAX_T_K 20.0 -> 12.0 survives the suite.
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    t = [13.0, 13.5, 14.0, 14.5, 15.0, 15.5]
    s = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]                      # 5 reversals in 2.5 K
    assert _seebeck_oscillation_warning([_curve(t=t, seebeck=s)]) == [
        "S changes sign 5 times between 13.000 K and 15.500 K (a 2.500 K window) — "
        "the low-T sign structure oscillates from point to point"]


def test_sign_oscillation_needs_at_least_five_reversals():
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    t = [10.0, 10.2, 10.4, 10.6, 10.8]
    s = [1.0, -1.0, 1.0, -1.0, 1.0]                            # only 4
    assert _seebeck_oscillation_warning([_curve(t=t, seebeck=s)]) == []


def test_sign_oscillation_does_not_use_seebeck_std():
    # C1: the discarded rule gated on |S| <= seebeck_std and measured ZERO crossings on the
    # gate file at every multiplier (each bracketing point is 11.4-45.5 sigma from zero).
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    t = [10.0, 10.2, 10.4, 10.6, 10.8, 11.0]
    s = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
    huge = [1e6] * 6
    tiny = [1e-9] * 6
    a = _seebeck_oscillation_warning([_curve(t=t, seebeck=s, seebeck_std=huge)])
    b = _seebeck_oscillation_warning([_curve(t=t, seebeck=s, seebeck_std=tiny)])
    assert a == b != []


def test_sign_oscillation_is_silent_without_seebeck():
    from cryosweep_core.analyzers.tto import _seebeck_oscillation_warning
    assert _seebeck_oscillation_warning([_curve(t=[2.0, 3.0, 4.0])]) == []


def test_real_file_sign_oscillation_oracle(tto_real_path):
    r = _run(tto_real_path)
    hits = [w for w in r.warnings if "changes sign" in w]
    # The width is `.3f` of the RAW difference, 11.9104129 − 10.1857324 = 1.7246805 -> 1.725.
    # It is NOT 11.910 − 10.186 = 1.724 (the difference of the already-rounded bounds); if this
    # assertion fails with "1.724", fix the ORACLE's reader, never the implementation.
    assert hits == ["S changes sign 11 times between 10.186 K and 11.910 K "
                    "(a 1.725 K window) — the low-T sign structure oscillates "
                    "from point to point"]


# ---- 3. rrr_std + classification guard ---------------------------------------------------

def test_endpoint_sigma_is_the_standard_error_of_a_MEDIAN_not_of_one_point():
    # C3: 1.2533 = sqrt(pi/2) is the median's efficiency penalty; /sqrt(5) is the averaging.
    # Propagating the per-point sigma raw overstates sigma_RRR by 1/0.5605 = 1.784x.
    from cryosweep_core.analyzers.tto import _endpoint_sigma
    T = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 100.0])
    rho = np.full(6, 1e-5)
    sd = np.full(6, 1e-7)
    got = _endpoint_sigma(T, rho, sd, True)
    assert got == pytest.approx(1.2533 * 1e-7 / math.sqrt(5))
    assert got == pytest.approx(0.5605 * 1e-7, rel=1e-3)


def test_endpoint_sigma_uses_k_equals_min_five_and_the_same_mask_as_endpoint():
    from cryosweep_core.analyzers.tto import _endpoint_sigma
    T = np.array([2.0, 4.0, 6.0])
    rho = np.array([1e-5, -1.0, 1e-5])          # the negative row is masked out by rho > 0
    sd = np.array([1e-7, 9.9, 1e-7])
    assert _endpoint_sigma(T, rho, sd, True) == pytest.approx(1.2533 * 1e-7 / math.sqrt(2))


def test_endpoint_sigma_is_none_when_no_finite_sigma_survives():
    from cryosweep_core.analyzers.tto import _endpoint_sigma
    T = np.array([2.0, 4.0, 6.0])
    assert _endpoint_sigma(T, np.full(3, 1e-5), np.full(3, np.nan), True) is None


def test_rrr_std_on_the_powerlaw_fixture_is_the_measured_oracle():
    d = _run(FX / "tto_powerlaw_synth.dat").data
    assert d["rrr"]["rrr"] == pytest.approx(1.0, rel=1e-12)
    # MEASURED here, not taken from the brief (which quoted an unverified 0.00793 for a field
    # that did not exist yet): rrr_std = 0.007926565182978058. The brief's figure is the same
    # number to 3 s.f., but this pin is the one that was actually observed.
    assert d["rrr"]["rrr_std"] == pytest.approx(0.007926565182978058, rel=1e-12)
    assert d["rrr"]["classification"] == "non_monotonic"


def test_endpoint_sigma_is_none_rather_than_inf_when_the_scaling_overflows():
    # tto.py:442. `s` is already isfinite-filtered, so the only way `val` goes non-finite is
    # overflow in 1.2533 * median(s): sigma = 1.7e308 -> 2.13e308 -> inf. `_endpoint_sigma`'s
    # result becomes RRRBlock.rrr_std, a BARE float `_san` never walks, so an inf there would
    # break the standing json.dumps(allow_nan=False) gate. Guard at the source.
    from cryosweep_core.analyzers.tto import _endpoint_sigma
    T = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 100.0])
    assert _endpoint_sigma(T, np.full(6, 1e-5), np.full(6, 1.7e308), True) is None


def test_rrr_sigma_is_none_rather_than_inf_when_a_subnormal_rho_overflows_the_ratio():
    # tto.py:458, and the SAME subnormal-rho hazard `_rrr` documents at tto.py:233: rho_lo is
    # subnormal but > 0, so every earlier guard passes, and sigma_lo/rho_lo overflows to inf.
    # NOTE the reviewer's sketched route (a huge sigma with a small rho) does NOT reach this
    # line: `(s_hi / r_hi) ** 2` on two ordinary floats raises OverflowError before the guard.
    # The subnormal route divides to inf first, and inf ** 2 is inf, so the guard is reached.
    from cryosweep_core.analyzers.tto import _rrr_sigma
    T = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 100.0])
    rho = np.array([5e-324, 5e-324, 5e-324, 5e-324, 5e-324, 1e-5])
    assert _rrr_sigma(T, rho, np.full(6, 1e-5), 1.0) is None


def test_rrr_sigma_is_none_when_every_sigma_is_non_finite():
    # Unit level: an all-NaN sigma array (NOT "a file with no rho_std column" — that is the
    # `sel.rho_std is None` branch, covered by the integration test below).
    from cryosweep_core.analyzers.tto import _rrr_sigma
    T = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 100.0])
    rho = np.linspace(1e-8, 1e-7, 6)
    assert _rrr_sigma(T, rho, np.full(6, np.nan), 2.0) is None


def test_rrr_std_is_none_when_the_curve_carries_rho_but_no_rho_std_column():
    # tto.py:633 (`if sel.rho_std is not None`). No committed fixture has rho WITHOUT its
    # sigma, so this branch had no coverage; blanking the raw sigma column in memory makes
    # `_san` emit None for the whole array and the RRR block must still be produced.
    r = _run_mutated("tto_powerlaw_synth.dat",
                     lambda df: df.__setitem__("Resist Std.Dev.", np.nan))
    assert r.data["rrr"]["rrr"] == pytest.approx(1.0, rel=1e-12)
    assert r.data["rrr"]["rrr_std"] is None
    assert r.data["rrr"]["classification"] == "non_monotonic"
    assert not any("classification_uncertain" in w for w in r.warnings)
    json.dumps(r.data, allow_nan=False)


def test_rrr_std_on_the_real_subset_fixture_is_the_measured_oracle():
    # THE headline rrr_std pin that is not skippable. Without it, `median(sigma)` ->
    # `mean(sigma)` at tto.py:441 survives the whole suite: the only other rrr_std pin
    # (powerlaw) has a CONSTANT sigma = 1e-7 where median == mean identically.
    # Measured: median 0.02642013284369091, mean-mutant 0.043203181262137694 (+64 %).
    d = _run(FX / "tto_real_subset.dat").data
    assert d["rrr"]["rrr"] == pytest.approx(1.452392853636452, rel=1e-9)
    assert d["rrr"]["rrr_std"] == pytest.approx(0.02642013284369091, rel=1e-9)


def test_real_file_rrr_std_oracle(tto_real_path):
    # The gate-file value itself was pinned by no test anywhere. Measured: 0.01741809200431333;
    # the mean-mutant reads 0.027756039381275226 (+59 %) — a silent 59 % error bar inflation.
    d = _run(tto_real_path).data
    assert d["rrr"]["rrr"] == pytest.approx(1.455546631882915, rel=1e-9)
    assert d["rrr"]["rrr_std"] == pytest.approx(0.01741809200431333, rel=1e-9)


def test_classification_guard_fires_only_when_the_band_straddles_a_threshold():
    from cryosweep_core.analyzers.tto import _straddles_threshold
    assert _straddles_threshold(1.00, 0.05) is True          # band [0.95, 1.05] spans BOTH
    assert _straddles_threshold(1.03, 0.02) is True          # spans 1.02
    assert _straddles_threshold(0.99, 0.02) is True          # spans 0.98
    assert _straddles_threshold(1.00, 0.005) is False        # band [0.995, 1.005]: neither
    assert _straddles_threshold(8.373, 0.0664) is False      # tto_synth: nowhere near


def test_no_committed_fixture_trips_the_classification_guard():
    # Regression guard the spec REQUIRES: the guard must not fire on good data.
    for name, cls in (("tto_synth.dat", "metallic"), ("tto_gap_synth.dat", "metallic"),
                      ("tto_real_subset.dat", "metallic"),
                      ("tto_deltat_synth.dat", "metallic"),      # RRR 1.683, band ~ ±0.013
                      ("tto_powerlaw_synth.dat", "non_monotonic")):
        r = _run(FX / name)
        assert r.data["rrr"]["classification"] == cls, name
        assert not any("classification_uncertain" in w for w in r.warnings), name


def test_classification_guard_positive_path_through_analyze_pins_the_warning_string():
    # The guard's POSITIVE branch (tto.py:636-644) never ran through analyze(): only the
    # `_straddles_threshold` predicate was unit-tested, which proves the predicate and nothing
    # about the wiring — the warning STRING was entirely unasserted and could be rewritten
    # freely. Task 9 consumes that string (the `classification_uncertain:` prefix is THE
    # disambiguator for the overloaded "unknown"), so it is pinned here.
    #
    # Built IN MEMORY from tto_powerlaw_synth (no new fixture .dat): a 1 % linear-in-T tilt on
    # the constant rho lifts RRR to 1.00908, and doubling the raw sigma column widens the band
    # to +-0.01591 = [0.99316, 1.02499], which straddles 1.02 (and only 1.02).
    def mutate(df):
        T = np.array(df["Sample Temp. (K)"], float)
        df["Resistivity (Ohm-m)"] = np.array(df["Resistivity (Ohm-m)"], float) * (
            1.0 + 0.01 * T / 30.0)
        df["Resist Std.Dev."] = np.array(df["Resist Std.Dev."], float) * 2.0
    r = _run_mutated("tto_powerlaw_synth.dat", mutate)
    assert r.data["rrr"]["rrr"] == pytest.approx(1.0090755866792849, rel=1e-9)
    assert r.data["rrr"]["rrr_std"] == pytest.approx(0.015912629129687068, rel=1e-9)
    # Without the guard `_classify` would have called this "metallic" (1.00908 < 1.02 -> in
    # fact "non_monotonic"); the guard must overwrite it AND say why.
    assert r.data["rrr"]["classification"] == "unknown"
    hits = [w for w in r.warnings if w.startswith("classification_uncertain")]
    assert hits == ["classification_uncertain: RRR = 1.009 ± 0.016 straddles a "
                    "metal/insulator threshold (1.02 / 0.98)"]


def test_unknown_from_invalid_endpoints_carries_NO_uncertainty_warning():
    # M7: "unknown" is overloaded (_classify already returns it for rho_lo <= 0). The WARNING
    # is the disambiguator, so it must be absent in the invalid-endpoint case.
    r = _run(FX / "tto_norho_synth.dat")
    assert r.data["rrr"] is None                              # no rho at all -> no RRR block
    assert not any("classification_uncertain" in w for w in r.warnings)


def test_rrr_std_and_zt_peak_std_never_reach_json_as_nan():
    for name in ("tto_synth.dat", "tto_gap_synth.dat", "tto_norho_synth.dat",
                 "tto_real_subset.dat", "tto_powerlaw_synth.dat",
                 "tto_deltat_synth.dat"):
        json.dumps(_run(FX / name).data, allow_nan=False)


# ---- 4. zt_peak_std ----------------------------------------------------------------------

def test_zt_peak_returns_a_four_tuple_carrying_the_std_at_the_winning_row():
    # I4: the value is NOT recoverable after the fact -- ties keep the FIRST maximum, so
    # re-scanning by value is fragile. It is tracked in the same loop.
    from cryosweep_core.analyzers.tto import _zt_peak
    c = _curve(t=[2.0, 3.0, 4.0], zt=[1e-4, 3e-4, 2e-4], zt_std=[1e-6, 7e-6, 2e-6])
    assert _zt_peak([c]) == (pytest.approx(3e-4), pytest.approx(3.0), False,
                             pytest.approx(7e-6))


def test_zt_peak_std_is_none_when_zt_std_is_absent_short_or_non_finite():
    from cryosweep_core.analyzers.tto import _zt_peak
    base = dict(t=[2.0, 3.0, 4.0], zt=[1e-4, 3e-4, 2e-4])
    assert _zt_peak([_curve(**base)])[3] is None                            # absent
    assert _zt_peak([_curve(**base, zt_std=[1e-6])])[3] is None             # short
    assert _zt_peak([_curve(**base, zt_std=[1e-6, None, 2e-6])])[3] is None  # None at the row


def test_zt_peak_std_is_surfaced_in_the_summary():
    d = _run(FX / "tto_synth.dat").data
    assert d["summary"]["zt_peak_std"] == pytest.approx(0.01 * d["summary"]["zt_peak"],
                                                        rel=1e-9)


def test_real_file_zt_peak_std_oracle(tto_real_path):
    s = _run(tto_real_path).data["summary"]
    assert s["zt_peak_std"] == pytest.approx(1.59828e-5, rel=1e-4)
    assert s["zt_peak"] == pytest.approx(3.92322e-4, rel=1e-4)
