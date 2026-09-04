"""TTO analyzer core: gates, row filter, error-code counting, field grouping, ramp split,
curve arrays (spec §2 steps 1-4)."""
import dataclasses
import json
import pathlib

import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat

FX = pathlib.Path("tests/core/fixtures")


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def test_analyzer_contract():
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    a = TTOAnalyzer()
    assert a.probe == "tto"
    assert a.needs == ()


def test_registered_in_builtin_analyzers():
    from cryosweep_core.registry import build_default_registry
    reg = build_default_registry()
    assert "tto" in reg.analyzer_keys()
    assert reg.get_analyzer("tto") is not None


def test_synth_status_ok_and_two_groups_one_curve_each():
    r = _run(FX / "tto_synth.dat")
    assert r.status == "ok"
    d = r.data
    assert d["probe"] == "tto"
    assert len(d["curves"]) == 2
    fields = sorted(round(c["field_oe"], 3) for c in d["curves"])
    assert fields == [0.0, 90000.0]
    assert d["dropped_groups"] == []


def test_synth_curves_are_temperature_ascending_and_full_length():
    d = _run(FX / "tto_synth.dat").data
    by_field = {round(c["field_oe"], 3): c for c in d["curves"]}
    main = by_field[0.0]
    assert main["n_points"] == 150          # the inclusive-i1 +1 slice keeps the last point
    assert by_field[90000.0]["n_points"] == 30
    t = np.asarray(main["t"], float)
    assert np.all(np.diff(t) > 0)
    assert t[0] == pytest.approx(2.0) and t[-1] == pytest.approx(300.0)


def test_synth_parallel_arrays_stay_aligned_after_the_t_sort():
    d = _run(FX / "tto_synth.dat").data
    c = next(c for c in d["curves"] if round(c["field_oe"], 3) == 0.0)
    t = np.asarray(c["t"], float)
    assert np.asarray(c["seebeck"], float) == pytest.approx(0.01 * t, rel=1e-7)
    assert np.asarray(c["rho"], float) == pytest.approx(1e-8 * (1 + 9 * t / 300), rel=1e-7)
    assert np.asarray(c["kappa_std"], float) == pytest.approx(
        0.01 * np.asarray(c["kappa"], float), rel=1e-7)


def test_direction_is_down_for_a_cooling_sweep():
    d = _run(FX / "tto_synth.dat").data
    assert {c["direction"] for c in d["curves"]} == {"down"}


def test_synth_error_rows_counted_and_warned():
    r = _run(FX / "tto_synth.dat")
    assert r.data["n_error_rows"] == 3
    assert any("instrument error codes (kept)" in w for w in r.warnings)


def test_gap_fixture_drops_nonpositive_kappa_and_warns():
    r = _run(FX / "tto_gap_synth.dat")
    assert r.status == "ok"
    c = r.data["curves"][0]
    assert c["n_points"] == 147                      # 150 - 3 kappa<=0 rows
    assert all(k > 0 for k in c["kappa"])
    assert any("dropped" in w for w in r.warnings)


def test_gap_fixture_emits_none_for_the_wholly_missing_seebeck_array():
    # Emission rule: an all-None optional array is None, never a list of nulls.
    c = _run(FX / "tto_gap_synth.dat").data["curves"][0]
    assert c["seebeck"] is None
    assert c["seebeck_std"] is None


def test_json_is_serialisable_without_nan_on_every_fixture():
    for name in ("tto_synth.dat", "tto_gap_synth.dat", "tto_norho_synth.dat"):
        r = _run(FX / name)
        json.dumps(r.data, allow_nan=False)           # D11: must not raise


def test_determinism_same_file_twice_is_byte_identical():
    a = json.dumps(_run(FX / "tto_synth.dat").data, sort_keys=True)
    b = json.dumps(_run(FX / "tto_synth.dat").data, sort_keys=True)
    assert a == b


def test_real_subset_fixture_runs_end_to_end_without_the_real_file():
    # NO _require_real(): tto_real_subset.dat is COMMITTED. Without this test every
    # real-SHAPE assertion in the slice hides behind _require_real() and silently skips on a
    # machine that lacks the local-only measurement file — which is exactly
    # what committing the subset was supposed to prevent. Header + every 4th body row:
    # 244 points, one cooling curve, 2 surviving error rows (spec §6).
    r = _run(FX / "tto_real_subset.dat")
    assert r.status == "ok"
    assert len(r.data["curves"]) == 1
    c = r.data["curves"][0]
    assert c["direction"] == "down" and c["n_points"] == 244
    assert c["field_oe"] == pytest.approx(0.077, abs=1e-6)
    assert r.data["n_error_rows"] == 2
    json.dumps(r.data, allow_nan=False)


def test_gated_when_kappa_column_absent():
    import pandas as pd
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.drop(columns=["Conductivity (W/K-m)"])
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    assert r.status == "gated"
    assert [g.need for g in r.gate] == ["kappa"]


def test_gated_when_temperature_column_absent():
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.drop(columns=["Sample Temp. (K)"])
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    assert r.status == "gated"
    assert [g.need for g in r.gate] == ["temperature"]


def test_gated_when_every_row_is_filtered_out():
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    df["Conductivity (W/K-m)"] = -1.0
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    assert r.status == "gated"
    assert [g.need for g in r.gate] == ["tto_data"]


def test_small_group_is_dropped_and_logged_not_silently_eaten():
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    df.loc[df.index[:3], "Magnetic Field (Oe)"] = 5000.0   # a 3-point stray group
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    assert r.status == "ok"
    assert any(g["n_points"] == 3 and g["reason"] == "< 5 points"
               for g in r.data["dropped_groups"])


# ---- row-filter / error-count edge behaviours (mutation-pinning, spec §2 steps 1-2) ----
# tto_synth.dat: 180 rows -> a 150-point 0 Oe curve + a 30-point 90 kOe curve; the three
# rows carrying Error (code) 16 are df rows 40-42, all inside the 0 Oe group. Each test
# below perturbs ONE cell of an in-memory copy (no fixture byte is touched) so that a
# single production behaviour is the only thing standing between it and a wrong answer.
_ERR_ROWS = (40, 41, 42)


def _synth_variant():
    """(RawTable-replacer, df copy) for tto_synth.dat — mutate df, then run `_run_df(df)`."""
    return load_dat(str(FX / "tto_synth.dat"))


def _run_df(rt, df):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())


def _zero_field_curve(result):
    return next(c for c in result.data["curves"] if round(c["field_oe"], 3) == 0.0)


def test_error_codes_are_counted_over_surviving_rows_only():
    # Filtering a row out must also remove it from the error tally: counting the codes over
    # ALL rows (i.e. forgetting the [keep] mask) would still report 3 here, and would also
    # mis-align the codes against the filtered arrays.
    rt = _synth_variant()
    df = rt.df.copy()
    df.loc[df.index[_ERR_ROWS[0]], "Conductivity (W/K-m)"] = -1.0
    r = _run_df(rt, df)
    assert r.status == "ok"
    assert r.data["n_error_rows"] == 2                    # 3 minus the dropped error row
    assert _zero_field_curve(r)["n_points"] == 149        # 150 minus that same row


def test_kappa_of_exactly_zero_is_dropped_not_kept():
    # Pins the STRICT `kappa > 0` boundary: with `>= 0` the zeroed row survives.
    rt = _synth_variant()
    df = rt.df.copy()
    df.loc[df.index[100], "Conductivity (W/K-m)"] = 0.0
    r = _run_df(rt, df)
    assert r.status == "ok"
    c = _zero_field_curve(r)
    assert c["n_points"] == 149
    assert all(k > 0.0 for k in c["kappa"])


def test_non_numeric_temperature_row_is_dropped_and_never_reaches_t():
    # Pins the finiteness half of the row filter. A NaN T surviving would not merely inflate
    # the count: `t` is written raw (no _san), so the argsort would be corrupted and a NaN
    # would land in the emitted t array (and break json.dumps(allow_nan=False)).
    rt = _synth_variant()
    df = rt.df.copy()
    df["Sample Temp. (K)"] = df["Sample Temp. (K)"].astype(object)
    df.loc[df.index[101], "Sample Temp. (K)"] = "n/a"
    r = _run_df(rt, df)
    assert r.status == "ok"
    c = _zero_field_curve(r)
    assert c["n_points"] == 149
    t = np.asarray(c["t"], float)
    assert np.all(np.isfinite(t))
    assert np.all(np.diff(t) > 0)                         # sort order still intact
    json.dumps(r.data, allow_nan=False)


def test_blank_error_code_cell_counts_as_zero_not_as_an_error():
    # Pins the np.isfinite(codes) guard: without it a blank/NaN code compares != 0 and would
    # be tallied as a fourth error row.
    rt = _synth_variant()
    df = rt.df.copy()
    df["Error (code)"] = df["Error (code)"].astype(object)
    df.loc[df.index[102], "Error (code)"] = ""            # was 0, and stays "not an error"
    r = _run_df(rt, df)
    assert r.status == "ok"
    assert r.data["n_error_rows"] == 3
    assert _zero_field_curve(r)["n_points"] == 150        # blanking a code drops no row


def test_one_blank_field_cell_never_fragments_or_unlabels_a_field_group():
    # C1 regression. `field` is the SOLE grouping key. Without `np.isfinite(field)` in the D6
    # row filter a NaN field sorts last in _cluster_1d's stable argsort, the gap test
    # `gap > tol` is False for a NaN gap, so the row is absorbed into the LAST cluster and
    # that cluster's median representative becomes NaN — measured: the two clean groups
    # [(0.0, 150), (90000.0, 30)] became [(0.0, 149), (nan, 2), (nan, 29)]. The 90 kOe group
    # then loses its field identity, is split, and (abs(nan) < 50 being False) drops silently
    # out of the |H| < 50 Oe RRR selection, with status still "ok" and no warning. It also
    # puts a non-finite value in the bare-float `TTOCurve.field_oe`, which the D11 sanitiser
    # never walks -> json.dumps(allow_nan=False) raises.
    rt = _synth_variant()
    df = rt.df.copy()
    df["Magnetic Field (Oe)"] = df["Magnetic Field (Oe)"].astype(object)
    hi = [i for i in range(len(df)) if float(df.iloc[i]["Magnetic Field (Oe)"]) > 50.0]
    assert len(hi) == 30
    df.loc[df.index[hi[3]], "Magnetic Field (Oe)"] = ""        # blank cell in the 90 kOe group
    df.loc[df.index[10], "Magnetic Field (Oe)"] = ""           # and one in the 0 Oe group
    r = _run_df(rt, df)
    assert r.status == "ok"
    fields = sorted(c["field_oe"] for c in r.data["curves"])
    assert len(fields) == 2                                    # still TWO groups, not three
    assert all(np.isfinite(f) for f in fields)                 # both keep their identity
    assert fields[0] == pytest.approx(0.0)
    assert fields[1] == pytest.approx(90000.0)
    assert _zero_field_curve(r)["n_points"] == 149             # only the blanked rows dropped
    assert next(c for c in r.data["curves"] if c["field_oe"] > 50.0)["n_points"] == 29
    assert r.data["rrr"] is not None                           # zero-field selection intact
    assert any("2 rows dropped" in w for w in r.warnings)      # counted, not silent
    json.dumps(r.data, allow_nan=False)                        # the D11 contract holds


def test_a_wholly_blank_field_column_gates_instead_of_emitting_a_nan_field():
    # The "column absent -> assume 0 Oe" fallback covers an ABSENT column only. A present but
    # wholly blank column must not produce a curve whose field_oe is NaN.
    rt = _synth_variant()
    df = rt.df.copy()
    df["Magnetic Field (Oe)"] = np.nan
    r = _run_df(rt, df)
    assert r.status == "gated"
    assert r.gate[0].need == "tto_data"


def test_sample_block_carries_the_header_geometry():
    # Runs against the committed anonymized subset, not the local-only source file: this is the
    # only coverage that sample["material"] is wired at all, and after the Phase-2b anonymization
    # only the subset carries the neutral placeholder (the local file keeps its own header).
    # The five geometry INFO lines are byte-identical in both (spec §2b keeps them verbatim).
    s = _run(FX / "tto_real_subset.dat").data["sample"]
    assert s["cross_section"] == pytest.approx(2.8565)
    assert s["vlead_separation"] == pytest.approx(2.5)
    assert s["ilead_separation"] == pytest.approx(2.5)
    assert s["emissivity"] == pytest.approx(0.3)
    assert s["material"] == "anonymized"


def test_real_file_one_cooling_curve_of_976_points_and_six_error_rows(tto_real_path):
    r = _run(tto_real_path)
    assert r.status == "ok"
    d = r.data
    assert len(d["curves"]) == 1
    c = d["curves"][0]
    assert c["direction"] == "down" and c["n_points"] == 976
    assert c["field_oe"] == pytest.approx(0.077, abs=1e-6)
    assert d["n_error_rows"] == 6
    json.dumps(d, allow_nan=False)
