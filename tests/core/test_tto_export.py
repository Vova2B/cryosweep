"""TTO CSV export (spec §5): pinned headers, dotted-stem filenames, no NaN cells."""
import csv
import math
import pathlib

import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.export import export_result
from cryosweep_core.io.loader import load_dat

FX = pathlib.Path("tests/core/fixtures")

# Closed O6 (2026-08-10 uncertainty slice): 15 -> 18, the three derived _std columns
# APPENDED last (same precedent as SUMMARY_HEADER's 9 -> 20). Positional readers break;
# name-keyed readers are unaffected.
LONG_HEADER = ["field_oe", "direction", "T", "kappa", "kappa_std", "seebeck", "seebeck_std",
               "rho_ohm_m", "rho_std", "zt", "zt_std", "kappa_e", "kappa_ph",
               "lorenz_ratio", "power_factor", "kappa_e_std", "kappa_ph_std",
               "lorenz_ratio_std"]
SUMMARY_HEADER = ["rrr", "rrr_t_high_k", "rrr_t_low_k", "classification",
                  "pf_at_thigh_w_k2m", "zt_peak", "zt_peak_t_k", "zt_peak_at_edge",
                  "n_error_rows",
                  # --- appended by the integrity slice (I11): name-keyed readers are
                  # unaffected; POSITIONAL readers (Origin templates) break. No shim.
                  "rrr_std", "zt_peak_std",
                  "kappa_ph_n", "kappa_ph_n_sigma", "kappa_ph_n_spread",
                  "kappa_ph_n_loglog", "kappa_ph_n_method_delta",
                  "kappa_ph_b", "kappa_ph_r2", "kappa_ph_window_k_max", "kappa_ph_flags"]


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def _rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_export_returns_exactly_two_csvs(tmp_path):
    out = export_result(_run(FX / "tto_synth.dat"), tmp_path / "tto_synth")
    assert set(out) == {"tto", "tto_summary"}
    assert pathlib.Path(out["tto"]).exists()
    assert pathlib.Path(out["tto_summary"]).exists()
    # acms precedent: no capabilities/meta sidecars for this probe
    assert not (tmp_path / "tto_synth.meta.json").exists()
    assert not (tmp_path / "tto_synth.capabilities.csv").exists()


def test_filenames_survive_a_dotted_stem(tmp_path):
    # with_suffix(".tto.csv") would truncate this dotted stem to "sample.tto.csv" (measured).
    stem = tmp_path / "sample.a1_export"
    out = export_result(_run(FX / "tto_synth.dat"), stem)
    assert pathlib.Path(out["tto"]).name == "sample.a1_export.tto.csv"
    assert pathlib.Path(out["tto_summary"]).name == "sample.a1_export.tto_summary.csv"


def test_long_csv_header_is_pinned_and_complete(tmp_path):
    out = export_result(_run(FX / "tto_synth.dat"), tmp_path / "s")
    with open(out["tto"], newline="") as f:
        assert next(csv.reader(f)) == LONG_HEADER


def test_long_csv_has_one_row_per_curve_point(tmp_path):
    r = _run(FX / "tto_synth.dat")
    out = export_result(r, tmp_path / "s")
    assert len(_rows(out["tto"])) == sum(c["n_points"] for c in r.data["curves"])


def test_long_csv_writes_empty_cells_for_missing_values_never_nan(tmp_path):
    out = export_result(_run(FX / "tto_gap_synth.dat"), tmp_path / "gap")
    rows = _rows(out["tto"])
    assert all(r["seebeck"] == "" for r in rows)
    assert all(r["power_factor"] == "" for r in rows)
    assert all("nan" not in v.lower() for r in rows for v in r.values())
    assert all(r["kappa_e"] != "" for r in rows)


# Column name -> curve key. Only `T` and `rho_ohm_m` are renamed; the rename is exactly
# what makes a silent mis-mapping (e.g. rho_ohm_m <- rho_std) plausible, so pin every cell.
COL_TO_KEY = {"T": "t", "kappa": "kappa", "kappa_std": "kappa_std",
              "seebeck": "seebeck", "seebeck_std": "seebeck_std",
              "rho_ohm_m": "rho", "rho_std": "rho_std", "zt": "zt", "zt_std": "zt_std",
              "kappa_e": "kappa_e", "kappa_ph": "kappa_ph",
              "lorenz_ratio": "lorenz_ratio", "power_factor": "power_factor"}


@pytest.mark.parametrize("fixture", ["tto_synth.dat", "tto_real_subset.dat"])
def test_long_csv_cells_match_the_result(tmp_path, fixture):
    r = _run(FX / fixture)
    out = export_result(r, tmp_path / "cells")
    rows = _rows(out["tto"])
    seen = 0
    for c in r.data["curves"]:
        n = len(c["t"])
        for i in range(n):
            row = rows[seen + i]
            assert float(row["field_oe"]) == pytest.approx(c["field_oe"])
            assert row["direction"] == c["direction"]
            for col, key in COL_TO_KEY.items():
                arr = c.get(key)
                want = arr[i] if arr else None
                if want is None:
                    assert row[col] == "", f"{fixture} {col}[{i}] should be blank"
                else:
                    assert float(row[col]) == pytest.approx(want), f"{fixture} {col}[{i}]"
        seen += n
    assert seen == len(rows)


def test_long_csv_blanks_non_finite_values(tmp_path):
    """The `math.isfinite` guard in `_cell` is the binding never-emit-NaN constraint; the
    analyzer sanitizes upstream, so exercise it directly with a Result-like object."""
    class _R:
        data = {"probe": "tto",
                "curves": [{"field_oe": 0.0, "direction": "warming", "n_points": 3,
                            "t": [2.0, float("nan"), 4.0],
                            "kappa": [1.0, 2.0, float("inf")],
                            "seebeck": [float("-inf"), 1.0, None]}],
                "rrr": {"rrr": float("nan"), "t_high_k": 300.0, "t_low_k": 2.0,
                        "classification": None},
                "summary": {"pf_at_thigh": float("inf"), "zt_peak": None,
                            "zt_peak_t_k": None},
                "n_error_rows": 0}

    out = export_result(_R(), tmp_path / "nonfinite")
    rows = _rows(out["tto"])
    assert len(rows) == 3
    assert rows[1]["T"] == "" and rows[2]["kappa"] == "" and rows[0]["seebeck"] == ""
    assert rows[0]["T"] and rows[0]["kappa"]  # finite values still written
    assert all(v == "" or math.isfinite(float(v))
               for row in rows for k, v in row.items() if k != "direction")
    srow = _rows(out["tto_summary"])[0]
    assert srow["rrr"] == "" and srow["pf_at_thigh_w_k2m"] == ""
    assert all("nan" not in v.lower() and "inf" not in v.lower()
               for row in rows + [srow] for v in row.values())


def test_long_csv_tolerates_a_short_optional_array(tmp_path):
    """A shorter-than-`t` optional array blanks rather than raising IndexError."""
    class _R:
        data = {"probe": "tto",
                "curves": [{"field_oe": 0.0, "direction": "warming", "n_points": 3,
                            "t": [2.0, 3.0, 4.0], "kappa": [1.0, 2.0, 3.0],
                            "zt": [0.1]}],
                "rrr": {}, "summary": {}, "n_error_rows": 0}

    rows = _rows(export_result(_R(), tmp_path / "short")["tto"])
    assert [r_["zt"] for r_ in rows] == ["0.1", "", ""]


def test_summary_csv_header_and_single_row(tmp_path):
    out = export_result(_run(FX / "tto_synth.dat"), tmp_path / "s")
    with open(out["tto_summary"], newline="") as f:
        rdr = csv.reader(f)
        assert next(rdr) == SUMMARY_HEADER
        body = list(rdr)
    assert len(body) == 1


def test_summary_csv_values_match_the_result(tmp_path):
    r = _run(FX / "tto_synth.dat")
    out = export_result(r, tmp_path / "s")
    row = _rows(out["tto_summary"])[0]
    rrr, summary = r.data["rrr"], r.data["summary"]
    # Sanity: the paired values must differ, else a swap mutation would go unnoticed.
    assert rrr["t_high_k"] != rrr["t_low_k"]
    assert summary["pf_at_thigh"] != summary["zt_peak_t_k"]
    assert float(row["rrr"]) == pytest.approx(rrr["rrr"])
    assert float(row["rrr_t_high_k"]) == pytest.approx(rrr["t_high_k"])
    assert float(row["rrr_t_low_k"]) == pytest.approx(rrr["t_low_k"])
    assert row["classification"] == rrr["classification"] == "metallic"
    assert float(row["pf_at_thigh_w_k2m"]) == pytest.approx(summary["pf_at_thigh"])
    assert float(row["zt_peak"]) == pytest.approx(summary["zt_peak"])
    assert float(row["zt_peak_t_k"]) == pytest.approx(summary["zt_peak_t_k"])
    # I3 honesty flag rides into the CSV. On the synth ZT rises monotonically with T, so the
    # reported maximum IS the last measured point.
    assert summary["zt_peak_at_edge"] is True
    assert row["zt_peak_at_edge"] == "True"
    assert int(row["n_error_rows"]) == r.data["n_error_rows"] == 3


def test_summary_csv_writes_blanks_when_rrr_is_absent(tmp_path):
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    df["Magnetic Field (Oe)"] = 90000.0
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    out = export_result(r, tmp_path / "nozf")
    row = _rows(out["tto_summary"])[0]
    assert row["rrr"] == "" and row["classification"] == ""


def test_norho_fixture_exports_blank_optional_columns(tmp_path):
    r = _run(FX / "tto_norho_synth.dat")
    out = export_result(r, tmp_path / "norho")
    rows = _rows(out["tto"])
    assert rows and all(r_["rho_ohm_m"] == "" and r_["zt"] == "" for r_ in rows)
    assert all("nan" not in v.lower() for r_ in rows for v in r_.values())
    srow = _rows(out["tto_summary"])[0]
    assert srow["rrr"] == "" and srow["classification"] == ""


def test_real_subset_fixture_exports_finite_cells_only(tmp_path):
    r = _run(FX / "tto_real_subset.dat")
    out = export_result(r, tmp_path / "subset")
    rows = _rows(out["tto"])
    assert len(rows) == sum(c["n_points"] for c in r.data["curves"])
    for row in rows:
        for v in row.values():
            assert "nan" not in v.lower() and "inf" not in v.lower()
            if v:
                try:
                    assert math.isfinite(float(v))
                except ValueError:
                    pass  # direction is a string label


def test_real_file_export_round_trips(tmp_path, tto_real_path):
    r = _run(tto_real_path)
    out = export_result(r, tmp_path / "sample.a1_export")
    rows = _rows(out["tto"])
    assert len(rows) == 976
    assert all("nan" not in v.lower() and "inf" not in v.lower()
               for row in rows for v in row.values())
    assert float(_rows(out["tto_summary"])[0]["rrr"]) == pytest.approx(1.4555, abs=1e-3)


def test_summary_header_is_exactly_twenty_columns_with_the_original_nine_first(tmp_path):
    out = export_result(_run(FX / "tto_synth.dat"), tmp_path / "s")
    with open(out["tto_summary"], newline="") as f:
        header = next(csv.reader(f))
    assert header == SUMMARY_HEADER
    assert len(header) == 20
    assert header[:9] == ["rrr", "rrr_t_high_k", "rrr_t_low_k", "classification",
                          "pf_at_thigh_w_k2m", "zt_peak", "zt_peak_t_k",
                          "zt_peak_at_edge", "n_error_rows"]


def test_declined_fit_writes_blank_kappa_ph_cells_not_none(tmp_path):
    # tto_synth has only 5 points below 10 K -> the fit declines.
    out = export_result(_run(FX / "tto_synth.dat"), tmp_path / "s")
    row = _rows(out["tto_summary"])[0]
    for k in ("kappa_ph_n", "kappa_ph_n_sigma", "kappa_ph_n_spread", "kappa_ph_n_loglog",
              "kappa_ph_n_method_delta", "kappa_ph_b", "kappa_ph_r2",
              "kappa_ph_window_k_max", "kappa_ph_flags"):
        assert row[k] == "", k
    assert row["rrr_std"] != "" and row["zt_peak_std"] != ""   # these are NOT gated on the fit


def test_fitted_file_writes_every_kappa_ph_cell(tmp_path):
    out = export_result(_run(FX / "tto_powerlaw_synth.dat"), tmp_path / "s")
    row = _rows(out["tto_summary"])[0]
    assert float(row["kappa_ph_n"]) == pytest.approx(3.0, abs=1e-6)
    assert float(row["kappa_ph_b"]) == pytest.approx(1.0e-3, rel=1e-6)
    assert float(row["kappa_ph_r2"]) == pytest.approx(1.0, abs=1e-7)
    assert float(row["kappa_ph_window_k_max"]) == 10.0
    assert float(row["kappa_ph_n_loglog"]) == pytest.approx(3.0, abs=1e-6)
    assert float(row["kappa_ph_n_spread"]) < 0.05
    assert row["kappa_ph_flags"] == ""                 # quality_flags == [] joins to ""


def test_n_sigma_is_the_stat_sigma_and_not_the_window_spread(tmp_path):
    """Review finding: feeding `n_spread` into the `kappa_ph_n_sigma` cell passed the whole core
    suite. On tto_powerlaw_synth (the fixture the neighbouring test uses) sigma = 1.29e-9 and
    spread = 7.37e-9, so even a straight SWAP is invisible there. tto_real_subset separates them
    by 74x -- sigma 0.0097 vs spread 0.7207 -- so it is the fixture this must be pinned on. The
    two are physically different quantities (fit scatter vs window sensitivity); confusing them
    is precisely the over-claim the integrity slice exists to prevent."""
    out = export_result(_run(FX / "tto_real_subset.dat"), tmp_path / "s")
    row = _rows(out["tto_summary"])[0]
    sigma = float(row["kappa_ph_n_sigma"])
    spread = float(row["kappa_ph_n_spread"])
    assert sigma == pytest.approx(0.009695422631348375, rel=1e-9)
    assert spread == pytest.approx(0.7207143396761797, rel=1e-9)
    assert spread > 70 * sigma                          # no swap, no aliasing, no mix-up
    assert float(row["kappa_ph_n"]) == pytest.approx(2.0350774348699807, rel=1e-9)


def test_flags_are_semicolon_joined_and_round_trip_through_a_csv_reader(tmp_path):
    # A ';' needs no escaping: DictWriter's delimiter is ',' and its quotechar is '"'.
    out = export_result(_run(FX / "tto_real_subset.dat"), tmp_path / "s")
    row = _rows(out["tto_summary"])[0]
    # A ',' would split the cell across columns; the flag vocabulary is closed. (Whether this
    # quarter-sampled subset happens to carry a flag is not asserted here — the REAL file's
    # window_sensitive is pinned in tests/core/test_tto_integration.py.)
    assert "," not in row["kappa_ph_flags"]
    parts = [x for x in row["kappa_ph_flags"].split(";") if x]
    assert set(parts) <= {"window_sensitive", "ladder_incomplete", "n_at_bound",
                          "kappa_e_dominant"}


def test_two_flags_are_joined_by_a_semicolon_and_stay_in_one_csv_cell(tmp_path):
    """The separator itself needs a MULTI-flag case. tto_real_subset yields exactly one flag,
    so `assert "," not in ...` above is vacuous there and swapping ';' for ',' in export.py
    passed every test. With ',' the cell would split across columns and shift the row.

    UPDATED (final-review C1): tto_deltat_synth used to be the 2-flag fixture, but its flags
    were `n_at_bound` + `kappa_e_dominant` -- a bound-pinned fit that the analyzer now
    DECLINES, so it exports no flags at all. No committed fixture produces two flags any more
    (a fit good enough to be reported carries at most `window_sensitive` /
    `ladder_incomplete` / `kappa_e_dominant`, and the fixtures produce one at a time), so the
    pair is CONSTRUCTED on a real exported result. That keeps the delimiter assertion
    non-vacuous without a fixture whose only purpose is to be declined."""
    res = _run(FX / "tto_real_subset.dat")
    res.data["kappa_ph_fit"]["quality_flags"] = ["window_sensitive", "kappa_e_dominant"]
    out = export_result(res, tmp_path / "s")
    with open(out["tto_summary"], newline="") as f:
        rows = list(csv.reader(f))
    assert len(rows[1]) == 20                       # not split into 21 by a stray delimiter
    row = _rows(out["tto_summary"])[0]
    assert row["kappa_ph_flags"] == "window_sensitive;kappa_e_dominant"


def test_a_numpy_non_finite_is_blanked_like_a_builtin_float(tmp_path):
    """`isinstance(v, float)` is False for np.float32 (measured), so a numpy NaN leaked into the
    CSV as a literal "nan". Not reachable today -- the analyzer wraps every fit scalar in
    float(...) -- but the never-emit-non-finite constraint is absolute, so the guard is made on
    the numeric TOWER (numbers.Real) rather than on the builtin type."""
    import numpy as np
    assert not isinstance(np.float32("nan"), float)          # the premise, pinned
    res = _run(FX / "tto_real_subset.dat")
    res.data["rrr"]["rrr_std"] = np.float32("nan")
    res.data["summary"]["zt_peak_std"] = np.float64("inf")
    row = _rows(export_result(res, tmp_path / "s")["tto_summary"])[0]
    assert row["rrr_std"] == "" and row["zt_peak_std"] == ""


def test_no_summary_cell_is_ever_nan_or_inf(tmp_path):
    for name in ("tto_synth.dat", "tto_gap_synth.dat", "tto_norho_synth.dat",
                 "tto_real_subset.dat", "tto_powerlaw_synth.dat",
                 "tto_deltat_synth.dat"):
        out = export_result(_run(FX / name), tmp_path / name)
        for k, v in _rows(out["tto_summary"])[0].items():
            # `kappa_ph_flags` is a closed vocabulary of words, not a number, and one of its
            # tokens contains "nan" as a SUBSTRING ("kappa_e_domi-nan-t" — measured on
            # tto_deltat_synth). Its content is pinned by the round-trip test above; the
            # never-emit-non-finite constraint applies to the NUMERIC cells.
            if k in ("kappa_ph_flags", "classification"):
                continue
            assert "nan" not in v.lower() and "inf" not in v.lower(), (name, k, v)
            if v:
                try:
                    assert math.isfinite(float(v)), (name, k, v)
                except ValueError:
                    pass          # bool-ish cells ("True"/"False")


def test_long_csv_header_is_unchanged(tmp_path):
    out = export_result(_run(FX / "tto_synth.dat"), tmp_path / "s")
    with open(out["tto"], newline="") as f:
        assert next(csv.reader(f)) == LONG_HEADER
