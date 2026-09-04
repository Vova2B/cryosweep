"""TTO fixture integrity (spec §6). These pin the generator's oracles at the FILE level so a
regenerated fixture that silently changed physics fails here, not three tasks later."""
import pathlib

import numpy as np
import pandas as pd
import pytest

from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.io.loader import load_dat

FX = pathlib.Path("tests/core/fixtures")
L0 = 2.443e-8      # no REAL constant here: every fixture this file checks is committed


def _load(name):
    rt = load_dat(str(FX / name))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    return rt, df, cmap


def _col(df, cmap, key):
    return pd.to_numeric(df[cmap.logical[key]], errors="coerce").to_numpy(float)


def test_all_four_fixtures_exist():
    for n in ("tto_synth.dat", "tto_gap_synth.dat", "tto_norho_synth.dat",
              "tto_real_subset.dat"):
        assert (FX / n).exists(), n


def test_synth_detects_as_tto_with_full_score():
    rt, df, _ = _load("tto_synth.dat")
    from cryosweep_core.detect.probe import TTODetector
    assert TTODetector.matches(rt.header, set(df.columns)) == 1.0


def test_synth_has_63_columns_matching_the_real_header():
    rt, df, _ = _load("tto_synth.dat")
    assert len(df.columns) == 63
    assert "Seebeck Coef. (µV/K)" in df.columns
    assert "Figure of Merit ZT" in df.columns


def test_synth_row_count_and_two_field_groups():
    _, df, cmap = _load("tto_synth.dat")
    assert len(df) == 180                      # 150 + 30
    fields = _col(df, cmap, "field")
    assert sorted(set(np.round(fields, 4))) == [0.0, 90000.0]
    # The T grid itself is an oracle input: RRR (Task 5) is read off the 2 K-spaced
    # 2-300 K zero-field ramp, so a moved endpoint or a changed point count silently
    # moves RRR. Pin span AND spacing here, not just the row count.
    T = _col(df, cmap, "temperature")
    assert T.min() == pytest.approx(2.0)
    assert T.max() == pytest.approx(300.0)
    t_zero = np.unique(np.round(T[np.abs(fields) < 50], 9))
    assert len(t_zero) == 150
    assert np.allclose(np.diff(t_zero), 2.0)


def test_synth_kappa_ph_is_exactly_one_at_write_precision():
    _, df, cmap = _load("tto_synth.dat")
    T = _col(df, cmap, "temperature")
    k = _col(df, cmap, "kappa")
    rho = _col(df, cmap, "rho_tto")
    kappa_ph = k - L0 * T / rho
    assert np.max(np.abs(kappa_ph - 1.0)) < 1e-6


def test_synth_error_code_rows_are_three():
    _, df, _ = _load("tto_synth.dat")
    codes = pd.to_numeric(df["Error (code)"], errors="coerce").to_numpy(float)
    assert int(np.sum(np.isfinite(codes) & (codes != 0))) == 3


def test_synth_seebeck_and_rho_follow_the_pinned_formulas():
    _, df, cmap = _load("tto_synth.dat")
    T = _col(df, cmap, "temperature")
    # rel=1e-8 tracks the "%.8e" write format: measured round-trip deviation is 2.2e-16
    # (S) and 2.4e-9 (rho), so a one-digit format regression ("%.7e") surfaces here.
    assert _col(df, cmap, "seebeck") == pytest.approx(0.01 * T, rel=1e-8)
    assert _col(df, cmap, "rho_tto") == pytest.approx(1e-8 * (1 + 9 * T / 300), rel=1e-8)


def test_synth_rrr_is_494_over_59():
    """The Task 5 RRR oracle, read off the FILE under the repo's median-of-5-nearest-each-
    T-extreme convention. rho(296 K)/rho(6 K) = 9.88e-8 / 1.18e-8 = 494/59 exactly, which
    holds only for the 2-300 K / 2 K-spaced zero-field grid — so this fails if the grid moves.
    """
    _, df, cmap = _load("tto_synth.dat")
    fields = _col(df, cmap, "field")
    T = _col(df, cmap, "temperature")
    rho = _col(df, cmap, "rho_tto")
    m = (np.abs(fields) < 50) & np.isfinite(T) & np.isfinite(rho) & (rho > 0)
    T, rho = T[m], rho[m]
    order = np.argsort(T)
    T, rho = T[order], rho[order]
    rho_low = float(np.median(rho[:5]))          # 5 points nearest T_min
    rho_high = float(np.median(rho[-5:]))        # 5 points nearest T_max
    assert rho_high / rho_low == pytest.approx(494 / 59, rel=1e-12)


def test_generator_reproduces_the_committed_fixtures(tmp_path):
    """The oracle dict is a named "Produces" interface and the .dat files are committed
    build products: regenerating into a temp dir must give back BOTH, byte for byte. This
    catches an edited generator that was never regenerated (and vice versa)."""
    import tests.core.fixtures.make_tto as mk

    assert mk.write_all(tmp_path) == {
        "rrr_synth": 494 / 59, "kappa_ph_synth": 1.0, "n_error_rows_synth": 3,
        "pf_at_300k_synth": 9e-5, "n_groups_synth": 2,
    }
    for n in ("tto_synth.dat", "tto_gap_synth.dat", "tto_norho_synth.dat",
              "tto_powerlaw_synth.dat", "tto_deltat_synth.dat"):
        assert (tmp_path / n).read_bytes() == (FX / n).read_bytes(), n


def test_gap_fixture_has_no_seebeck_and_three_nonpositive_kappa_rows():
    _, df, cmap = _load("tto_gap_synth.dat")
    assert not np.isfinite(_col(df, cmap, "seebeck")).any()
    k = _col(df, cmap, "kappa")
    assert int(np.sum(np.isfinite(k) & (k <= 0))) == 3


def test_norho_fixture_has_no_resistivity_and_no_zt():
    _, df, cmap = _load("tto_norho_synth.dat")
    assert not np.isfinite(_col(df, cmap, "rho_tto")).any()
    assert not np.isfinite(_col(df, cmap, "zt")).any()
    assert np.isfinite(_col(df, cmap, "kappa")).all()


def test_real_subset_keeps_two_error_rows_and_every_fourth_row():
    # NO skip guard: tto_real_subset.dat is COMMITTED, so this runs everywhere. The guard
    # would only re-introduce the coverage hole the committed subset exists to close.
    _, df, _ = _load("tto_real_subset.dat")
    assert len(df) == 244                       # ceil(976 / 4)
    codes = pd.to_numeric(df["Error (code)"], errors="coerce").to_numpy(float)
    assert int(np.sum(np.isfinite(codes) & (codes != 0))) == 2


def test_powerlaw_fixture_exists_and_carries_the_pinned_grid():
    # The grid IS the oracle input: n = 3.000000 at every rung is only reproducible on
    # linspace(30, 2, 150) with rho constant at 1e-5 Ohm*m.
    _, df, cmap = _load("tto_powerlaw_synth.dat")
    assert len(df) == 150
    T = _col(df, cmap, "temperature")
    assert T.max() == pytest.approx(30.0)
    assert T.min() == pytest.approx(2.0)
    assert int(np.count_nonzero(T <= 10.0)) == 43
    fields = _col(df, cmap, "field")
    assert np.allclose(fields, 0.0)
    rho = _col(df, cmap, "rho_tto")
    assert np.allclose(rho, 1e-5, rtol=1e-9)


def test_powerlaw_fixture_kappa_ph_is_an_exact_cube_and_kappa_e_is_not_dominant():
    _, df, cmap = _load("tto_powerlaw_synth.dat")
    T = _col(df, cmap, "temperature")
    kappa = _col(df, cmap, "kappa")
    rho = _col(df, cmap, "rho_tto")
    kappa_e = L0 * T / rho
    kappa_ph = kappa - kappa_e
    assert np.allclose(kappa_ph, 1.0e-3 * T ** 3, rtol=1e-7)
    # M6: at rho = 1e-7 this median is 0.874 and kappa_e_dominant would fire on a fixture
    # whose whole point is that NO quality flag fires.
    w = T <= 10.0
    assert float(np.median(kappa_e[w] / kappa[w])) == pytest.approx(0.0646, abs=5e-4)
    assert float(np.median(kappa_e[w] / kappa[w])) < 0.5


def test_powerlaw_fixture_carries_no_error_codes_and_no_seebeck_or_zt():
    _, df, _ = _load("tto_powerlaw_synth.dat")
    codes = pd.to_numeric(df["Error (code)"], errors="coerce").to_numpy(float)
    assert int(np.count_nonzero(np.isfinite(codes) & (codes != 0))) == 0
    for col in ("Seebeck Coef. (µV/K)", "Figure of Merit ZT"):
        v = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        assert not np.isfinite(v).any(), col


def test_deltat_fixture_populates_delta_temp_and_drops_three_rows_before_the_big_one():
    # I3: this is the ONLY fixture with a populated `Delta Temp. (K)` column. The three
    # kappa <= 0 rows sit BEFORE the single oversized DeltaT, which is what makes a `[keep]`-
    # less read of that column report a different temperature (M4).
    _, df, cmap = _load("tto_deltat_synth.dat")
    assert len(df) == 40
    T = _col(df, cmap, "temperature")
    assert T.max() == pytest.approx(30.0) and T.min() == pytest.approx(2.0)
    kappa = _col(df, cmap, "kappa")
    assert [int(i) for i in np.flatnonzero(kappa <= 0)] == [5, 6, 7]
    dt = pd.to_numeric(df["Delta Temp. (K)"], errors="coerce").to_numpy(float)
    assert np.isfinite(dt).all()
    assert int(np.count_nonzero(dt > 0.5)) == 1
    assert int(np.flatnonzero(dt > 0.5)[0]) == 30
    assert T[30] == pytest.approx(8.4615, abs=1e-4)     # the pinned aligned oracle
    # Pin the MAGNITUDES too, not just "one value > 0.5 at index 30". Task 4's warning string
    # ("1 rows have dT/T > 5% (max 10.64% at 8.462 K)") is derived from these numbers, and
    # three regenerating mutations were shown to pass the whole suite while silently moving
    # it: dT 0.9 -> 0.6 makes it 7.09%, and baseline 0.01 -> 0.2 makes it "4 rows". A number
    # measured only in a report is not a gate.
    assert dt[30] == pytest.approx(0.9)
    assert np.allclose(np.delete(dt, 30), 0.01)
    keep = np.isfinite(T) & np.isfinite(kappa) & (kappa > 0)
    ratio = 100.0 * dt[keep] / T[keep]
    assert int(np.count_nonzero(ratio > 5.0)) == 1                  # the "1 rows"
    assert float(ratio.max()) == pytest.approx(10.6364, abs=1e-3)   # the "max 10.64%"


def test_real_subset_header_carries_no_identity():
    """The committed subset ships publicly: its header must be neutral while the load-bearing
    BYAPP token and the five geometry INFO lines survive verbatim (spec §2b)."""
    text = (FX / "tto_real_subset.dat").read_text(encoding="latin-1")
    head = text.split("[Data]")[0]
    assert "TITLE,tto_real_subset.dat" in head
    assert "FILEOPENTIME,0.00,09/01/2026,12:00 am" in head
    assert "INFO,anonymized,SAMPLE_MATERIAL" in head
    assert "INFO,anonymized,SAMPLE_COMMENT" in head   # operator free text, neutralised too
    assert "BYAPP,THERMAL_TRANSPORT,1.0,1.1" in head
    for keep in ("INFO,2.5,SAMPLE_VLEAD_SEPARATION", "INFO,2.5,SAMPLE_ILEAD_SEPARATION",
                 "INFO,2.8565,SAMPLE_CROSS_SECTION", "INFO,38.545,SAMPLE_SURFACE_AREA",
                 "INFO,0.3,SAMPLE_EMISSIVITY"):
        assert keep in head, keep
    assert "Quantum Design" in head          # instrument-format provenance, deliberately kept


def _leak_guard():
    """make_tto._assert_no_identity_leak, imported off the fixtures dir (not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("_make_tto", FX / "make_tto.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._assert_no_identity_leak


_SAMPLE_LINE = (lambda ln: ln.split(",")[1] if ln.startswith("INFO,")
                and ln.endswith(",SAMPLE_MATERIAL") else None)
# Fictional sample string. No ".dat" suffix: the identity gate allowlists every .dat
# reference in shippable paths, and an invented filename is (correctly) not on it.
_SRC = ["TITLE,Aa1.11Bb2.22C_1Jan1900", "INFO,Aa1.11Bb2.22C,SAMPLE_MATERIAL"]


def test_identity_leak_guard_passes_a_clean_header():
    """The anonymised header keeps APPNAME/Quantum Design provenance: these must NOT trip it."""
    clean = ["TITLE,tto_real_subset.dat", "; Copyright 2000, Quantum Design, Inc.",
             "INFO,PPMS Thermal Transport Option Version: Release 1.1.5 Build 5,APPNAME",
             "INFO,anonymized,SAMPLE_MATERIAL", "INFO,anonymized,SAMPLE_COMMENT"]
    _leak_guard()(clean, _SRC, _SAMPLE_LINE)          # must not raise


def test_identity_leak_guard_fires_on_an_uncovered_field():
    """The rewrite rules are a fixed list; the guard is the net for a field they do not cover.
    Here the sample string survives in a SAMPLE_NOTE line no rule touches."""
    leaky = ["TITLE,tto_real_subset.dat", "INFO,anonymized,SAMPLE_MATERIAL",
             "INFO,Aa1.11Bb2.22C pellet,SAMPLE_NOTE"]
    with pytest.raises(AssertionError, match="anonymisation leak"):
        _leak_guard()(leaky, _SRC, _SAMPLE_LINE)
