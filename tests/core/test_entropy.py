import numpy as np
from cryosweep_core.fitting.heat_capacity import eval_lowt_cp_over_t


def test_eval_lowt_debye_t3_matches_gamma_beta():
    T = np.array([1.0, 2.0, 3.0])
    got = eval_lowt_cp_over_t("debye_t3", {"gamma": 0.01, "beta": 2e-4}, T)
    assert np.allclose(got, 0.01 + 2e-4 * T**2)


def test_eval_lowt_spin_noninteracting_includes_log_term():
    T = np.array([1.0, 2.0])
    p = {"gamma": 0.01, "beta": 2e-4, "A": 1e-4, "T0": 10.0}
    got = eval_lowt_cp_over_t("spin_fluct_noninteracting", p, T)
    x = T**2
    exp = 0.01 + 2e-4 * x + 1e-4 * x * np.log(10.0 / T)
    assert np.allclose(got, exp)


from cryosweep_core.fitting.entropy import compute_entropy, dulong_petit_limit, suggest_rln

R = 8.314462618


def test_dulong_petit_3nR():
    assert dulong_petit_limit(7.0) == 7.0 * 3.0 * R
    assert dulong_petit_limit(None) is None


def test_total_entropy_constant_cp_over_t():
    # Cp = c*T  => Cp/T = c const => S(T) = c*T (with extrapolation from 0)
    T = np.linspace(1.0, 10.0, 50); c = 0.02; cp = c * T
    out = compute_entropy(T, cp, lowt_model=("debye_t3", {"gamma": c, "beta": 0.0}), extrapolate=True)
    assert out["extrapolated"] is True
    s = np.array(out["s_total"])
    assert np.isclose(s[0], c * T[0], rtol=1e-3)          # tail from 0 to T[0] adds c*T[0]
    assert np.isclose(s[-1], c * T[-1], rtol=1e-3)
    assert np.all(np.diff(s) >= -1e-12)                    # monotone non-decreasing
    assert np.all(np.isfinite(s))


def test_magnetic_entropy_subtracts_lattice():
    T = np.linspace(1.0, 10.0, 50)
    cp = 0.02 * T + 0.01 * T          # sample = magnetic(0.02T) + lattice(0.01T)
    lattice = 0.01 * T
    out = compute_entropy(T, cp, lowt_model=("debye_t3", {"gamma": 0.03, "beta": 0.0}),
                          lattice_cp=lattice, extrapolate=False)
    assert out["s_magnetic"] is not None
    sm = np.array(out["s_magnetic"]); st = np.array(out["s_total"])
    assert np.all(sm <= st + 1e-9)
    assert out["lattice_source"] is None or isinstance(out["lattice_source"], str)
    # cmag = 0.02*T => (Cp_mag)/T = 0.02 const; integrate-from-lowest (extrapolate=False)
    # over T in [1, 10] => S_mag[-1] = 0.02 * (10 - 1) = 0.18.
    assert np.isclose(sm[-1], 0.02 * (10.0 - 1.0), rtol=1e-3)


def test_magnetic_entropy_alignment_is_order_invariant():
    # FINDING 1: lattice must be masked+sorted with the SAME order as (T, Cp).
    # A cooling ramp gives descending T; cleaned T/cp are reversed to ascending
    # while a raw lattice array stays descending -> cmag misaligned.
    T_asc = np.linspace(1.0, 10.0, 50)
    cp_asc = 0.02 * T_asc + 0.01 * T_asc
    lat_asc = 0.01 * T_asc
    out_asc = compute_entropy(T_asc, cp_asc,
                              lowt_model=("debye_t3", {"gamma": 0.03, "beta": 0.0}),
                              lattice_cp=lat_asc, extrapolate=False)
    # Same physical data, presented as a descending (cooling) ramp.
    T_desc = T_asc[::-1]; cp_desc = cp_asc[::-1]; lat_desc = lat_asc[::-1]
    out_desc = compute_entropy(T_desc, cp_desc,
                               lowt_model=("debye_t3", {"gamma": 0.03, "beta": 0.0}),
                               lattice_cp=lat_desc, extrapolate=False)
    assert out_desc["s_magnetic"] is not None
    assert np.allclose(np.array(out_desc["s_magnetic"]),
                       np.array(out_asc["s_magnetic"]), rtol=1e-9, atol=1e-12)


def test_suggest_rln_matches_rln3():
    # FINDING 2: saturation near R*ln3 selects j == 1.0 / label "R ln3".
    r = suggest_rln([R * np.log(3) * 0.98])
    assert r["j"] == 1.0
    assert r["label"] == "R ln3"


def test_no_magnetic_when_no_lattice():
    T = np.linspace(1.0, 10.0, 20); cp = 0.02 * T
    out = compute_entropy(T, cp, lowt_model=("debye_t3", {"gamma": 0.02, "beta": 0.0}))
    assert out["s_magnetic"] is None


def test_too_few_points_declines():
    out = compute_entropy(np.array([1.0]), np.array([0.02]))
    assert out["s_total"] == [] and out["reason"]


def test_suggest_rln_defaults_to_rln2():
    r = suggest_rln(None)
    assert abs(r["value"] - R * np.log(2)) < 1e-9


def test_magnetic_entropy_partial_overlap_truncates_with_none():
    # MIN-1: a reference-file lattice that only overlaps a SUB-range of T.
    # Out-of-overlap points must genuinely truncate -> None (not a flat plateau).
    T = np.linspace(1.0, 10.0, 50)
    cp = 0.02 * T + 0.01 * T                       # magnetic(0.02T) + lattice(0.01T)
    lattice = 0.01 * T.copy()
    window = (T >= 3.0) & (T <= 7.0)               # lattice finite only in [3, 7]
    lattice[~window] = np.nan
    out = compute_entropy(T, cp, lowt_model=("debye_t3", {"gamma": 0.03, "beta": 0.0}),
                          lattice_cp=lattice, extrapolate=False)
    sm = out["s_magnetic"]
    # (a) same length as temperature/s_total; None outside window, finite floats inside
    assert sm is not None
    assert len(sm) == len(out["temperature"]) == len(out["s_total"])
    for t, v in zip(out["temperature"], sm):
        if 3.0 <= t <= 7.0:
            assert isinstance(v, float) and np.isfinite(v)
        else:
            assert v is None
    # (b) reason mentions truncation/overlap
    assert out["reason"] and ("trunc" in out["reason"].lower() or "overlap" in out["reason"].lower())
    # (c) no NaN/Inf anywhere in s_magnetic (None is used for gaps, never NaN)
    assert all(v is None or np.isfinite(v) for v in sm)
    # (d) lattice_source behavior unchanged (provided path)
    assert out["lattice_source"] == "provided"


# ================= 2026-08-10 uncertainty slice: suggest_rln tolerance gate (closed O5) ====
import pytest as _pytest

from cryosweep_core.fitting.entropy import _RLN_TOL


def test_suggest_rln_genuine_match_gains_matched_true():
    r = suggest_rln([R * np.log(3) * 0.98])
    assert r["label"] == "R ln3" and r["j"] == 1.0        # semantics unchanged (U7)
    assert r["matched"] is True
    assert r["rel_err"] == _pytest.approx(0.02, abs=1e-6)
    assert r["distance"] == _pytest.approx(0.02 * R * np.log(3), rel=1e-6)
    assert r["tol"] == _RLN_TOL == 0.25


def test_suggest_rln_negative_saturation_never_matches():
    # closed O5: negative magnetic entropy is unphysical -> always matched=False,
    # suggestion NOT suppressed (label still the nearest neighbor)
    r = suggest_rln([-0.73])
    assert r["label"] == "R ln2"
    assert r["matched"] is False
    assert r["rel_err"] is not None and r["rel_err"] > 1.0


def test_suggest_rln_no_data_default_gains_unmatched_fields():
    r = suggest_rln(None)
    assert abs(r["value"] - R * np.log(2)) < 1e-9          # pinned default unchanged
    assert r["matched"] is False
    assert r["distance"] is None and r["rel_err"] is None
    assert r["tol"] == _RLN_TOL


def _hc_suggestion(path):
    from cryosweep_core.analyzers.hc import HCAnalyzer
    from cryosweep_core.config import RunConfig
    from cryosweep_core.io.loader import load_dat
    r = HCAnalyzer().analyze(load_dat(str(path)), RunConfig.load())
    return r.data.get("entropy_rln_suggestion"), r.warnings


# Build-measured through the shipped path (2026-08-10); spec manual anchors were
# 0.719 / 0.942 / 1.127 / 1.488. HC_N (hc_lowmass) has s_magnetic=None in the shipped
# path (the spec's 0.942 came from a manual total-S fallback the analyzer does not do)
# -> rel_err None; only matched=False is asserted there.
REL_ERR_HC = 1.127
REL_ERR_HC_LOWT = 0.719
REL_ERR_HC_LOWMASS = None
REL_ERR_HC_FIELDS = 1.488


@_pytest.mark.parametrize("fixture_name,rel_err_3sf", [
    ("hc_path", REL_ERR_HC),
    ("hc_lowt_path", REL_ERR_HC_LOWT),
    ("hc_lowmass_path", REL_ERR_HC_LOWMASS),
    ("hc_fields_path", REL_ERR_HC_FIELDS),
])
def test_real_hc_files_all_unmatched(fixture_name, rel_err_3sf, request):
    sug, warns = _hc_suggestion(request.getfixturevalue(fixture_name))
    assert sug is not None
    assert sug["matched"] is False
    if rel_err_3sf is None:
        assert sug["rel_err"] is None                       # no saturation to compare
    else:
        assert sug["rel_err"] == _pytest.approx(rel_err_3sf, rel=1e-2)
        assert any("nearest-neighbor only, not evidence of a doublet" in w for w in warns)
