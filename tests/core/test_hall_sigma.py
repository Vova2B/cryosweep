"""Hall slope-sigma propagation (2026-08-10 spec §2.1, closed O4) — Task 8; Task 9 appends
hall_tempdep residual + instrument sigma tests."""
import json
import pathlib

import numpy as np
import pytest
from scipy.stats import linregress

from cryosweep_core.analyzers.hall import HallAnalyzer, _stage_fit
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat

FIX = pathlib.Path("tests/core/fixtures")
_OE_PER_T = 1e4


def _analyze_hall(path, **hall):
    rt = load_dat(str(path))
    cfg = RunConfig.load(hall=hall)
    return HallAnalyzer().analyze(rt, cfg)


def test_stage_fit_slope_sigma_matches_linregress():
    rng = np.random.default_rng(42)
    H = np.linspace(-9e4, 9e4, 20)                       # Oe
    R = -5e-4 * (H / _OE_PER_T) + 1e-6 + rng.normal(0.0, 1e-6, H.size)
    d = _stage_fit(H, R, thickness_m=1e-4, geometry_sign=1)
    lr = linregress(H / _OE_PER_T, R)
    assert d["slope_sigma_ohm_per_T"] == pytest.approx(float(lr.stderr), abs=1e-12)
    assert d["r_h_sigma"] == pytest.approx(float(lr.stderr) * 1e-4, rel=1e-12)
    assert "sigma_zero_dof" not in d


def test_stage_fit_two_points_sigma_none_zero_dof():
    H = np.array([-9e4, 9e4])
    R = np.array([4.5e-3, -4.5e-3])
    d = _stage_fit(H, R, thickness_m=1e-4, geometry_sign=1)
    # linregress stderr at n=2 is 0.0 (measured) — 0.0 asserts perfect certainty (U4)
    assert d["slope_sigma_ohm_per_T"] is None
    assert d["r_h_sigma"] is None
    assert d["sigma_zero_dof"] is True
    assert d["n_points"] == 2


def test_hall_synth_points_gain_sigma_keys_json_safe():
    r = _analyze_hall(FIX / "hall_synth.dat", hall_channel=1, thickness_mm=0.1)
    pts = r.data["points"]
    assert pts
    for p in pts:
        assert "slope_sigma_ohm_per_T" in p and "r_h_sigma" in p
        assert "carrier_n_sigma" in p and "mobility_sigma" in p
    # existing pinned oracle untouched: slope -5.000e-4 Ohm/T
    assert pts[0]["slope_ohm_per_T"] == pytest.approx(-5.000e-4, rel=1e-3)
    json.dumps(r.data, allow_nan=False)
    r2 = _analyze_hall(FIX / "hall_synth.dat", hall_channel=1, thickness_mm=0.1)
    assert json.dumps(r.data, sort_keys=True) == json.dumps(r2.data, sort_keys=True)


def test_stage_a_only_point_uses_raw_sigma_and_is_warned():
    """F6 (final-review): the Stage-A-only branch — no antisym stage, so R_H and its sigma
    come from the raw fit. Mutation M14 (dropping `else pt.r_h_sigma_raw` at hall.py:245)
    SURVIVED the full suite because nothing reached this branch; this test kills it.

    `hall_onesided_synth.dat` is positive-field-only, so `_antisymmetrize` returns empty.
    Oracles measured through the shipped path (seed 7, thickness 0.1 mm)."""
    r = _analyze_hall(FIX / "hall_onesided_synth.dat", hall_channel=1, thickness_mm=0.1)
    pts = r.data["points"]
    assert len(pts) == 1
    p = pts[0]
    assert p["antisymmetrized"] is False and p["R_H"] is None
    assert p["r_h_sigma"] is None                      # Stage B never ran
    assert p["r_h_sigma_raw"] == pytest.approx(1.2311e-10, rel=1e-3)
    # #20 (2026-09-02) supersedes the M14 raw-sigma companion pin: with no trusted Stage B
    # R_H the derived quantities are WITHHELD, not derived from Stage A — so the sigma
    # companions are None alongside them, and the decline reason is machine-readable.
    # (M14's intent survives: derived sigmas always track the same stage as the values.)
    assert p["carrier_n"] is None and p["carrier_n_sigma"] is None
    assert p["mobility"] is None
    assert p["derived_flags"] == ["antisym_r_h_missing"]
    # F6 warning-gap kill: the "always-on" >50 % warning must cover Stage A too, and say so
    noisy = [w for w in r.warnings if "treat as noise, not a carrier density" in w]
    assert len(noisy) == 1
    assert "78%" in noisy[0] and "Stage A raw fit scatter" in noisy[0]
    json.dumps(r.data, allow_nan=False)


def test_real_qd_300k_warning_fires_2k_silent(res_path):
    r = _analyze_hall(res_path, hall_channel=2, thickness_mm=0.1)
    noisy = [w for w in r.warnings if "treat as noise, not a carrier density" in w]
    assert any("T = 300.0 K" in w and "154%" in w for w in noisy)
    assert not any("T = 2.0 K" in w for w in noisy)       # 2 K point: rel sigma 0.21 %
    json.dumps(r.data, allow_nan=False)


# ================= Task 9: hall_tempdep residual + instrument sigma (closed O4) ==========
import math

from cryosweep_core.analyzers.hall_tempdep import (HallTempDepAnalyzer, _reconstruct_points,
                                              _interp_fixed_field_sigma_curves)
from cryosweep_core.io.columns import canonicalize_columns

_TH_M = 5e-5      # fixture thickness 0.05 mm
_SD_R = 1e-3      # fixture sigma_R = std(1e-6 Ohm-m) * ratio(1000) per row


def _tdep(path, **hall):
    rt = load_dat(str(path))
    return HallTempDepAnalyzer().analyze(rt, RunConfig(hall=hall))


def test_reconstruct_two_antisym_points_zero_dof_residual_none_instrument_present():
    # hand-built curves: exact line R_asym, 2 antisym pairs -> zero residual DOF (U4)
    Tg = np.arange(2.0, 41.0)
    slope = -6e-4
    curves, sd_curves = {}, {}
    for f_oe in (20000.0, -20000.0, 40000.0, -40000.0):
        B = f_oe / 1e4
        curves[f_oe] = (Tg, slope * B + 1e-3 + np.zeros_like(Tg))
        sd_curves[f_oe] = (Tg, np.full_like(Tg, _SD_R))
    pts, _ = _reconstruct_points(curves, _TH_M, 1, 3, want_stages=False,
                                 sd_curves=sd_curves)
    p = pts[5]
    assert p.antisym_points == 2
    assert p.slope_sigma_ohm_per_T is None and p.r_h_sigma is None   # NEVER 0.0 (U4)
    assert p.sigma_zero_dof is True
    # closed-form instrument sigma: B in {2,4}, mean 3, denom 2, sig_asym = 1e-3/sqrt(2)
    exp = math.sqrt(((2 - 3) / 2.0) ** 2 * (_SD_R ** 2 / 2.0) * 2)
    assert p.slope_sigma_instrument_ohm_per_T == pytest.approx(exp, rel=1e-9)
    assert p.r_h_sigma_instrument == pytest.approx(exp * _TH_M, rel=1e-9)


def test_two_point_fallback_shared_zero_is_correlated_k2_closed_form():
    """F4 (final-review): through-origin y_i = R(B_i) - R(0) share the SAME R(0), so the
    shared-zero term enters as (sum B_i)^2 sigma_0^2, NOT (sum B_i^2) sigma_0^2.

    Hand-built k = 2 case at B = +-2 T with equal per-point sigma: sum B_i = 0, so the
    shared-zero term VANISHES and var = (B1^2 + B2^2) s^2 / (B1^2 + B2^2)^2 = s^2 / 8,
    i.e. slope sigma = s / (2 sqrt 2). The pre-fix independent-sum formula gave s / 2 —
    exactly sqrt(2) too large, which is what it was doing on 121 of the real Hall
    file's 138 points.
    The k = 1 closed form below is UNCHANGED by the fix (the two agree at k = 1), which is
    why the synthetic fixture alone could not catch this."""
    Tg = np.arange(2.0, 41.0)
    curves, sd_curves = {}, {}
    for f_oe, val in ((0.0, 1e-3), (20000.0, 1e-3 - 1.2e-3), (-20000.0, 1e-3 + 1.2e-3)):
        curves[f_oe] = (Tg, np.full_like(Tg, val))
        sd_curves[f_oe] = (Tg, np.full_like(Tg, _SD_R))
    pts, _ = _reconstruct_points(curves, _TH_M, 1, 3, want_stages=False,
                                 two_point_fallback=True, sd_curves=sd_curves)
    p = pts[5]
    # #18 (2026-09-02): the ± pair is now fitted as a single-pair antisym through the
    # origin. Its instrument sigma sqrt(s+² + s-²)/(2B) equals the SAME correlated
    # closed form s/(2 sqrt 2) — the F4 physics is method-independent at ΣB = 0.
    assert p.r_h_method == "antisym"
    assert p.slope_sigma_ohm_per_T is None                   # zero residual DOF (U4)
    assert p.slope_sigma_instrument_ohm_per_T == pytest.approx(
        _SD_R / (2.0 * math.sqrt(2.0)), rel=1e-9)
    assert p.slope_sigma_instrument_ohm_per_T != pytest.approx(_SD_R / 2.0, rel=1e-6)

    # The correlated 2point estimator (F4) is still reachable — via genuinely UNPAIRED
    # fields, where ΣB ≠ 0 and the shared-zero term survives:
    # var = [ΣB_i² s² + (ΣB_i)² s0²]/(ΣB_i²)², B = (2, 4) T -> (20 + 36) s²/400.
    curves2, sd2 = {}, {}
    for f_oe, val in ((0.0, 1e-3), (20000.0, 1e-3 - 1.2e-3), (40000.0, 1e-3 - 2.4e-3)):
        curves2[f_oe] = (Tg, np.full_like(Tg, val))
        sd2[f_oe] = (Tg, np.full_like(Tg, _SD_R))
    q = _reconstruct_points(curves2, _TH_M, 1, 3, want_stages=False,
                            two_point_fallback=True, sd_curves=sd2)[0][5]
    assert q.r_h_method == "2point" and q.antisym_points == 0
    assert q.slope_sigma_instrument_ohm_per_T == pytest.approx(
        _SD_R * math.sqrt(56.0) / 20.0, rel=1e-9)


def test_synth_std_fixture_antisym_and_2point_closed_forms():
    r = _tdep(FIX / "hall_tdep_std_synth.dat", hall_channel=1, thickness_mm=0.05,
              longitudinal_channel=2)
    pts = {p["temperature"]: p for p in r.data["points"]}
    p30 = pts[30.0]                       # 3 antisym pairs at B = 2/4/6 T
    assert p30["antisym_points"] == 3
    # residual sigma: exact-line fit -> finite, tiny (fit noise), NOT None at n >= 3
    assert p30["slope_sigma_ohm_per_T"] is not None
    assert abs(p30["slope_sigma_ohm_per_T"]) < 1e-10
    # instrument sigma closed form: 2.5e-4 Ohm/T (see make_hall_tdep_std.py)
    assert p30["slope_sigma_instrument_ohm_per_T"] == pytest.approx(2.5e-4, rel=1e-6)
    assert p30["r_h_sigma_instrument"] == pytest.approx(1.25e-8, rel=1e-6)
    assert p30["carrier_n_sigma_instrument"] is not None
    assert p30["mobility_sigma_instrument"] is not None
    p50 = pts[50.0]                       # 2-point tail (0 + 20000 Oe)
    assert p50["r_h_method"] == "2point"
    assert p50["slope_sigma_ohm_per_T"] is None            # zero DOF by construction
    assert p50["slope_sigma_instrument_ohm_per_T"] == pytest.approx(
        math.sqrt(2.0) * _SD_R / 2.0, rel=1e-6)
    json.dumps(r.data, allow_nan=False)
    r2 = _tdep(FIX / "hall_tdep_std_synth.dat", hall_channel=1, thickness_mm=0.05,
               longitudinal_channel=2)
    assert json.dumps(r.data, sort_keys=True) == json.dumps(r2.data, sort_keys=True)


def test_sigma_curves_decline_when_ratio_not_constant(tmp_path):
    # hardening 1: a wandering Resistance/Resistivity ratio DECLINES the instrument sigma
    src = (FIX / "hall_tdep_std_synth.dat").read_text().splitlines()
    out = [src[0], src[1], src[2], src[3], src[4]]
    import random
    rnd = random.Random(0)
    for ln in src[5:]:
        parts = ln.split(",")
        parts[3] = f"{float(parts[3]) * (1.0 + 0.01 * rnd.random()):.10e}"  # wobble rho
        out.append(",".join(parts))
    p = tmp_path / "wobble.dat"
    p.write_text("\n".join(out) + "\n")
    rt = load_dat(str(p))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    cfg = RunConfig(hall={"hall_channel": 1})
    assert _interp_fixed_field_sigma_curves(df, cmap, cfg, 1, 1.0) is None
    r = _tdep(p, hall_channel=1, thickness_mm=0.05)
    for pt in r.data["points"]:
        assert pt["slope_sigma_instrument_ohm_per_T"] is None
        assert pt["r_h_sigma_instrument"] is None


def test_no_std_column_all_instrument_fields_none():
    r = _tdep(FIX / "hall_tdep_synth.dat", hall_channel=1, thickness_mm=0.05)
    for p in r.data["points"]:
        assert p["slope_sigma_instrument_ohm_per_T"] is None
        assert p["r_h_sigma_instrument"] is None


def test_real_hall_file_oracles(hall_real_path):
    r = _tdep(hall_real_path, hall_channel=1, thickness_mm=0.07)
    pts = r.data["points"]
    from collections import Counter
    hist = Counter(p["antisym_points"] for p in pts)
    assert hist == {1: 121, 2: 16, 0: 1}                  # pre-existing behavior, now pinned
    assert len(pts) == 138
    # ALL 138 honestly report NO residual sigma — the histogram above shows no point has
    # >= 3 antisym points. (Spec §2.2 said "137/138", an arithmetic slip inconsistent with
    # its own {1:121, 2:16, 0:1} histogram; re-measured through the shipped path.)
    assert sum(1 for p in pts if p["r_h_sigma"] is None) == 138
    # #18: sigma_zero_dof now also covers the 121 single-pair antisym points (1 pair
    # through the origin = zero residual DOF), joining the 16 exactly-two-pair points.
    assert sum(1 for p in pts if p["sigma_zero_dof"]) == 137
    # instrument sigma present on every point with std-column coverage
    covered = [p for p in pts if p["r_h_sigma_instrument"] is not None]
    assert covered and all(p["r_h_sigma_instrument"] > 0 for p in covered)
    # spot value pinned at build (3 s.f.) — see task report
    spot = next(p for p in pts if p["temperature"] == REAL_HALL_SPOT_T)
    assert spot["r_h_sigma_instrument"] == pytest.approx(REAL_HALL_SPOT_RH_SIG_INST, rel=1e-3)
    # hardening 2: the honest >50% rel instrument-sigma warning FIRES on the nV-level
    # Hall channel (median std/rho 61%) — a later fixer must not "repair" this.
    assert any("relative instrument sigma" in w for w in r.warnings)
    json.dumps(r.data, allow_nan=False)
    r2 = _tdep(hall_real_path, hall_channel=1, thickness_mm=0.07)
    assert json.dumps(r.data, sort_keys=True) == json.dumps(r2.data, sort_keys=True)


# Pinned at build through the shipped path (2026-08-10, thickness 0.07 mm): T = 74 K point,
# r_h_sigma_instrument = 1.0617e-11 m^3/C vs R_H = 1.297e-11 — 82 % relative. Median rel
# instrument sigma across the file is 101 % (all 138 points > 50 %): the file's R_H(T) is
# instrument-noise-dominated, which is exactly what the honest sigma now says out loud.
#
# RE-PINNED by final-review F4: the through-origin fallback previously summed the SHARED
# R(0) noise as if the y_i were independent — (sum B_i^2) sigma_0^2 instead of
# (sum B_i)^2 sigma_0^2. 121 of the 138 points use k = 2 fields at exactly +-9 T, where
# sum B_i = 0 and the correct shared-zero coefficient is 0, so the old numbers were ~sqrt(2)
# too large (spot 1.567e-11 -> 1.0617e-11; median rel 139 % -> 101 %). The direction was
# conservative but the estimator was described as "exact" in both the code and the shipped
# physics reference. The verdict is UNCHANGED: all 138 points still exceed 50 %.
REAL_HALL_SPOT_T = 74.0
REAL_HALL_SPOT_RH_SIG_INST = 1.0617e-11
