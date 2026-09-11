# tests/core/test_hall_confidence.py
"""Two coupled defects.

(a) r2 = 1.0 from a two-point fit asserts a perfect fit from a line through two points.
    Measured: the ONLY r2 values hall-tdep reports on the real file are 16 of them, every
    one with antisym_points == 2 and r2 == 1.0 exactly. Same zero-residual-DOF rule that
    already sets sigma to None must set r2 to None. Spec §4.5(a) makes clear this rule is
    not `hall`-specific: it names hall-tdep's own 2-point antisym fit as the file's only
    r2-carrying points, so hall_tempdep.py's inline fit (_reconstruct_points) needs the
    identical fix alongside `_stage_fit`.

(b) hall-tdep's confidence was the fraction of points passing a threshold set to 1
    (config.tdep_min_antisym_points), i.e. 1.0 by construction -- reported beside its own
    warning that every point is instrument noise.
"""
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer, _stage_fit


def test_r2_is_none_when_the_fit_has_zero_residual_dof():
    out = _stage_fit(np.array([1e4, 2e4]), np.array([1e-3, 2e-3]), 1e-4, 1)
    assert out is not None
    assert out["r2"] is None, "a line through two points makes no linearity claim"
    assert out["sigma_zero_dof"] is True

def test_r2_survives_with_three_or_more_points():
    out = _stage_fit(np.array([1e4, 2e4, 3e4]), np.array([1e-3, 2e-3, 3.1e-3]), 1e-4, 1)
    assert out["r2"] is not None

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, conf_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")
_RATIO = 1000.0

def _write(tmp_path, hall_slope, sd_rho, name):
    """Deliberately duplicated from test_hall_unresolved_decline: test helpers in this
    project stay module-local. Do NOT factor them together."""
    rows = []
    for b in np.arange(-20000.0, 20000.1, 250.0):
        rxy = 1e-3 + hall_slope * (b / 1e4)
        rows.append(f"10.0000,{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e},"
                    f"{1e-3:.10e},{1e-6:.10e}")
    rows += [f"{t:.4f},20000.0,1.5e-3,1.5e-6,{sd_rho:.10e},1e-3,1e-6"
             for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / name
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p

def _analyze(path):
    return HallAnalyzer().analyze(load_dat(path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))

def test_confidence_is_capped_by_the_resolved_fraction(tmp_path):
    """Every point unresolved -> confidence 0.0 -> status low_confidence."""
    res = _analyze(_write(tmp_path, 1e-9, 1e-4, "conf.dat"))
    assert res.confidence_parts["resolved"] == 0.0
    assert res.confidence == 0.0
    assert res.status == "low_confidence"

def test_confidence_parts_report_both_ceilings(tmp_path):
    res = _analyze(_write(tmp_path, 5e-4, 1e-12, "conf_ok.dat"))
    assert set(res.confidence_parts) >= {"fit", "resolved"}
    assert res.confidence == min(res.confidence_parts["fit"] or 1.0,
                                 res.confidence_parts["resolved"])


# ---- hall_tempdep's own inline fit needs the same zero-DOF r2 rule ---------------------
# hall.py's _stage_fit is shared with the field-sweep probe, but hall_tempdep.py builds its
# antisym fit inline (_reconstruct_points) and, before this task, set `pt.r2 = fit.r2`
# unconditionally at antisym_points == 2 -- the exact tautology (a) describes. These tests
# exercise that function directly, the same way test_hall_tempdep.py's existing
# `test_reconstructs_exact_known_R_H` (antisym_points == 3, r2 untouched) and
# `test_single_pair_yields_antisym_R_H` (antisym_points == 1, r2 already None) do.

def _antisym_curves(mags):
    """A single-T curve set with a genuinely linear (odd-in-B) R(B), so a real fit at >=3
    points is a real r2 close to 1.0 -- not a tautology -- while >=2 points always hits the
    tautology regardless of how linear the data is. `mags` is a list of |B| (Oe) magnitudes,
    each given both a + and - field."""
    T = np.array([5.0])
    def R(b_oe):
        return 1e-3 + 5e-4 * (b_oe / 1e4)
    curves = {}
    for m in mags:
        curves[m] = (T, np.array([R(m)]))
        curves[-m] = (T, np.array([R(-m)]))
    return curves

def test_tempdep_r2_is_none_at_two_antisym_points():
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    pts, _ = _reconstruct_points(_antisym_curves([10000.0, 20000.0]), thickness_m=5e-5,
                                 geometry_sign=1, min_antisym=1, want_stages=False)
    assert pts and pts[0].antisym_points == 2
    assert pts[0].r2 is None, "a 2-point antisym fit is a tautology, not a fit-quality measurement"
    assert pts[0].sigma_zero_dof is True

def test_tempdep_r2_survives_at_three_antisym_points():
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    pts, _ = _reconstruct_points(_antisym_curves([10000.0, 20000.0, 30000.0]),
                                 thickness_m=5e-5, geometry_sign=1, min_antisym=1,
                                 want_stages=False)
    assert pts and pts[0].antisym_points == 3
    assert pts[0].r2 is not None
    assert pts[0].sigma_zero_dof is False


# ---- hall_tempdep's confidence rebase --------------------------------------------------
# Same formula as `hall`, over `points` (not `fitted`): resolved_fraction's denominator is
# every point, including one with no R_H at all (spec §4.1/§4.5). Built from a file with two
# distinct held-field pairs (+/-10000, +/-20000 Oe) across a T ramp, so every T lands at
# antisym_points == 2 -- r2 stays None everywhere by (a), so fit_quality defaults to 1.0 and
# confidence reduces to resolved_fraction alone. One branch carries tiny (resolved)
# instrument noise, the other huge (unresolved) noise, so the fraction is a real number
# strictly between 0 and 1 -- a formula that only ever lands on 0.0 or 1.0 could not be
# distinguished from the old antisym_fraction-only rule.

_TDEP_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, tdep_conf_synth, SAMPLE\n[Data]\n"
             "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
             "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m)\n")

def _write_tdep_mixed_resolution(tmp_path):
    # Each field is its OWN continuous temperature ramp (hall_tempdep reconstructs from
    # fixed-field T-ramps via the production segmenter, not from per-T field sweeps like
    # the field-sweep helper above) -- mirrors test_hall_tempdep.py's
    # `_write_single_pair_dat`. T < 30 carries tiny (resolved) instrument sigma, T >= 30
    # huge (unresolved) sigma, within the SAME ramp so every field still yields one curve.
    rows = []
    for b, sign in ((10000.0, 1.0), (-10000.0, -1.0), (20000.0, 1.0), (-20000.0, -1.0)):
        rxy = 1e-3 + sign * 5e-4 * (abs(b) / 1e4)
        for i in range(70):
            T = 2.0 + i
            sd_rho = 1e-12 if T < 30.0 else 1e-4
            rows.append(f"{T:.4f},{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e}")
    p = tmp_path / "tdep_conf.dat"
    p.write_text(_TDEP_HDR + "\n".join(rows) + "\n")
    return p

def test_tempdep_confidence_is_capped_by_resolved_fraction(tmp_path):
    from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
    res = HallTempDepAnalyzer().analyze(
        load_dat(_write_tdep_mixed_resolution(tmp_path)),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1}))
    parts = res.confidence_parts
    assert set(parts) >= {"antisym_fraction", "fit", "resolved"}
    pts = res.data["points"]
    assert pts and all(p["antisym_points"] == 2 for p in pts)
    assert all(p["r2"] is None for p in pts), "every point here is a 2-point tautology"
    assert parts["fit"] is None                      # (a) leaves no r2 at all
    n_resolved = sum(1 for p in pts if "r_h_unresolved" not in p["derived_flags"])
    assert 0.0 < n_resolved / len(pts) < 1.0, "fixture must mix resolved and unresolved points"
    assert parts["resolved"] == n_resolved / len(pts)
    assert res.confidence == parts["resolved"]        # fit_quality defaults to 1.0
    assert res.status == ("ok" if res.confidence >= 0.5 else "low_confidence")


# ---- Carried from Task 2's review: the gated (no-thickness) branch must report a real
# `fit` value when one is available, not hardcode None while `r2s` is genuinely populated ---

def test_gated_branch_reports_real_fit_when_r2_is_available(hall_synth_path):
    res = HallAnalyzer().analyze(load_dat(hall_synth_path), RunConfig(hall={"hall_channel": 1}))
    assert res.status == "gated"
    assert res.data["points"], "slope-only points must survive the gate"
    r2s = [p["r2"] for p in res.data["points"] if p.get("r2") is not None]
    assert r2s, "fixture must exercise a real, non-tautological r2 for this test to mean anything"
    assert res.confidence_parts["fit"] == float(np.mean(r2s))
