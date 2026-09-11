# tests/core/test_hall_field_window_ladder.py
"""Every sibling probe re-fits across a ladder of windows and reports the max-min spread
BESIDE sigma, never merged with it. Hall had none: grep -rn "ladder" cryosweep_core/
returned tto/mag/resistivity/fitting and ZERO hits in either Hall analyzer, so the
product could not say whether the fit window moves R_H.

Judged against each RUNG's own sigma, not the full fit's: narrower rungs have fewer
points and larger sigma, and comparing against the full fit's sigma makes ordinary sample
size look like window sensitivity (measured: 4x median that way, under 1x at 10-100 K the
correct way).
"""
import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
import cryosweep_core.analyzers.hall as hall_mod

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, ladder_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")

def _write(tmp_path, curvature, name):
    """curvature=0 -> a straight line, identical R_H in every window.
    curvature>0 -> R_xy = a*B + c*B^3, so narrow windows see a different slope."""
    rows = []
    for b in np.arange(-20000.0, 20000.1, 200.0):
        x = b / 1e4
        rxy = 1e-3 + 5e-4 * x + curvature * x ** 3
        rows.append(f"10.0000,{b:.1f},{rxy:.10e},{1e-3:.10e},{1e-6:.10e}")
    rows += [f"{t:.4f},20000.0,1.5e-3,1e-3,1e-6" for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / name
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p

def _analyze(path):
    return HallAnalyzer().analyze(load_dat(path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))

def test_ladder_has_a_rung_per_window_and_the_full_rung_matches_the_shipped_fit(tmp_path):
    res = _analyze(_write(tmp_path, 0.0, "straight.dat"))
    p = res.data["points"][0]
    fracs = [r["f"] for r in p["r_h_ladder"]]
    assert fracs == [1.0, 0.75, 0.5, 0.25]
    full = next(r for r in p["r_h_ladder"] if r["f"] == 1.0)
    assert full["R_H"] == p["R_H"], "the f=1.0 rung IS the shipped fit"

def test_a_straight_line_is_not_window_sensitive(tmp_path):
    res = _analyze(_write(tmp_path, 0.0, "straight2.dat"))
    p = res.data["points"][0]
    assert "window_sensitive" not in p["derived_flags"]

def test_a_curved_R_xy_is_window_sensitive(tmp_path):
    res = _analyze(_write(tmp_path, 4e-4, "curved.dat"))
    p = res.data["points"][0]
    assert p["r_h_spread"] is not None and p["r_h_spread"] > 0
    assert "window_sensitive" in p["derived_flags"]

def test_spread_is_none_never_zero_when_too_few_rungs_fit(tmp_path):
    """Fewer than two resolved rungs -> None plus ladder_incomplete. 0.0 would read as a
    measured window-stable exponent, which is the opposite claim.

    Brief called for 12 repeats/plateau; MEASURED that this merges the whole 4-plateau
    field block with the trailing temperature ramp (find_blocks' centred activity window
    is 16 rows, so a 12-row plateau bleeds into its neighbour and the "fixed" temperature
    axis then reads as moving over the merged span -- Bug 3a's own reject guard -- so
    `points` comes back EMPTY, not a point with a too-thin ladder). 20 repeats/plateau
    clears that window with margin (measured: 16 already suffices; 20 kept for headroom)
    while the antisymmetrised grid still holds only two distinct |B| (10000, 20000 Oe),
    so no rung reaches _LADDER_MIN_POINTS regardless.
    """
    rows = []
    for b in (-20000.0, -10000.0, 10000.0, 20000.0):
        rows += [f"10.0000,{b:.1f},{1e-3 + 5e-4 * (b / 1e4):.10e},{1e-3:.10e},{1e-6:.10e}"] * 20
    rows += [f"{t:.4f},20000.0,1.5e-3,1e-3,1e-6" for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / "sparse.dat"
    p.write_text(_HDR + "\n".join(rows) + "\n")
    pt = _analyze(p).data["points"][0]
    assert pt["r_h_spread"] is None
    assert "ladder_incomplete" in pt["derived_flags"]


# ---- Controller audit item 3: a rung's sigma must follow the SAME instrument-preferred
# precedence resolved_sigma()/is_resolved() use for the point, not residual sigma alone.
# MEASURED on the real Hall file before choosing: the two rules are NOT equivalent
# (residual excluded zero rungs at every one of 9 temperatures; instrument excluded 1-2).
# This fixture makes the same gap reproducible without the real file: an exactly-linear
# sweep has residual sigma of 0.0 (a perfect fit trivially "resolves" every rung), while a
# declared instrument Std. Dev. large enough to swamp R_H must still decline every rung.
# A residual-only implementation would report this point window-stable; it is the opposite.

_HDR_INSTSIG = ("[Header]\nBYAPP, Resistivity\nINFO, ladder_instsig, SAMPLE\n[Data]\n"
                "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
                "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
                "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")
_INSTSIG_SD = 5e-5   # Ohm-m; ratio (R/rho)=1000 -> instrument sigma_R = SD*1000 = 5e-2 Ohm/row


def _write_instrument_sigma(tmp_path, name, sd=_INSTSIG_SD):
    rows = []
    for b in np.arange(-20000.0, 20000.1, 200.0):
        rxy = 1e-3 + 5e-4 * (b / 1e4)
        rows.append(f"10.0000,{b:.1f},{rxy:.10e},{rxy / 1000:.10e},"
                    f"{sd:.10e},{1e-3:.10e},{1e-6:.10e}")
    rows += [f"{t:.4f},20000.0,1.5e-3,1.5e-6,{sd:.10e},1e-3,1e-6"
             for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / name
    p.write_text(_HDR_INSTSIG + "\n".join(rows) + "\n")
    return p


def test_a_dominant_instrument_sigma_declines_every_rung_even_when_residual_sigma_is_zero(tmp_path):
    res = _analyze(_write_instrument_sigma(tmp_path, "instsig.dat"))
    p = res.data["points"][0]
    assert p["r_h_sigma"] == 0.0                          # perfectly linear: residual is nil
    assert p["r_h_sigma_instrument"] is not None
    assert abs(p["r_h_sigma_instrument"]) > abs(p["R_H"])  # instrument sigma swamps R_H
    assert all(r["unresolved"] for r in p["r_h_ladder"])   # every rung follows suit
    assert p["r_h_spread"] is None
    assert "ladder_incomplete" in p["derived_flags"]


# ---- Fix round 1, Important #2: a spread built from only the two widest rungs must say so.

def test_ladder_thin_fires_when_exactly_two_of_four_rungs_resolve(tmp_path):
    """MEASURED (fix round 1): sweeping the same instrument-sigma fixture's Std. Dev. from
    1e-6 to 2e-5 Ohm-m gives good-rung counts 3, 2, 1, 0, 0, 0... -- 2e-6 is the one value
    in that sweep landing exactly on n_good == 2, the boundary case ladder_thin exists for.
    Distinct from ladder_incomplete (< 2 resolved, no spread at all): here a spread IS
    reported, but only between the two widest windows.
    """
    res = _analyze(_write_instrument_sigma(tmp_path, "instsig_thin.dat", sd=2e-6))
    p = res.data["points"][0]
    good = [r for r in p["r_h_ladder"] if not r["unresolved"]]
    assert len(good) == 2
    assert {r["f"] for r in good} == {1.00, 0.75}          # the two WIDEST windows
    assert p["r_h_spread"] is not None                     # a spread IS reported
    assert "ladder_thin" in p["derived_flags"]
    assert "ladder_incomplete" not in p["derived_flags"]   # not the same claim


# ---- Controller audit item 5: a missing thickness gates R_H itself, so field_sweep_points
# must skip the ladder outright rather than stamp every point ladder_incomplete for a
# missing USER INPUT the thickness gate already names with its own remedy.

def test_ladder_is_skipped_not_incomplete_when_thickness_is_not_supplied(tmp_path):
    res = HallAnalyzer().analyze(load_dat(_write(tmp_path, 0.0, "gated.dat")), RunConfig(
        hall={"hall_channel": 1, "longitudinal_channel": 2}))   # no thickness_mm
    assert res.status == "gated"
    p = res.data["points"][0]
    # Fix round 1, Minor: r_h_ladder is None (never []) when no ladder ran at all, matching
    # power_law_ladder/cw_ladder's sibling convention -- was [] before this fix round.
    assert p["r_h_ladder"] is None
    assert p["r_h_spread"] is None
    assert "ladder_incomplete" not in p["derived_flags"]


# ---- Step 6: pin _LADDER_REL_FLOOR by measurement, then pin its insensitivity ----------

def test_ladder_floor_is_not_finely_tuned(tmp_path):
    """The window_sensitive verdict must not depend on the exact floor constant.

    Measured (see the constant's comment): on the straight and curved synthetic fixtures,
    and on the real Hall file, 3*sigma -- not the floor -- is the term that actually binds
    in every case observed. The verdict is therefore identical across a decade-plus of
    floor values on both fixtures here; if it ever stops being identical, the noise/signal
    separation has narrowed and the rule needs rethinking, not retuning (the 1/chi-guard
    convention, docs/physics-reference.md).
    """
    straight_path = _write(tmp_path, 0.0, "straight4.dat")
    curved_path = _write(tmp_path, 4e-4, "curved4.dat")
    original = hall_mod._LADDER_REL_FLOOR
    verdicts = {"straight": set(), "curved": set()}
    try:
        for floor in (0.005, 0.01, 0.05, 0.1, 0.5):
            hall_mod._LADDER_REL_FLOOR = floor
            verdicts["straight"].add(
                "window_sensitive" in _analyze(straight_path).data["points"][0]["derived_flags"])
            verdicts["curved"].add(
                "window_sensitive" in _analyze(curved_path).data["points"][0]["derived_flags"])
    finally:
        hall_mod._LADDER_REL_FLOOR = original
    assert verdicts["straight"] == {False}, verdicts["straight"]
    assert verdicts["curved"] == {True}, verdicts["curved"]


# ---- Step 7: the real-file outcome, recorded as measured (not as the brief guessed) ----

def test_real_file_ladder_is_quiet(hall_real_path):
    """Spec §4.6 anticipated the flag would be "quiet except possibly at 200/300 K".
    MEASURED (2026-09, reconfirmed fix round 1): on the real Hall file (hall_channel=1,
    thickness=0.07 mm, longitudinal_channel=2 -- the same config
    test_real_file_field_sweep_declines_nothing uses), 3*sigma dominates the spread at all
    9 temperatures -- spread/(3*sig_max) ratios run 0.0035-0.0870, i.e. 3*sigma is ~11x to
    ~290x the spread -- including 200 and 300 K, so window_sensitive fires NOWHERE on this
    sample. That is the informative result for a linear sample, not a sign the flag is dead
    code -- it says the fit window does not move R_H here, which is itself worth being able
    to state. See test_real_file_ladder_thin_flags_exactly_the_three_thin_points for the
    per-temperature detail (three of the nine points reach this quiet verdict from only two
    resolved rungs, not the full four).
    """
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file gitignored/absent")
    res = HallAnalyzer().analyze(load_dat(hall_real_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.07, "longitudinal_channel": 2}))
    pts = res.data["points"]
    assert len(pts) == 9
    for p in pts:
        good = [r for r in p["r_h_ladder"] if not r["unresolved"]]
        assert len(good) >= 2, p["temperature"]           # every point resolves a spread
        assert p["r_h_spread"] is not None
        assert "window_sensitive" not in p["derived_flags"]
        assert "ladder_incomplete" not in p["derived_flags"]


def test_real_file_ladder_thin_flags_exactly_the_three_thin_points(hall_real_path):
    """Fix round 1, Important #2: at 100/200/300 K, instrument sigma excludes BOTH
    narrower rungs, so `good` holds exactly the two widest windows -- the least-different
    pair the ladder can compare, sitting right at the ladder_incomplete floor of two.
    window_sensitive's absence there is real, but it is a verdict over half the window
    range, not the full f=1.00->0.25 span the other six points get. MEASURED (fix round 1):
    good-rung counts are 3,3,3,3,3,3,2,2,2 for T=2,5,10,15,20,50,100,200,300 -- ladder_thin
    must fire on exactly the last three and none of the first six. This is a strong,
    falsifiable oracle: it fails if the flag fires anywhere else, or fails to fire on any
    of the three.
    """
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file gitignored/absent")
    res = HallAnalyzer().analyze(load_dat(hall_real_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.07, "longitudinal_channel": 2}))
    pts = {p["temperature"]: p for p in res.data["points"]}
    assert set(pts) == {2.0, 5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 200.0, 300.0}
    thin_expected = {100.0, 200.0, 300.0}
    for T, p in pts.items():
        good = [r for r in p["r_h_ladder"] if not r["unresolved"]]
        is_thin = "ladder_thin" in p["derived_flags"]
        if T in thin_expected:
            assert len(good) == 2, (T, len(good))
            assert is_thin, T
        else:
            assert len(good) == 3, (T, len(good))
            assert not is_thin, T
        # oracle every fix in this round must preserve, re-measured (fix round 1):
        full = next(r for r in p["r_h_ladder"] if r["f"] == 1.00)
        assert full["R_H"] == p["R_H"], T
        assert full["sigma"] == p["r_h_sigma_instrument"], T
        sig_max = max(abs(r["sigma"]) for r in good)
        spread = p["r_h_spread"]
        ratio = spread / (3.0 * sig_max)
        assert 0.003 <= ratio <= 0.087, (T, ratio)
        assert "window_sensitive" not in p["derived_flags"], T
