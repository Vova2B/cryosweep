# tests/core/test_hall_sigma_degenerate.py
"""A residual sigma that is float noise is not an uncertainty estimate.

`is_resolved` reads `abs(sigma) < abs(R_H)`, so a residual sigma of exactly 0.0 -- which a
zero-residual fit at n >= 3 produces -- reads as MAXIMALLY resolved: it certifies a carrier
density to every digit of R_H from the absence of scatter, not from its presence. Measured
on `examples/hall_temperature_dependence.dat` channel 1: the relative residual sigma takes
exactly two values across all 23 sigma-bearing points, 0.0 and 1.49e-08, on what is the SAME
noiseless measurement. On real files the minimum relative residual sigma is ~2.7e-2.

The rule is a RELATIVE floor (sigma_slope / |slope|, thickness cancels), never `== 0.0`: an
exact-zero test is itself a float-noise predicate that would decline 19 of those points and
publish the other 4 as certain to eight significant digits.
"""
import json
import pathlib
import numpy as np
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.analyzers import hall as hall_mod
from cryosweep_core.analyzers.hall import _stage_fit, HallAnalyzer
from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
from tests.core.conftest import real_data

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
_REG = build_default_registry()


# ---- _stage_fit (field-sweep probe; also every ladder rung) ----------------------------

def test_exact_line_at_three_points_declines_its_residual_sigma():
    H = np.array([1e4, 2e4, 3e4, 4e4])
    R = 1e-3 + 5e-4 * (H / 1e4)                      # exactly linear: zero residual
    out = _stage_fit(H, R, 1e-4, 1)
    assert out["sigma_zero_dof"] is False, "n = 4 has residual DOF to spare"
    assert out["sigma_degenerate"] is True
    assert out["slope_sigma_ohm_per_T"] is None
    assert out["r_h_sigma"] is None
    assert out["r2"] is not None, "r2 is a different claim and is kept"


def test_scattered_line_keeps_its_residual_sigma():
    H = np.array([1e4, 2e4, 3e4, 4e4])
    R = 1e-3 + 5e-4 * (H / 1e4) + np.array([1e-6, -2e-6, 1.5e-6, -0.5e-6])
    out = _stage_fit(H, R, 1e-4, 1)
    assert out["sigma_degenerate"] is False
    assert out["slope_sigma_ohm_per_T"] is not None and out["slope_sigma_ohm_per_T"] > 0


def test_constant_y_carries_the_reason_not_a_bare_none():
    # slope 0 and sigma 0 (or nan): previously None with no reason attached at all
    H = np.array([1e4, 2e4, 3e4, 4e4])
    out = _stage_fit(H, np.full(4, 1e-3), 1e-4, 1)
    assert out["slope_sigma_ohm_per_T"] is None
    assert out["sigma_degenerate"] is True


def test_zero_dof_and_degenerate_are_distinct_flags():
    out = _stage_fit(np.array([1e4, 2e4]), np.array([1e-3, 2e-3]), 1e-4, 1)
    assert out["sigma_zero_dof"] is True
    assert out["sigma_degenerate"] is False, "n < 3 is a different fact from 'residuals vanished'"


def test_relative_floor_is_not_an_exact_zero_test():
    # a sigma of 1e-9 relative is float noise on a noiseless fit, not a measurement
    H = np.array([1e4, 2e4, 3e4, 4e4, 5e4])
    R = 1e-3 + 5e-4 * (H / 1e4) + np.array([1e-15, -1e-15, 1e-15, -1e-15, 0.0])
    out = _stage_fit(H, R, 1e-4, 1)
    assert out["sigma_degenerate"] is True
    assert out["slope_sigma_ohm_per_T"] is None


# ---- hall_tempdep's inline fit needs the identical rule --------------------------------

def _antisym_curves(mags, noise=0.0):
    T = np.array([5.0])
    curves = {}
    for i, m in enumerate(mags):
        curves[m] = (T, np.array([1e-3 + 5e-4 * (m / 1e4) + noise * (-1) ** i]))
        curves[-m] = (T, np.array([1e-3 + 5e-4 * (-m / 1e4)]))
    return curves


def test_tempdep_exact_line_declines_its_residual_sigma():
    pts, _ = _reconstruct_points(_antisym_curves([1e4, 2e4, 3e4]), thickness_m=5e-5,
                                 geometry_sign=1, min_antisym=1, want_stages=False)
    p = pts[0]
    assert p.antisym_points == 3 and p.sigma_zero_dof is False
    assert p.sigma_degenerate is True
    assert p.slope_sigma_ohm_per_T is None and p.r_h_sigma is None
    assert p.r2 is not None


def test_tempdep_scattered_line_keeps_its_residual_sigma():
    pts, _ = _reconstruct_points(_antisym_curves([1e4, 2e4, 3e4], noise=2e-6),
                                 thickness_m=5e-5, geometry_sign=1, min_antisym=1,
                                 want_stages=False)
    assert pts[0].sigma_degenerate is False
    assert pts[0].r_h_sigma is not None


def test_tempdep_two_points_is_zero_dof_not_degenerate():
    pts, _ = _reconstruct_points(_antisym_curves([1e4, 2e4]), thickness_m=5e-5,
                                 geometry_sign=1, min_antisym=1, want_stages=False)
    assert pts[0].sigma_zero_dof is True and pts[0].sigma_degenerate is False


# ---- the accepted consequence on the shipped noiseless example --------------------------

def _analyze(path, probe, ch=1):
    cfg = RunConfig.load(probe_override=probe)
    cfg.hall.hall_channel = ch
    cfg.hall.thickness_mm = 0.07
    return analyze_file(load_dat(str(path)), cfg, _REG)


def test_noiseless_example_publishes_no_carrier_density():
    # Oracle moved 23 -> 0 (accepted consequence): every one of the 23 antisym points'
    # residual sigma is float noise (0.0 or 1.49e-8 relative) and the file carries no
    # instrument std column, so nothing quantifies the uncertainty -- an unquantified
    # uncertainty is not evidence of a small one.
    res = _analyze(EXAMPLES / "hall_temperature_dependence.dat", "hall_tdep")
    pts = res.data["points"]
    anti = [p for p in pts if p["r_h_method"] == "antisym"]
    assert len(anti) == 23
    assert all(p["sigma_degenerate"] for p in anti)
    assert all(p["r_h_sigma"] is None for p in anti)
    assert all(p["carrier_n"] is None for p in pts)
    assert all("r_h_unresolved" in p["derived_flags"] for p in anti)
    assert res.confidence == 0.0
    assert res.status == "low_confidence"


def test_degenerate_flag_reaches_the_field_sweep_envelope(tmp_path):
    hdr = ("[Header]\nBYAPP, Resistivity\nINFO, degen_synth, SAMPLE\n[Data]\n"
           "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms)\n")
    rows = [f"10.0000,{b:.1f},{1e-3 + 5e-4 * (b / 1e4):.17e}"
            for b in np.arange(-20000.0, 20000.1, 250.0)]
    rows += [f"{t:.4f},20000.0,1.5e-3" for t in np.arange(11.0, 51.0, 2.0)]
    p = tmp_path / "degen.dat"
    p.write_text(hdr + "\n".join(rows) + "\n")
    res = HallAnalyzer().analyze(load_dat(p), RunConfig(hall={"hall_channel": 1,
                                                              "thickness_mm": 0.1}))
    pt = res.data["points"][0]
    assert pt["sigma_degenerate"] is True and pt["sigma_zero_dof"] is False
    assert pt["r_h_sigma"] is None
    assert pt["carrier_n"] is None and "r_h_unresolved" in pt["derived_flags"]


# ---- the floor is not finely tuned -------------------------------------------------------

def _decline_signature(path, probe, ch):
    res = _analyze(path, probe, ch)
    return tuple((p["temperature"], p["carrier_n"] is None, p.get("sigma_degenerate"))
                 for p in res.data.get("points", []))


def test_floor_is_not_finely_tuned(monkeypatch):
    """Any floor in [1e-7, 1e-5] gives an identical decline set on every file: the float-
    noise population (<= 1.49e-8 relative) and the real-data population (>= 2.7e-2) leave a
    gap more than three orders wide on each side of 1e-6. Same spirit as the VSM 1/chi
    threshold-insensitivity test: if this ever becomes sensitive the two populations have
    narrowed and the rule needs rethinking, not retuning."""
    files = [EXAMPLES / "hall_temperature_dependence.dat", EXAMPLES / "hall_mixed_sweeps.dat",
             EXAMPLES / "hall_field_sweeps.dat", FIX / "hall_synth.dat",
             FIX / "hall_tdep_synth.dat", FIX / "hall_tdep_std_synth.dat"]
    real = real_data("hall")
    if real is not None:
        files.append(real)
    assert hall_mod.SIGMA_REL_FLOOR == 1e-6
    sigs = {}
    for floor in (1e-7, 1e-6, 1e-5):
        monkeypatch.setattr(hall_mod, "SIGMA_REL_FLOOR", floor)
        sigs[floor] = {(f.name, probe, ch): _decline_signature(f, probe, ch)
                       for f in files for probe in ("hall", "hall_tdep") for ch in (1, 2)}
    assert sigs[1e-7] == sigs[1e-6] == sigs[1e-5]
    # and the rule bites somewhere, so the equality above is not vacuous
    assert any(flag for sig in sigs[1e-6].values() for _t, _d, flag in sig)


# ---- the reason reaches the CSV beside sigma_zero_dof, on both probes ---------------------

def test_sigma_degenerate_reaches_both_csvs(tmp_path):
    import csv
    from cryosweep_core.io.export import export_result
    res = _analyze(EXAMPLES / "hall_temperature_dependence.dat", "hall_tdep")
    outs = export_result(res, tmp_path / "tdep")
    with open(outs["points"]) as f:
        rows = [r for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))]
    assert "sigma_degenerate" in rows[0]
    anti = [r for r in rows if r["r_h_method"] == "antisym"]
    assert anti and all(r["sigma_degenerate"] == "True" for r in anti)
    assert all(r["r_h_sigma (m^3/C)"] == "" for r in anti)
    res = _analyze(EXAMPLES / "hall_field_sweeps.dat", "hall")
    outs = export_result(res, tmp_path / "fs")
    with open(outs["points"]) as f:
        rows = [r for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))]
    assert "sigma_degenerate" in rows[0]
