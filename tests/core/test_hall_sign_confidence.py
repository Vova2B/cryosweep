# tests/core/test_hall_sign_confidence.py
"""A sign claim and a reciprocal are not a symmetric error bar.

`is_resolved` applies one 1-sigma bar to three quantities with different noise behaviour.
`carrier_type` is a SIGN claim: at the real Hall file's median relative sigma of 0.905 the
sign has a 13.5 % chance of being wrong, and it was published as a bare string. `carrier_n
= 1/(e|R_H|)` is a RECIPROCAL: the linearized symmetric `carrier_n_sigma` (n * sigma/|R_H|)
understates the true upper excursion, which at rel = 0.905 is 10.5x n0, and the reported
"carrier_n_sigma/n = 0.91" reads as if the interval were +-91 %.

So every published carrier type now carries `carrier_sign_confidence` = Phi(|R_H|/sigma),
and every published carrier density carries the EXACT interval `carrier_n_ci_low` =
1/(e(|R_H|+sigma)), `carrier_n_ci_high` = 1/(e(|R_H|-sigma)) -- a transform, not a
propagation. The sigma is the SAME one the decline judges by (resolved_sigma), so a point
can never be resolved on one sigma and sign-scored on another. No new decline threshold:
`carrier_n_ci_high` diverges exactly as sigma -> |R_H|, so the existing sigma >= |R_H| rule
already IS "the upper bound on n becomes unbounded".
"""
import csv
import math
import pathlib
import numpy as np
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.analyzers.hall import HallAnalyzer, E_CHG
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from tests.core.conftest import real_data

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
_REG = build_default_registry()
_RATIO = 1000.0


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---- one held-T field loop whose instrument sigma sets the relative sigma directly -------

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, signconf_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")


def _write(tmp_path, hall_slope, sd_rho, name, temps=(10.0,)):
    rows = []
    for T in temps:
        for b in np.arange(-20000.0, 20000.1, 250.0):
            rxy = 1e-3 + hall_slope * (b / 1e4)
            rows.append(f"{T:.4f},{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e},"
                        f"{1e-3:.10e},{1e-6:.10e}")
        rows += [f"{t:.4f},20000.0,1.5e-3,1.5e-6,{sd_rho:.10e},1e-3,1e-6"
                 for t in np.arange(T + 1.0, T + 41.0, 2.0)]
    p = tmp_path / name
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p


def _analyze(path):
    return HallAnalyzer().analyze(load_dat(path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 2}))


def _point_with_rel(tmp_path, target_rel, name):
    """Scale the std column so the instrument sigma lands near `target_rel` of |R_H|."""
    # sigma_slope_inst for this grid with a constant per-row sigma_R = sd*RATIO is
    # sigma_R/sqrt(2) * sqrt(sum w_i^2); calibrate once with a probe run
    res = _analyze(_write(tmp_path, 5e-4, 1e-9, "probe_" + name))
    p = res.data["points"][0]
    rel0 = p["r_h_sigma_instrument"] / abs(p["R_H"])
    res = _analyze(_write(tmp_path, 5e-4, 1e-9 * target_rel / rel0, name))
    return res, res.data["points"][0]


def test_sign_confidence_is_the_normal_cdf_of_R_H_over_the_decline_sigma(tmp_path):
    _, p = _point_with_rel(tmp_path, 0.5, "half.dat")
    sig = p["r_h_sigma_instrument"]          # resolved_sigma prefers the instrument family
    assert p["carrier_type"] is not None
    assert p["carrier_sign_confidence"] == pytest.approx(_phi(abs(p["R_H"]) / sig), rel=1e-9)
    assert 0.97 < p["carrier_sign_confidence"] < 0.985      # Phi(2) = 0.9772


def test_carrier_n_interval_is_the_exact_transform_not_a_propagation(tmp_path):
    _, p = _point_with_rel(tmp_path, 0.5, "ci.dat")
    sig = p["r_h_sigma_instrument"]
    lo = 1.0 / (E_CHG * (abs(p["R_H"]) + sig))
    hi = 1.0 / (E_CHG * (abs(p["R_H"]) - sig))
    assert p["carrier_n_ci_low"] == pytest.approx(lo, rel=1e-9)
    assert p["carrier_n_ci_high"] == pytest.approx(hi, rel=1e-9)
    assert p["carrier_n_ci_low"] < p["carrier_n"] < p["carrier_n_ci_high"]
    # the linearized symmetric sigma is UNCHANGED (it is n * sigma/|R_H|) ...
    assert p["carrier_n_sigma_instrument"] == pytest.approx(p["carrier_n"] * sig / abs(p["R_H"]))
    # ... and understates the exact upper bound by exactly 1/(1 - rel^2)
    rel = sig / abs(p["R_H"])
    sym_hi = p["carrier_n"] * (1.0 + rel)
    assert p["carrier_n_ci_high"] / sym_hi == pytest.approx(1.0 / (1.0 - rel ** 2), rel=1e-9)


def test_linearized_flag_fires_above_the_derived_threshold(tmp_path):
    # 1/(1 - rel^2) > 1.1  <=>  rel > sqrt(1 - 1/1.1) = 0.3015
    _, p_lo = _point_with_rel(tmp_path, 0.25, "lo.dat")
    _, p_hi = _point_with_rel(tmp_path, 0.5, "hi.dat")
    assert "carrier_n_sigma_linearized" not in p_lo["derived_flags"]
    assert "carrier_n_sigma_linearized" in p_hi["derived_flags"]
    rel_hi = p_hi["r_h_sigma_instrument"] / abs(p_hi["R_H"])
    assert 1.0 / (1.0 - rel_hi ** 2) > 1.1


def test_declined_point_carries_neither_sign_confidence_nor_interval(tmp_path):
    res = _analyze(_write(tmp_path, 1e-9, 1e-4, "declined.dat"))
    p = res.data["points"][0]
    assert p["carrier_type"] is None and "r_h_unresolved" in p["derived_flags"]
    assert p["carrier_sign_confidence"] is None
    assert p["carrier_n_ci_low"] is None and p["carrier_n_ci_high"] is None


def test_the_decline_boundary_is_unchanged(tmp_path):
    # sigma just below |R_H| still publishes (with a divergent-looking interval and a
    # sign confidence barely above Phi(1) = 0.841); sigma at/above it still declines.
    _, p = _point_with_rel(tmp_path, 0.98, "edge.dat")
    assert p["carrier_type"] is not None
    assert p["carrier_sign_confidence"] == pytest.approx(_phi(1.0 / 0.98), rel=0.02)
    assert p["carrier_n_ci_high"] > 20 * p["carrier_n"]
    res = _analyze(_write(tmp_path, 1e-9, 1e-4, "over.dat"))
    assert res.data["points"][0]["carrier_type"] is None


def test_majority_below_95_percent_sign_confidence_warns_with_the_median(tmp_path):
    res, p = _point_with_rel(tmp_path, 0.8, "warn.dat")     # Phi(1.25) = 0.894
    assert p["carrier_sign_confidence"] < 0.95
    w = [w for w in res.warnings if "sign" in w.lower()]
    assert w, res.warnings
    assert f"{p['carrier_sign_confidence']:.2f}" in w[0] or f"{100 * p['carrier_sign_confidence']:.0f}" in w[0]
    res2, p2 = _point_with_rel(tmp_path, 0.4, "quiet.dat")  # Phi(2.5) = 0.994
    assert p2["carrier_sign_confidence"] > 0.95
    assert not any("sign" in w.lower() for w in res2.warnings)


# ---- the temperature-dependent probe carries the same fields --------------------------

def test_tempdep_points_carry_sign_confidence_and_interval(hall_tdep_std_synth_path):
    res = HallTempDepAnalyzer().analyze(load_dat(hall_tdep_std_synth_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.05}))
    pub = [p for p in res.data["points"] if p["carrier_type"] is not None]
    assert pub
    for p in pub:
        sig = p["r_h_sigma_instrument"]
        assert p["carrier_sign_confidence"] == pytest.approx(_phi(abs(p["R_H"]) / sig), rel=1e-9)
        assert p["carrier_n_ci_low"] == pytest.approx(1.0 / (E_CHG * (abs(p["R_H"]) + sig)), rel=1e-9)
        assert p["carrier_n_ci_high"] == pytest.approx(1.0 / (E_CHG * (abs(p["R_H"]) - sig)), rel=1e-9)


# ---- carrier_type is never again a bare unqualified string ------------------------------

def _every_hall_result():
    files = [EXAMPLES / "hall_field_sweeps.dat", EXAMPLES / "hall_mixed_sweeps.dat",
             EXAMPLES / "hall_temperature_dependence.dat", FIX / "hall_synth.dat",
             FIX / "hall_tdep_synth.dat", FIX / "hall_tdep_std_synth.dat",
             FIX / "hall_long_synth.dat"]
    real = real_data("hall")
    if real is not None:
        files.append(real)
    for f in files:
        for probe in ("hall", "hall_tdep"):
            for ch in (1, 2):
                cfg = RunConfig.load(probe_override=probe)
                cfg.hall.hall_channel = ch
                cfg.hall.thickness_mm = 0.07
                cfg.hall.longitudinal_channel = 2 if ch == 1 else 1
                yield f.name, probe, ch, analyze_file(load_dat(str(f)), cfg, _REG)


def test_carrier_type_is_always_qualified():
    seen = 0
    for name, probe, ch, res in _every_hall_result():
        for p in res.data.get("points", []) or []:
            if p.get("carrier_type") is None:
                assert p.get("carrier_sign_confidence") is None, (name, probe, ch)
                continue
            seen += 1
            assert p["carrier_sign_confidence"] is not None, (name, probe, ch, p["temperature"])
            assert p["carrier_n_ci_low"] is not None and p["carrier_n_ci_high"] is not None
            assert 0.5 < p["carrier_sign_confidence"] <= 1.0
    assert seen > 0, "no published carrier type anywhere -- the check would be vacuous"


def test_real_file_channel_1_reports_the_measured_sign_confidence():
    path = real_data("hall")
    if path is None:
        pytest.skip("local-only measurement file for key 'hall' is not available")
    res = HallTempDepAnalyzer().analyze(load_dat(path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.07}))
    pub = [p for p in res.data["points"] if p["carrier_type"] is not None]
    assert len(pub) == 66
    conf = sorted(p["carrier_sign_confidence"] for p in pub)
    # relative sigma in [0.721, 0.997], median 0.905 -> Phi(1/0.905) = 0.865
    assert conf[len(conf) // 2] == pytest.approx(_phi(1.0 / 0.905), abs=0.01)
    assert all(c < 0.95 for c in conf)
    assert any("sign" in w.lower() and "0.8" in w for w in res.warnings), res.warnings
    assert all("carrier_n_sigma_linearized" in p["derived_flags"] for p in pub)


# ---- the new fields reach both CSVs ---------------------------------------------------

def test_new_fields_reach_both_csvs(tmp_path, hall_tdep_std_synth_path):
    from cryosweep_core.io.export import export_result
    res = HallTempDepAnalyzer().analyze(load_dat(hall_tdep_std_synth_path), RunConfig(
        hall={"hall_channel": 1, "thickness_mm": 0.05}))
    outs = export_result(res, tmp_path / "tdep")
    with open(outs["points"]) as f:
        rows = [r for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))]
    for col in ("carrier_sign_confidence", "carrier_n_ci_low (1/m^3)", "carrier_n_ci_high (1/m^3)"):
        assert col in rows[0], col
    pub = [r for r in rows if r["carrier_type"] != ""]
    assert pub and all(r["carrier_sign_confidence"] != "" for r in pub)
    res = _analyze(_write(tmp_path, 5e-4, 1e-9, "fs.dat"))
    outs = export_result(res, tmp_path / "fs")
    with open(outs["points"]) as f:
        rows = [r for r in csv.DictReader(ln for ln in f if not ln.startswith("#"))]
    assert rows[0]["carrier_sign_confidence"] != ""
    assert rows[0]["carrier_n_ci_high (1/m^3)"] != ""
