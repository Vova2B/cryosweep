import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry

def _bridge(res, ch):
    return next(b for b in res.data["bridges"] if b["channel"] == ch)

def test_act_synth_detects_resistivity(act_synth_path):
    rt = load_dat(str(act_synth_path)); df, _ = canonicalize_columns(rt.df, rt.header)
    score, key = detect_probe(rt.header, set(df.columns), build_default_registry())
    assert key == "resistivity" and score >= 0.5

def test_act_synth_unit_and_rrr(act_synth_path):
    res = analyze_file(load_dat(str(act_synth_path)), RunConfig.load(), build_default_registry())
    assert res.status in ("ok", "low_confidence")          # clean ACT data; status not the oracle
    assert res.data["rho_source"] == "instrument_column"
    b1, b2 = _bridge(res, 1), _bridge(res, 2)
    # unit-correctness (strict): rho in Ohm-cm x1 (a x100 bug -> ~6e-2)
    rho1 = [r for c in b1["rho_t_curves"] for r in c["rho"]]
    assert max(rho1) == pytest.approx(6.0e-4, rel=2e-2)    # rho1(300) = 3e-4 + 1e-6*300
    # RRR (median-of-5 endpoints ~ analytic ~1.95); both channels metallic (rho rises with T)
    assert 1.8 < b1["rrr"] < 2.1 and b1["classification"] == "metallic"
    assert b2["rrr"] > 5.0 and b2["classification"] == "metallic"

def test_qd_absolute_rho_guard_end_to_end(hall_synth_path):
    # Spec §3.4 QD absolute-rho guard (end-to-end through analyze_file, NOT just the
    # _bridge_rho unit test in Task 3). hall_synth.dat is a QD Resistivity-option file
    # whose Bridge 2 Resistivity = 1.0e-6 Ohm-m (constant) -> analyzer rho must be
    # 1e-6 * 100 = 1.0e-4 Ohm-cm. Locks the QD x100 branch against a unit regression
    # (existing real-file tests assert only ratios; one even hardcodes *100 in-test).
    cfg = RunConfig.load().model_copy(update={"probe_override": "resistivity"})
    res = analyze_file(load_dat(str(hall_synth_path)), cfg, build_default_registry())
    b2 = _bridge(res, 2)
    rho2 = [r for c in b2["rho_h_curves"] for r in c["rho"]]   # field-sweep file -> rho_h_curves
    assert max(rho2) == pytest.approx(1.0e-4, rel=1e-3)        # 1e-6 Ohm-m x100; a x1 regression -> 1e-6
