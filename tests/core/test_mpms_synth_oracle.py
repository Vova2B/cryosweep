import dataclasses
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry

# Ground truth — MUST match tests/core/fixtures/make_mpms.py
_C, _THETA, _MOLAR, _MASS_MG = 1.5, -30.0, 200.0, 10.0
_MU_EFF = 2.827 * (_C ** 0.5)

def test_mpms_synth_bare_csv_loads(mpms_synth_path):
    rt = load_dat(str(mpms_synth_path))
    assert rt.header.bare_csv is True
    assert len(rt.df) == 150                          # row count (not column count)
    assert "Long Moment (emu)" in rt.df.columns

def test_mpms_synth_detects_vsm(mpms_synth_path):
    rt = load_dat(str(mpms_synth_path))
    df, _ = canonicalize_columns(rt.df, rt.header)
    score, key = detect_probe(rt.header, set(df.columns), build_default_registry())   # (score, key)
    assert key == "vsm" and score >= 0.5

def test_mpms_synth_curie_weiss_recovery(mpms_synth_path):
    rt = load_dat(str(mpms_synth_path))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=_MOLAR, mass_mg=_MASS_MG))
    res = analyze_file(rt, RunConfig.load(unit_system="CGS"), build_default_registry())
    assert res.status == "ok"
    fit = res.data["fit"]
    assert fit["params"]["theta"] == pytest.approx(_THETA, rel=1e-6, abs=1e-4)
    assert fit["params"]["C"] == pytest.approx(_C, rel=1e-6)
    assert fit["params"]["mu_eff"] == pytest.approx(_MU_EFF, rel=1e-6)
    assert fit["r2"] > 1 - 1e-9
