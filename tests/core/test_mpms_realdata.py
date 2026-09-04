import dataclasses
import pathlib
import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry

# Molar mass computed from the sample's formula; sample mass PROVISIONAL (12 mg).
_MOLAR, _MASS_MG = 683.22, 12.0

def test_real_mpms_vsm(mpms_real_path):
    if not pathlib.Path(mpms_real_path).exists():
        pytest.skip("real MPMS file not present (gitignored)")
    rt = load_dat(str(mpms_real_path))
    # loader lock: bare CSV must keep ALL rows (a magic-27 regression -> ~447). Assert rows, not columns
    # (the real header row has a trailing comma -> pandas adds a harmless 'Unnamed' column).
    assert rt.header.bare_csv is True
    assert len(rt.df) == 474
    df, _ = canonicalize_columns(rt.df, rt.header)
    score, key = detect_probe(rt.header, set(df.columns), build_default_registry())
    assert key == "vsm" and score >= 0.5

    # WITHOUT masses -> gated on molar_mass/sample_mass (reaches the physics gate)
    gated = analyze_file(rt, RunConfig.load(unit_system="CGS"), build_default_registry())
    assert gated.status == "gated"

    # WITH masses patched onto the header -> ok, with robust (mass-independent) bands
    rt2 = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=_MOLAR, mass_mg=_MASS_MG))
    res = analyze_file(rt2, RunConfig.load(unit_system="CGS"), build_default_registry())
    # Closed O1 (2026-08-10 uncertainty spec §1.3): the CW fit on this file is window-
    # sensitive (theta spread 12.7 K vs sigma 0.99 K), so the confidence penalty flips the
    # certificate ok -> low_confidence. theta itself is unchanged. Allow-listed edit.
    assert res.status == "low_confidence"
    fit = res.data["fit"]
    # theta and r2 are mass-INDEPENDENT (mass only vertically scales chi); these carry the oracle.
    assert fit["r2"] > 0.99                          # actual ~0.9965 (full-range fit)
    assert -55.0 < fit["params"]["theta"] < -40.0    # actual ~ -50.3 K
    # mu_eff is mass-PROVISIONAL (scales with the guessed 12 mg) -> loose band only.
    assert 2.0 < fit["params"]["mu_eff"] < 8.0       # actual ~4.5 mu_B at 12 mg
    # susceptibility finite and positive
    chi = np.array(res.data["chi_molar_cgs"], float)
    assert np.isfinite(chi).all() and (chi > 0).all()
