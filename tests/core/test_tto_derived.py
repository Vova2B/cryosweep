"""Per-point derived quantities: WF decomposition, Lorenz ratio, power factor (spec §2 step 5,
D7, D11). The synth fixture is built so kappa_ph == 1.0 exactly, and PF at 300 K is a
hand-computed 9e-5 — asserted against the hand number, never against its own formula."""
import json
import math
import pathlib

import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat

FX = pathlib.Path("tests/core/fixtures")
L0 = 2.443e-8


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def _main_curve(d):
    return next(c for c in d["curves"] if abs(c["field_oe"]) < 50.0)


def test_kappa_ph_is_one_on_the_synth_fixture():
    # ORACLE. Tolerance 1e-6 is pinned to the fixture's %.8e write format: measured
    # round-trip error is 4.8521e-08 on THIS curve (the 150-point 0 Oe sweep) and 2.2526e-07
    # over the whole file including the 30-point 90000 Oe group — see Task 2's write-format
    # rule. Do not loosen either without re-measuring both.
    c = _main_curve(_run(FX / "tto_synth.dat").data)
    kph = np.asarray(c["kappa_ph"], float)
    assert np.max(np.abs(kph - 1.0)) < 1e-6


def test_kappa_e_plus_kappa_ph_reconstructs_kappa():
    # IDENTITY SMOKE TEST, NOT AN ORACLE: kappa_ph is DEFINED as kappa - kappa_e, so this
    # holds for any implementation of those two lines and cannot fail. It guards wiring
    # (arrays aligned, nothing None) only. The oracles are kappa_ph == 1.0 above and
    # PF == 9e-5 below.
    c = _main_curve(_run(FX / "tto_synth.dat").data)
    k = np.asarray(c["kappa"], float)
    assert np.asarray(c["kappa_e"], float) + np.asarray(c["kappa_ph"], float) == \
        pytest.approx(k, rel=1e-9)


def test_power_factor_hand_computed_at_300_kelvin():
    # At T = 300 K the fixture has S = 3 uV/K and rho = 1e-7 Ohm*m:
    #   PF = (3e-6)^2 / 1e-7 = 9e-5 W/(K^2*m)
    c = _main_curve(_run(FX / "tto_synth.dat").data)
    t = np.asarray(c["t"], float)
    i = int(np.argmin(np.abs(t - 300.0)))
    assert t[i] == pytest.approx(300.0)
    assert c["power_factor"][i] == pytest.approx(9e-5, rel=1e-6)


def test_lorenz_ratio_is_dimensionless_and_positive_on_synth():
    c = _main_curve(_run(FX / "tto_synth.dat").data)
    lr = np.asarray(c["lorenz_ratio"], float)
    k = np.asarray(c["kappa"], float)
    kappa_e = np.asarray(c["kappa_e"], float)
    # First assertion is an IDENTITY SMOKE TEST, NOT AN ORACLE: L/L0 := kappa*rho/(L0*T) and
    # kappa_e := L0*T/rho, so kappa/kappa_e is the same expression rearranged and this cannot
    # fail. The second line IS the oracle: the fixture is built with kappa_ph == 1.0 > 0, so
    # L/L0 must exceed 1 everywhere, which a sign or L0 error would break.
    assert lr == pytest.approx(k / kappa_e, rel=1e-9)
    assert np.all(lr > 1.0)                       # kappa_ph = 1.0 > 0 everywhere


def test_derived_arrays_are_none_when_rho_is_absent():
    # Emission rule + D7: no rho -> kappa_e/kappa_ph/L-ratio/PF are all wholly invalid.
    c = _run(FX / "tto_norho_synth.dat").data["curves"][0]
    assert c["rho"] is None
    assert c["kappa_e"] is None
    assert c["kappa_ph"] is None
    assert c["lorenz_ratio"] is None
    assert c["power_factor"] is None


def test_power_factor_is_none_when_seebeck_is_absent_but_wf_survives():
    c = _run(FX / "tto_gap_synth.dat").data["curves"][0]
    assert c["seebeck"] is None
    assert c["power_factor"] is None
    assert c["kappa_e"] is not None               # rho is present -> WF still computable
    assert c["lorenz_ratio"] is not None


def test_partially_invalid_rho_leaves_none_holes_not_nan():
    import dataclasses
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    rt = load_dat(str(FX / "tto_synth.dat"))
    df = rt.df.copy()
    df.loc[df.index[:5], "Resistivity (Ohm-m)"] = np.nan
    r = TTOAnalyzer().analyze(dataclasses.replace(rt, df=df), RunConfig())
    c = _main_curve(r.data)
    holes = [v for v in c["kappa_e"] if v is None]
    assert len(holes) == 5
    assert all(v is None or math.isfinite(v) for v in c["kappa_e"])
    json.dumps(r.data, allow_nan=False)           # D11: no literal NaN token


def test_negative_kappa_ph_is_reported_not_clipped():
    # A synthetic point with L/L0 < 1 must keep its negative kappa_ph (physics integrity:
    # inelastic scattering is a real regime, not a bug to clamp away).
    from cryosweep_core.analyzers.tto import _derive_wf
    ke, kph, lr = _derive_wf([100.0], [1e-4], [1e-3])
    assert lr[0] < 1.0
    assert kph[0] < 0.0
    assert ke[0] == pytest.approx(L0 * 100.0 / 1e-3)


def test_real_file_lorenz_ratio_band_and_kappa_ph_stays_positive(tto_real_path):
    c = _run(tto_real_path).data["curves"][0]
    lr = np.asarray([v for v in c["lorenz_ratio"] if v is not None], float)
    assert lr.min() == pytest.approx(1.874, abs=0.01)
    assert lr.max() == pytest.approx(7.786, abs=0.01)
    kph = np.asarray([v for v in c["kappa_ph"] if v is not None], float)
    assert kph.min() > 0.0                        # phonon-dominated on this file


def test_d7_rejects_zero_rho_and_zero_temperature():
    # The strict inequalities in D7 (rho > 0, T > 0) are not exercised by any fixture or by
    # the real file, so relaxing `>` to `>=` -- or deleting the `T > 0` clause -- used to pass
    # the whole suite. These are not equivalent mutants: they fabricate a finite 0.0 where the
    # spec requires null (rho==0 -> lorenz 0.0; T==0 -> kappa_e 0.0 and kappa_ph == kappa).
    from cryosweep_core.analyzers.tto import _derive_pf, _derive_wf
    ke, kph, lr = _derive_wf([300.0, 300.0, 0.0], [10.0, 10.0, 10.0], [1e-7, 0.0, 1e-7])
    assert np.isnan(ke[1]) and np.isnan(kph[1]) and np.isnan(lr[1])   # rho == 0
    assert np.isnan(ke[2]) and np.isnan(kph[2]) and np.isnan(lr[2])   # T == 0
    assert np.isfinite(ke[0]) and np.isfinite(kph[0]) and np.isfinite(lr[0])
    assert np.isnan(_derive_pf([3.0], [0.0], 1)[0])                   # PF, rho == 0
    assert np.isfinite(_derive_pf([3.0], [1e-7], 1)[0])
