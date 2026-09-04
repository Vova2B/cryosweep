"""TTO derived-quantity _std companions + long CSV 15 -> 18 (2026-08-10 spec §5, closed O6)."""
import csv
import json
import pathlib

import numpy as np
import pytest

from cryosweep_core.config import RunConfig
from cryosweep_core.io.export import export_result
from cryosweep_core.io.loader import load_dat

FX = pathlib.Path("tests/core/fixtures")


def _run(path):
    from cryosweep_core.analyzers.tto import TTOAnalyzer
    return TTOAnalyzer().analyze(load_dat(str(path)), RunConfig())


def _arr(c, key):
    return np.array([np.nan if v is None else float(v) for v in (c.get(key) or [])], float)


def test_powerlaw_fixture_kappa_e_std_is_exact_one_percent():
    # tto_powerlaw_synth.dat carries rho_std = 1% of rho by construction: kappa_e = L0*T/rho
    # is single-variable, so kappa_e_std/kappa_e == 0.01 exactly (1e-12) wherever finite.
    r = _run(FX / "tto_powerlaw_synth.dat")
    checked = 0
    for c in r.data["curves"]:
        ke, ks = _arr(c, "kappa_e"), _arr(c, "kappa_e_std")
        m = np.isfinite(ke) & np.isfinite(ks) & (ke != 0)
        checked += int(m.sum())
        assert np.allclose(ks[m] / ke[m], 0.01, rtol=0, atol=1e-12)
    assert checked > 0
    json.dumps(r.data, allow_nan=False)


def test_element_wise_formulas_on_fixture():
    r = _run(FX / "tto_powerlaw_synth.dat")
    c = r.data["curves"][0]
    K, Ks = _arr(c, "kappa"), _arr(c, "kappa_std")
    R, Rs = _arr(c, "rho"), _arr(c, "rho_std")
    ke, kes = _arr(c, "kappa_e"), _arr(c, "kappa_e_std")
    kps = _arr(c, "kappa_ph_std")
    L, Ls = _arr(c, "lorenz_ratio"), _arr(c, "lorenz_ratio_std")
    m = np.isfinite(kes) & np.isfinite(kps) & np.isfinite(Ls)
    assert m.any()
    assert np.allclose(kes[m], ke[m] * (Rs[m] / R[m]), rtol=1e-12)
    assert np.allclose(kps[m], np.sqrt(Ks[m] ** 2 + kes[m] ** 2), rtol=1e-12)
    assert np.allclose(Ls[m], L[m] * np.sqrt((Ks[m] / K[m]) ** 2 + (Rs[m] / R[m]) ** 2),
                       rtol=1e-12)


def test_norho_fixture_std_companions_all_none():
    r = _run(FX / "tto_norho_synth.dat")
    for c in r.data["curves"]:
        for key in ("kappa_e_std", "kappa_ph_std", "lorenz_ratio_std"):
            vals = c.get(key)
            assert vals is None or all(v is None for v in vals)
    json.dumps(r.data, allow_nan=False)


def test_long_csv_header_18_columns_std_appended_last(tmp_path):
    r = _run(FX / "tto_powerlaw_synth.dat")
    out = export_result(r, str(tmp_path / "sample"))
    with open(out["tto"]) as f:
        header = next(csv.reader(f))
    assert len(header) == 18                              # closed O6: 15 -> 18, appended
    assert header[-3:] == ["kappa_e_std", "kappa_ph_std", "lorenz_ratio_std"]
    assert header[:15] == ["field_oe", "direction", "T", "kappa", "kappa_std", "seebeck",
                           "seebeck_std", "rho_ohm_m", "rho_std", "zt", "zt_std", "kappa_e",
                           "kappa_ph", "lorenz_ratio", "power_factor"]


def test_gate_file_rel_sigma_oracles(tto_real_path):
    r = _run(tto_real_path)
    # widest zero-field curve carries rho: use the curve with most finite kappa_e_std
    best, best_n = None, -1
    for c in r.data["curves"]:
        n = int(np.isfinite(_arr(c, "kappa_e_std")).sum())
        if n > best_n:
            best, best_n = c, n
    assert best_n > 0
    T = _arr(best, "t")

    def rel_at_tmin(vkey, skey):
        v, s = _arr(best, vkey), _arr(best, skey)
        m = np.isfinite(v) & np.isfinite(s) & (v != 0)
        i = int(np.argmin(T[m]))
        return abs(float(s[m][i] / v[m][i]))

    # re-measured through the shipped path, pinned 3 s.f. (spec anchors 8.15/10.52/8.46 %)
    assert rel_at_tmin("kappa_e", "kappa_e_std") == pytest.approx(0.0815, rel=0.02)
    assert rel_at_tmin("kappa_ph", "kappa_ph_std") == pytest.approx(0.1052, rel=0.02)
    assert rel_at_tmin("lorenz_ratio", "lorenz_ratio_std") == pytest.approx(0.0846, rel=0.02)
    json.dumps(r.data, allow_nan=False)
    r2 = _run(tto_real_path)
    assert json.dumps(r.data, sort_keys=True) == json.dumps(r2.data, sort_keys=True)
