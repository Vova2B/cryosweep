import numpy as np
import pytest
from cryosweep_core.fitting.heat_capacity import fit_delta_h_overlay, MU_B_OVER_KB


def test_zeeman_recovers_g():
    B = np.array([0.5, 1.0, 2.0, 3.0, 5.0])            # Tesla
    fields_oe = (B * 1e4).tolist()
    g = 2.1
    deltas = (g * MU_B_OVER_KB * B).tolist()
    out = fit_delta_h_overlay(fields_oe, deltas, model="zeeman")
    assert out["ok"] and out["model"] == "zeeman"
    assert out["g_factor"] == pytest.approx(g, rel=1e-3)


def test_zfs_recovers_delta0_and_g():
    B = np.array([0.0, 1.0, 2.0, 3.0, 5.0]); g, D0 = 2.0, 4.0
    deltas = np.sqrt(D0 ** 2 + (g * MU_B_OVER_KB * B) ** 2).tolist()
    out = fit_delta_h_overlay((B * 1e4).tolist(), deltas, model="zfs")
    assert out["ok"] and out["Delta0"] == pytest.approx(D0, rel=0.05)
    assert out["g_factor"] == pytest.approx(g, rel=0.05)


def test_too_few_points_not_ok():
    out = fit_delta_h_overlay([0.0, 1e4], [0.0, 1.3], model="zeeman")
    assert out["ok"] is False
