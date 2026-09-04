import numpy as np
import pytest
from cryosweep_core.fitting.heat_capacity import (
    schottky_two_level, nuclear_tail, R,
    SCHOTTKY_ZPEAK, SCHOTTKY_CMAX_R1, MU_B_OVER_KB,
)

def test_peak_location_and_height_r1():
    Delta = 10.0
    T = np.linspace(0.5, 20.0, 4000)
    C = schottky_two_level(T, f=1.0, Delta=Delta, r=1.0)
    T_peak = T[int(np.argmax(C))]
    assert T_peak == pytest.approx(Delta / SCHOTTKY_ZPEAK, rel=2e-3)   # ~0.417*Delta
    assert C.max() == pytest.approx(SCHOTTKY_CMAX_R1 * R, rel=2e-3)     # 0.4392*R per mole TLS

def test_prefactor_is_fR_not_natoms():
    # doubling f doubles C; there is no n_atoms dependence
    T = np.array([3.0, 5.0])
    c1 = schottky_two_level(T, f=0.5, Delta=8.0)
    c2 = schottky_two_level(T, f=1.0, Delta=8.0)
    np.testing.assert_allclose(c2, 2.0 * c1, rtol=1e-12)

def test_high_T_limit_is_delta_squared_over_T2():
    # z<<1: C -> f*R*z^2 * r/(1+r)^2 = f*R*(Delta/T)^2 * 1/4  (r=1)
    T = np.array([500.0, 1000.0])
    Delta = 5.0
    C = schottky_two_level(T, f=1.0, Delta=Delta, r=1.0)
    expect = R * (Delta / T) ** 2 * 0.25
    np.testing.assert_allclose(C, expect, rtol=1e-3)

def test_overflow_safe_at_large_z():
    # T=1.9 K, Delta=90 K -> z ~ 47; must be finite and ~0, no overflow warning/inf
    with np.errstate(all="raise"):
        C = schottky_two_level(np.array([1.9]), f=1.0, Delta=90.0)
    assert np.isfinite(C).all()
    assert C[0] >= 0.0 and C[0] < 1e-6

def test_degeneracy_ratio_raises_peak_temperature():
    # Correct two-level Schottky physics: r=g0/g1 > 1 shifts the peak to HIGHER T
    # (z* drops 2.399 -> 2.228, so T_peak/Delta rises 0.417 -> 0.449) and lowers C_max.
    T = np.linspace(0.5, 20.0, 4000)
    tp1 = T[int(np.argmax(schottky_two_level(T, 1.0, 10.0, r=1.0)))]
    tp2 = T[int(np.argmax(schottky_two_level(T, 1.0, 10.0, r=2.0)))]
    assert tp2 > tp1

def test_nuclear_tail_and_constants():
    T = np.array([2.0, 4.0])
    np.testing.assert_allclose(nuclear_tail(T, 0.5), 0.5 / T ** 2, rtol=1e-12)
    assert MU_B_OVER_KB == pytest.approx(0.6717, abs=1e-4)
