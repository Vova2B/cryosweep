import numpy as np
import pytest
from cryosweep_core.analyzers.hall import (
    _antisymmetrize, _stage_fit, _carrier_n, _mobility, E_CHG,
)

def test_antisymmetrize_removes_even_component():
    H = np.linspace(-90000, 90000, 181)
    B = H / 10000.0
    slope = -5.0e-4
    R = slope * B + 3.0e-5 * B**2 + 1.0e-6      # odd + even + offset
    Hp, R_asym, sigma_asym = _antisymmetrize(H, R)
    assert sigma_asym is None
    assert np.all(Hp >= 0)
    # antisymmetric part recovers the pure odd term slope*B (even + offset removed)
    assert np.allclose(R_asym, slope * (Hp / 10000.0), atol=1e-9)

def test_antisymmetrize_sigma_never_moves_R_or_H():
    """Fix round 1 (2026-09): a row's own sigma validity must never change which R,H rows
    feed the fit -- R/Hp are independent of sigma_R entirely. Before this fix, a NaN in
    sigma_R at a row with perfectly finite H,R dropped that row from R,H too (the mask
    was AND-ed together), silently shifting R_asym/R_H/r2 for that setpoint even though
    Task 4's brief was to ADD a sigma family, not change the existing fit. Reproduction
    matches the one the reviewer ran: H = -100 has finite R but a NaN sigma; dropping it
    moves R_asym(100) from 0.9 to 1.1 by changing what np.interp(-100, ...) returns."""
    H = np.array([-300.0, -200.0, -100.0, 100.0, 200.0, 300.0])
    R = np.array([-3.5, -2.2, -0.7, 1.1, 2.3, 3.6])
    sig_all_finite = np.array([1e-9] * 6)
    sig_with_nan = np.array([1e-9, 1e-9, np.nan, 1e-9, 1e-9, 1e-9])

    Hp0, R0, S0 = _antisymmetrize(H, R, None)
    Hp1, R1, S1 = _antisymmetrize(H, R, sig_all_finite)
    Hp2, R2, S2 = _antisymmetrize(H, R, sig_with_nan)

    # the invariant under test: identical H, R -> identical Hp, R_asym, no matter what
    # sigma_R looks like (None, all-finite, or holding a NaN at a perfectly finite row)
    assert np.array_equal(Hp0, Hp1) and np.array_equal(Hp0, Hp2)
    assert np.array_equal(R0, R1) and np.array_equal(R0, R2)
    # pinned to the reviewer's own reproduction: R_asym(H=100) stays 0.9, never 1.1
    i100 = list(Hp0).index(100.0)
    assert R0[i100] == pytest.approx(0.9) and R2[i100] == pytest.approx(0.9)

    assert S0 is None
    assert S1 is not None                 # sigma still resolves when it is fully finite
    assert S2 is not None                 # and when only one row's sigma is NaN (>=2 remain)

    # ...and pin WHAT it resolves to. A single NaN between two valid readings is BACKFILLED
    # by interpolation from its neighbours, not declined at that grid point, so S2 equals
    # the all-finite S1 exactly here. That is a deliberate approximation (instrument noise
    # is smooth; a neighbour beats nothing) and it is documented in _antisymmetrize -- but
    # it means a resolved sigma_asym can carry a value the file never supplied. Asserting
    # it keeps the behaviour a decision rather than an accident: if someone later switches
    # to declining pointwise, this test is where they must say so.
    assert np.array_equal(S1, S2)
    # both branches carry sigma 1e-9, so sigma_asym = sqrt(1e-9^2 + 1e-9^2)/2
    assert S1 == pytest.approx(np.full(Hp0.size, np.sqrt(2) * 1e-9 / 2.0))

    # and the decline path: no sigma survives at all -> sigma_asym is None, never zeros
    _, _, S3 = _antisymmetrize(H, R, np.full(6, np.nan))
    assert S3 is None


def test_stage_fit_recovers_R_H_from_clean_odd_signal():
    H = np.linspace(-90000, 90000, 181)
    B = H / 10000.0
    R = -5.0e-4 * B                              # pure Hall, slope = R_H/t
    res = _stage_fit(H, R, thickness_m=1.0e-4, geometry_sign=1)
    assert res["slope_ohm_per_T"] == pytest.approx(-5.0e-4, rel=1e-6)
    assert res["R_H"] == pytest.approx(-5.0e-8, rel=1e-6)   # slope * thickness
    assert res["r2"] > 0.9999

def test_carrier_n_and_sign():
    n, sign = _carrier_n(-5.0e-8)
    assert n == pytest.approx(1.0 / (E_CHG * 5.0e-8), rel=1e-12)
    assert sign == "electrons"
    assert _carrier_n(5.0e-8)[1] == "holes"

def test_carrier_n_none_for_zero_or_none():
    assert _carrier_n(0.0) == (None, None)
    assert _carrier_n(None) == (None, None)

def test_mobility():
    mu = _mobility(-5.0e-8, rho_xx=1.0e-6)
    assert mu == pytest.approx(0.05, rel=1e-9)     # |R_H|/rho_xx
    assert _mobility(-5.0e-8, rho_xx=None) is None

def test_long_rho_xx_same_file_interpolates(hall_synth_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.io.columns import canonicalize_columns
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.hall import _long_rho_xx
    rt = load_dat(hall_synth_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    # hall_synth.dat carries zero-field rows at T=10/100/300 (not 150); rho_xx is 1e-6
    # everywhere in the fixture. RunConfig()'s default HallCfg.temp_interval is 1.0 K
    # (review round 1 Important #1 follow-up ruling): a query within temp_interval of its
    # NEAREST zero-field node resolves, one farther away (150 is 50 K from both 100 and
    # 300) declines rather than blend two unrelated setpoints together.
    fn, reason = _long_rho_xx(df, cmap, long_channel=2, long_df=None, long_cmap=None,
                              cfg=RunConfig())
    assert fn is not None and reason is None
    rho10, field10 = fn(10.0)
    assert rho10 == pytest.approx(1.0e-6, rel=1e-6)        # constant rho_xx in the fixture
    assert field10 <= 50.0                                 # zero-field rows only, never negative
    assert fn(150.0) == (None, None)                       # 50 K from its nearest node: declines

def test_long_rho_xx_absent_returns_none(hall_synth_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.io.columns import canonicalize_columns
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.hall import _long_rho_xx
    rt = load_dat(hall_synth_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    assert _long_rho_xx(df, cmap, long_channel=None, long_df=None, long_cmap=None,
                        cfg=RunConfig()) == (None, None)
