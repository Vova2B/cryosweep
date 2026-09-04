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
    Hp, R_asym = _antisymmetrize(H, R)
    assert np.all(Hp >= 0)
    # antisymmetric part recovers the pure odd term slope*B (even + offset removed)
    assert np.allclose(R_asym, slope * (Hp / 10000.0), atol=1e-9)

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
    from cryosweep_core.analyzers.hall import _long_rho_xx
    rt = load_dat(hall_synth_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    fn = _long_rho_xx(df, cmap, long_channel=2, long_df=None, long_cmap=None)
    assert fn is not None
    assert fn(10.0) == pytest.approx(1.0e-6, rel=1e-6)     # constant rho_xx in the fixture
    assert fn(150.0) == pytest.approx(1.0e-6, rel=1e-6)

def test_long_rho_xx_absent_returns_none(hall_synth_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.io.columns import canonicalize_columns
    from cryosweep_core.analyzers.hall import _long_rho_xx
    rt = load_dat(hall_synth_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    assert _long_rho_xx(df, cmap, long_channel=None, long_df=None, long_cmap=None) is None
