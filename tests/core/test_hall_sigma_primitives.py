"""The instrument-sigma primitives shared by both Hall analyzers.

sigma_R (Ohm) = std_column (Ohm-m) * (Resistance/Resistivity), the file's own exact
geometry factor. The ratio's constancy is a RUNTIME GATE, not an assumption: measured at
1e-12 relative spread on the real files, and the header geometry fields are not a valid
substitute (unset dummies predict a 10x-wrong factor). A shaky ratio must DECLINE.
"""
import numpy as np
import pandas as pd
import pytest
from cryosweep_core.analyzers.hall_sigma import (row_sigma_R, slope_sigma_ols,
                                                 RATIO_CONSTANCY_TOL)

class _CMap:
    def __init__(self, logical): self.logical = logical

def _frame(ratio_jitter=0.0, n=50):
    rng = np.random.default_rng(0)
    rho = np.full(n, 1e-6)
    ratio = 1000.0 * (1.0 + ratio_jitter * rng.standard_normal(n))
    return (pd.DataFrame({"R": rho * ratio, "RHO": rho, "SD": np.full(n, 2e-9)}),
            _CMap({"resistance_ch1": "R", "resistivity_ch1": "RHO",
                   "rho_std_bridge1": "SD"}))

def test_row_sigma_is_std_times_the_files_own_ratio():
    df, cmap = _frame()
    s = row_sigma_R(df, cmap, 1)
    assert s is not None
    assert np.allclose(s, 2e-9 * 1000.0)

def test_row_sigma_declines_when_the_ratio_is_not_constant():
    df, cmap = _frame(ratio_jitter=10 * RATIO_CONSTANCY_TOL)
    assert row_sigma_R(df, cmap, 1) is None

def test_row_sigma_declines_when_a_column_is_missing():
    df, cmap = _frame()
    del cmap.logical["rho_std_bridge1"]
    assert row_sigma_R(df, cmap, 1) is None

def test_slope_sigma_matches_the_closed_form_for_equal_errors():
    B = np.array([-2.0, -1.0, 1.0, 2.0])
    sig = np.full(4, 3e-9)
    got = slope_sigma_ols(B, sig)
    expected = 3e-9 / np.sqrt(np.sum((B - B.mean()) ** 2))
    assert got == pytest.approx(expected, rel=1e-12)

def test_slope_sigma_declines_on_a_degenerate_design():
    assert slope_sigma_ols(np.array([1.0, 1.0]), np.array([1e-9, 1e-9])) is None
    assert slope_sigma_ols(np.array([1.0, 2.0]), np.array([1e-9, np.nan])) is None
