import numpy as np
import pytest
from cryosweep_core.primitives import (
    normalized_span, monotone_fraction, robust_slope, cluster_setpoints, linfit,
)


def test_robust_slope_ignores_nan():
    # Bug 4: NaNs must be filtered before theilslopes (else slope is nan -> direction 0)
    x = np.arange(50.0)
    y = 2.0 * x + 1.0
    y[10] = np.nan
    y[20] = np.nan
    y[35] = np.nan
    s = robust_slope(y)
    assert np.isfinite(s)
    assert abs(s - 2.0) < 0.05


def test_robust_slope_few_finite_returns_zero():
    # Bug 4: <2 finite points -> 0.0 (cannot define a slope)
    assert robust_slope(np.array([np.nan, np.nan, np.nan])) == 0.0
    assert robust_slope(np.array([np.nan, 5.0])) == 0.0


def test_cluster_setpoints_empty():
    # Bug 6: empty input must not IndexError
    out = cluster_setpoints(np.array([]), rel_tol=0.05)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_normalized_span_tol_zero_no_divide():
    # Bug 6: tol=0 returns raw ptp, no div-by-zero
    assert normalized_span(np.array([0.0, 3.0, 1.0]), tol=0.0) == 3.0
    assert normalized_span(np.array([5.0]), tol=0.0) == 0.0

def test_normalized_span():
    assert normalized_span(np.array([0.0, 10.0, 5.0]), tol=1.0) == 10.0
    # ptp([2.20,2.21,2.19]) = 0.02 (NOT 0.04); use approx for float error
    assert abs(normalized_span(np.array([2.20, 2.21, 2.19]), tol=0.5) - 0.02 / 0.5) < 1e-9

def test_normalized_span_ignores_nan():
    assert normalized_span(np.array([1.0, np.nan, 3.0]), tol=1.0) == 2.0

def test_monotone_fraction_ramp_vs_loop():
    ramp = np.arange(100.0)
    assert monotone_fraction(ramp) == 1.0
    loop = np.concatenate([np.arange(50.0), np.arange(50.0, 0, -1)])
    assert monotone_fraction(loop) < 0.6   # half up, half down

def test_robust_slope_ignores_outliers():
    x = np.arange(50.0)
    y = 2.0 * x + 1.0
    y[25] += 1000.0   # single spike
    assert abs(robust_slope(y) - 2.0) < 0.05   # Theil-Sen median slope over row order

def test_cluster_setpoints():
    vals = np.array([3.0, 3.01, 2.99, 5.0, 5.02])
    labels = cluster_setpoints(vals, rel_tol=0.05)
    assert len(set(labels)) == 2

def test_linfit_returns_sigma_and_r2():
    x = np.arange(20.0); y = 3.0 * x + 2.0
    fit = linfit(x, y)
    assert abs(fit.slope - 3.0) < 1e-9
    assert fit.r2 > 0.999
    assert hasattr(fit, "sigma_slope")
