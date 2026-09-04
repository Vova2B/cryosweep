import numpy as np
import pytest
from cryosweep_core.robust import robust_range, outlier_mask, outlier_stats, robust_decade_span, is_log_space

def test_clean_array_has_no_outliers_and_full_range():
    v = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02])
    assert not outlier_mask(v).any()
    lo, hi = robust_range(v)
    assert lo <= v.min() + 1e-12 and hi >= v.max() - 1e-12
    st = outlier_stats(v)
    assert st["n_outliers"] == 0 and st["fraction"] == 0.0

def test_heavy_tail_is_flagged_and_indices_reported():
    v = np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02, 50.0])
    m = outlier_mask(v, k=8.0)
    assert m[6] and m.sum() == 1
    st = outlier_stats(v, k=8.0)
    assert st["n_outliers"] == 1 and st["outlier_indices"] == [6]
    assert st["robust_range"][1] < 50.0
    assert st["max_over_median"] > 10.0

def test_zero_mad_returns_full_range_no_outliers():
    v = np.array([3.0, 3.0, 3.0, 3.0])
    assert robust_range(v) == (3.0, 3.0)
    assert not outlier_mask(v).any()
    assert outlier_stats(v)["n_outliers"] == 0

def test_empty_and_all_nan_never_raise():
    for v in (np.array([]), np.array([np.nan, np.nan])):
        lo, hi = robust_range(v)
        assert np.isnan(lo) and np.isnan(hi)
        assert not outlier_mask(v).any()
        st = outlier_stats(v)
        assert st["n"] == 0 and st["n_outliers"] == 0 and st["fraction"] == 0.0

def test_nonfinite_positions_are_false_and_mask_aligns_to_input():
    v = np.array([1.0, np.nan, 1.1, np.inf, 50.0])
    m = outlier_mask(v, k=8.0)
    assert m.shape == v.shape
    assert not m[1] and not m[3]
    assert m[4]

def test_robust_decade_span_ignores_tail():
    rng = np.random.default_rng(0)
    bulk = 10 ** rng.uniform(-5.3, -4.0, size=567)
    v = np.concatenate([bulk, np.full(7, 1.46e-2)])
    assert robust_decade_span(v) < 2.0
    assert np.log10(v.max() / v.min()) > 3.0

def test_log_space_for_multidecade_curve():
    v = np.logspace(-3, 3, 200)
    assert robust_decade_span(v) > 2.0
    st = outlier_stats(v)
    assert st["log_space"] is True
    assert st["n_outliers"] == 0
    lo, hi = st["robust_range"]
    assert lo <= v.min() and hi >= v.max()


def test_is_log_space_matches_threshold():
    import numpy as np
    assert is_log_space(np.logspace(-3, 3, 200)) is True
    assert is_log_space(np.array([1.0, 1.1, 0.9, 1.05, 0.95, 1.02])) is False
