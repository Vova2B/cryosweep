import math
import pytest
from cryosweep_core.plotting.catalog import fmt_field


@pytest.mark.parametrize("oe,expected", [
    (9999, "1 T"),
    (10000, "1 T"),
    (500, "0.05 T"),
    (40000, "4 T"),
    (137000, "13.7 T"),
    (9000, "0.9 T"),
    (9500, "0.95 T"),
    (0.481, "0 T"),     # near-zero instrument artifact rounds to 0 Oe -> "0 T" (not "4.81e-05 T")
    (0, "0 T"),
])
def test_tesla_three_sig_fig(oe, expected):
    assert fmt_field(oe, "T") == expected


@pytest.mark.parametrize("oe,expected", [
    (500, "500 Oe"),
    (0, "0 Oe"),
    (90000, "90000 Oe"),
])
def test_oe_path_unchanged(oe, expected):
    assert fmt_field(oe, "Oe") == expected
    assert fmt_field(oe) == expected          # default unit is Oe


def test_non_finite_and_none_return_empty():
    assert fmt_field(None, "T") == ""
    assert fmt_field(float("nan"), "T") == ""
    assert fmt_field(None, "Oe") == ""
    assert fmt_field(math.inf, "T") == ""
