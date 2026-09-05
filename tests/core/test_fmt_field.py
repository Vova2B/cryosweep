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


# KNOWN-ISSUES #7 (owner decision 2026-09-05): |H| >= FIELD_KOE_THRESHOLD_OE formats as
# kOe — '90 kOe', not '90000 Oe' — while everything below (the Curie-Weiss low-field
# regime, the MPMS 1000 Oe oracle) stays byte-identical.
@pytest.mark.parametrize("oe,expected", [
    (500, "500 Oe"),
    (0, "0 Oe"),
    (9999, "9999 Oe"),
    (10000, "10 kOe"),
    (90000, "90 kOe"),
    (45500, "45.5 kOe"),
])
def test_oe_path_koe_above_threshold(oe, expected):
    assert fmt_field(oe, "Oe") == expected
    assert fmt_field(oe) == expected          # default unit is Oe


def test_non_finite_and_none_return_empty():
    assert fmt_field(None, "T") == ""
    assert fmt_field(float("nan"), "T") == ""
    assert fmt_field(None, "Oe") == ""
    assert fmt_field(math.inf, "T") == ""
