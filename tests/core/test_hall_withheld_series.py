# tests/core/test_hall_withheld_series.py
"""Declined points must be inspectable without being published. They ride as a
default-OFF series, so every existing figure is byte-identical until a user ticks the
box, and they are drawn with the hollow-marker `two_point` role so they can never read as
trusted data.

Covers BOTH probes that carry a `withheld` dict on their points: the field-sweep `hall`
probe (series_hall_n_t / series_hall_mobility_t) and the temperature-dependent
`hall_tdep` probe (series_hall_tdep_n_t / series_hall_tdep_mobility_t). On the real Hall
file only `hall_tdep` ever withholds anything (72/138 points); `hall` withholds nothing on
any file we have, so its coverage here is necessarily synthetic. That asymmetry is the whole
reason both probes are covered: a field sweep whose R_H does not resolve is a real case, it
simply is not one the reference files exercise.
"""
from cryosweep_core.plotting.catalog import (
    series_hall_n_t, series_hall_mobility_t,
    series_hall_tdep_n_t, series_hall_tdep_mobility_t,
)
from cryosweep_core.result import Result, Provenance


def _res(points, probe="hall"):
    return Result(status="ok", confidence=1.0, data={"probe": probe, "points": points},
                  provenance=Provenance(file="x", sha256="", app_version=None, config={}))


# ---- field-sweep (`hall` probe) --------------------------------------------------

_RESOLVED = {"temperature": 10.0, "carrier_n": 1e28, "mobility": 1e-3, "withheld": None}
_WITHHELD = {"temperature": 20.0, "carrier_n": None, "mobility": None,
             "withheld": {"carrier_n": 5e28, "carrier_type": "holes", "mobility": 2e-3}}


def test_withheld_series_exists_and_is_off_by_default():
    ss = {s.key: s for s in series_hall_n_t(_res([_RESOLVED, _WITHHELD]))}
    assert "n" in ss and ss["n"].default_on is True
    assert "n_withheld" in ss
    assert ss["n_withheld"].default_on is False
    assert ss["n_withheld"].role == "two_point"      # hollow markers, dashed connector
    assert ss["n_withheld"].y == [5e28]
    assert ss["n_withheld"].x == [20.0]


def test_no_withheld_series_when_nothing_was_withheld():
    keys = {s.key for s in series_hall_n_t(_res([_RESOLVED]))}
    assert "n_withheld" not in keys


def test_mobility_gets_the_same_treatment():
    ss = {s.key: s for s in series_hall_mobility_t(_res([_RESOLVED, _WITHHELD]))}
    assert ss["mu_withheld"].default_on is False
    assert ss["mu_withheld"].role == "two_point"
    assert ss["mu_withheld"].y == [2e-3]


def test_withheld_only_still_yields_a_series():
    # Every point on this (synthetic) result is withheld -- the trusted series is absent,
    # but the withheld one must still be offered so the decline is inspectable.
    ss = {s.key: s for s in series_hall_n_t(_res([_WITHHELD]))}
    assert "n" not in ss
    assert "n_withheld" in ss and ss["n_withheld"].y == [5e28]


# ---- temp-dep (`hall_tdep` probe) ------------------------------------------------
# HallTDepPoint also carries `r_h_method` ("antisym" | "2point"), which the trusted n/mu
# series do not split on (unlike R_H's own series) -- confirm the withheld helper is not
# confused by that extra field.

_TDEP_RESOLVED = {"temperature": 10.0, "carrier_n": 1e28, "mobility": 1e-3,
                   "r_h_method": "antisym", "withheld": None}
_TDEP_WITHHELD = {"temperature": 20.0, "carrier_n": None, "mobility": None,
                   "r_h_method": "antisym",
                   "withheld": {"carrier_n": 5e28, "carrier_type": "holes", "mobility": 2e-3}}
_TDEP_WITHHELD_2POINT = {"temperature": 30.0, "carrier_n": None, "mobility": None,
                          "r_h_method": "2point",
                          "withheld": {"carrier_n": 7e28, "carrier_type": "electrons",
                                       "mobility": 4e-3}}


def test_tdep_withheld_series_exists_and_is_off_by_default():
    ss = {s.key: s for s in
          series_hall_tdep_n_t(_res([_TDEP_RESOLVED, _TDEP_WITHHELD], probe="hall_tdep"))}
    assert "n_antisym" in ss and ss["n_antisym"].default_on is True
    assert "n_withheld" in ss
    assert ss["n_withheld"].default_on is False
    assert ss["n_withheld"].role == "two_point"
    assert ss["n_withheld"].y == [5e28]


def test_tdep_no_withheld_series_when_nothing_was_withheld():
    keys = {s.key for s in
            series_hall_tdep_n_t(_res([_TDEP_RESOLVED], probe="hall_tdep"))}
    assert "n_withheld" not in keys


def test_tdep_mobility_gets_the_same_treatment():
    ss = {s.key: s for s in
          series_hall_tdep_mobility_t(_res([_TDEP_RESOLVED, _TDEP_WITHHELD], probe="hall_tdep"))}
    assert ss["mu_withheld"].default_on is False
    assert ss["mu_withheld"].role == "two_point"
    assert ss["mu_withheld"].y == [2e-3]


def test_tdep_withheld_series_pools_both_r_h_methods():
    # The trusted n series splits antisym vs 2-point into distinct series (n_antisym /
    # n_2point); the withheld series does not re-derive that split -- both a withheld
    # antisym point and a withheld 2-point point land in the one "n_withheld" series,
    # sorted by temperature, since neither carries a trustworthy R_H at all.
    ss = {s.key: s for s in series_hall_tdep_n_t(
        _res([_TDEP_RESOLVED, _TDEP_WITHHELD, _TDEP_WITHHELD_2POINT], probe="hall_tdep"))}
    assert ss["n_withheld"].x == [20.0, 30.0]
    assert ss["n_withheld"].y == [5e28, 7e28]
