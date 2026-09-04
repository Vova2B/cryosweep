import numpy as np
from cryosweep_core.plotting.catalog import (series_resistivity_mr, series_resistivity_mr_pct,
                                        _CH_MARKERS)


def _fake_result(nch=2, directions=(1, -1)):
    class R:                                            # duck-typed: builders only touch .data
        data = {"probe": "resistivity", "bridges": []}
    for ch in range(1, nch + 1):
        curves = []
        for t in (10.0, 2.0):                           # deliberately unsorted
            curves.append({"held_temp_k": t, "direction": 0, "n_points": 5,
                           "rho_zero_field": 1e-4, "mr_percent_at_max_field": 5.0,
                           "max_abs_field_oe": 9e4, "low_confidence": False,
                           "directions": list(directions),
                           "field": [-9e4, 0.0, 9e4], "rho": [1.1e-4, 1e-4, 1.2e-4]})
        R.data["bridges"].append({"channel": ch, "rho_t_curves": [], "rho_h_curves": curves})
    return R()


def test_mr_series_sorted_by_temperature():
    out = series_resistivity_mr(_fake_result())
    temps = [float(s.label.split()[1]) for s in out]    # "Ch1 2.0 K ..." -> 2.0
    assert temps == sorted(temps)


def test_mr_channel_markers_only_when_multi():
    multi = series_resistivity_mr(_fake_result(nch=2))
    assert {s.marker for s in multi} == {_CH_MARKERS[1], _CH_MARKERS[2]}
    single = series_resistivity_mr(_fake_result(nch=1))
    assert all(s.marker is None for s in single)


def test_mr_direction_arrow_suffix():
    both = series_resistivity_mr(_fake_result(directions=(1, -1)))
    assert all(s.label_suffix == " ↑↓" for s in both)
    one = series_resistivity_mr(_fake_result(directions=(1,)))
    assert all(s.label_suffix == "" for s in one)


def test_mr_pct_builder_same_craft():
    out = series_resistivity_mr_pct(_fake_result())
    assert {s.marker for s in out} == {_CH_MARKERS[1], _CH_MARKERS[2]}
    temps = [float(s.label.split()[1]) for s in out]
    assert temps == sorted(temps)
    assert all(s.label_suffix == " ↑↓" for s in out)


def test_mr_pct_t_series_one_per_channel_sorted():
    from cryosweep_core.plotting.catalog import series_resistivity_mr_pct_t
    out = series_resistivity_mr_pct_t(_fake_result(nch=2))
    assert len(out) == 2                                 # one series per channel
    for s in out:
        assert s.x == sorted(s.x)                        # T ascending
        assert len(s.x) == 2 and s.y == [5.0, 5.0]
    assert {s.marker for s in out} == {_CH_MARKERS[1], _CH_MARKERS[2]}


def test_mr_pct_t_unbacked_without_mr():
    from cryosweep_core.plotting.catalog import series_resistivity_mr_pct_t
    r = _fake_result()
    for b in r.data["bridges"]:
        for c in b["rho_h_curves"]:
            c["mr_percent_at_max_field"] = None
    assert series_resistivity_mr_pct_t(r) == []
