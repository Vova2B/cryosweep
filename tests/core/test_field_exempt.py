"""Task 6: exempt HC/Hall builders always render Tesla labels (field_unit-independent),
normalized to fmt_field spacing; and the Hall vs-B builders read pre-stored Tesla x-data
that must never be scaled by the field_unit toggle.

NOTE: the brief's original harness used ``RunConfig.load(probe_override=...)`` which supplies
no hall channel/thickness, so every hall series came back empty and the asserts ran over
nothing (vacuous green). This version drives the analyzers directly with the same config the
existing hall render tests use, so len(series) > 0 and the guards actually execute.
"""
import pathlib
import numpy as np
import pytest
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}


def _hall_res():
    rt = load_dat(str(FIX / "hall_synth.dat"))
    return HallAnalyzer().analyze(rt, RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.5}))


def _hall_tdep_res():
    rt = load_dat(str(FIX / "hall_tdep_synth.dat"))
    return HallTempDepAnalyzer().analyze(
        rt, RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05, "longitudinal_channel": 2}))


def _xdata_invariant(kind_key, res):
    oe = KINDS[kind_key].series(res, field_unit="Oe")
    t = KINDS[kind_key].series(res, field_unit="T")
    assert len(oe) == len(t)
    assert len(oe) > 0, f"{kind_key}: series empty — test would be vacuous"
    for a, b in zip(oe, t):
        assert np.array_equal(np.asarray(a.x), np.asarray(b.x)), kind_key


def test_hall_field_sweep_x_unit_invariant():
    res = _hall_res()
    _xdata_invariant("hall_rxy_vs_B", res)
    _xdata_invariant("hall_asym_vs_B", res)
    _xdata_invariant("hall_raw_vs_asym", res)


def test_hall_tdep_asym_x_unit_invariant():
    res = _hall_tdep_res()
    _xdata_invariant("hall_tdep_asym_vs_B", res)


def test_interp_rt_label_has_space():
    res = _hall_tdep_res()
    s = KINDS["hall_tdep_interp_RT"].series(res)
    assert len(s) > 0, "interp_RT series empty — test would be vacuous"
    for sr in s:
        assert sr.label.endswith(" T"), sr.label          # normalized spacing


def test_hc_full_cp_t_and_entropy_labels_have_space():
    """Multi-field HC groups → colour-by-field Tesla labels normalized to '<n> T'."""
    from cryosweep_core.plotting.catalog import series_hc_full_cp_t, series_hc_entropy_vs_t

    class _R:
        def __init__(self, data):
            self.data = data

    groups = [
        {"field_oe": 0.0, "full_temperature": [2.0, 5.0], "full_cp": [1.0, 3.0],
         "entropy": {"temperature": [2.0, 5.0], "s_total": [0.1, 0.4]}},
        {"field_oe": 50000.0, "full_temperature": [2.0, 5.0], "full_cp": [0.9, 2.8],
         "entropy": {"temperature": [2.0, 5.0], "s_total": [0.1, 0.3]}},
    ]
    res = _R({"field_groups": groups})
    cp = series_hc_full_cp_t(res, field_unit="Oe")
    field_series = [s for s in cp if s.role != "fit"]
    assert field_series and all(s.label.endswith(" T") for s in field_series), \
        [s.label for s in field_series]
    assert {s.label for s in field_series} == {"0 T", "5 T"}

    # entropy overlays are off-by-default field series labelled "S total <n> T"
    res2 = _R({"entropy_available": True, "entropy_temperature": [2.0, 5.0],
               "entropy_total": [0.1, 0.4], "field_groups": groups})
    ent = series_hc_entropy_vs_t(res2, field_unit="Oe")
    field_labels = [s.label for s in ent if s.label.startswith("S total ") and s.label != "S total"]
    assert field_labels and all(l.endswith(" T") for l in field_labels), field_labels
    assert set(field_labels) == {"S total 0 T", "S total 5 T"}
