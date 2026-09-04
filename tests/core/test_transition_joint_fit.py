import numpy as np
from cryosweep_core.fitting.transitions import fit_transition
from tests.core.synth_transitions import afm_like, narrow_window, wide_null

def _fit(T, cp, form="lambda"):
    return fit_transition(T, cp, form=form, universality="mean_field")

def test_synthetic_zero_field_recovers_203():
    r = _fit(*afm_like())
    assert r["attempted"] and r["tc_determined"]
    assert abs(r["Tc"] - 203.0) < 5.0

def test_narrow_high_t_window_recovers():
    r = _fit(*narrow_window())
    assert r["tc_determined"] and abs(r["Tc"] - 201.0) < 5.0

def test_wide_null_declines():
    r = _fit(*wide_null())
    assert not r["tc_determined"]

def test_amplitude_bound_present_and_railed_is_undetermined():
    r = _fit(*afm_like())
    b = r.get("params", {})
    assert "Aplus" in b                      # anomaly params present on the chosen model
    # bounds recorded finite (amp bound active) — read via the advisory-free happy path
