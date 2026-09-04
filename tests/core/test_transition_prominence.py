from cryosweep_core.fitting.transitions import fit_transition
from tests.core.synth_transitions import afm_like, wide_null, broad_transition

def _fit(T, cp):
    return fit_transition(T, cp, form="lambda", universality="mean_field")

def test_flat_residual_fails_prominence():
    r = _fit(*wide_null())
    assert not r["tc_determined"]
    if r["attempted"] and r.get("prominence") is not None:
        assert r["prominence"] < r["prominence_floor"]

def test_real_peak_passes_prominence():
    r = _fit(*afm_like())
    assert r["tc_determined"] and r["prominence"] >= r["prominence_floor"]

def test_broad_transition_survives():
    r = _fit(*broad_transition())
    assert r["tc_determined"]          # prominence is a floor, not a sharpness test
