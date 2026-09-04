import numpy as np
from cryosweep_core.fitting.transitions import fit_transition
from tests.core.synth_transitions import afm_like, broad_transition, debye_like, rng

def _fit(T, cp):
    return fit_transition(T, cp, form="lambda", universality="mean_field")

def test_localized_peak_passes_collapse():
    r = _fit(*afm_like())
    assert r["tc_determined"]
    # a real localized anomaly loses MOST of its advantage after near-Tc removal: the
    # residual advantage stays under the gate's ceiling max(margin, 0.4*original) — its
    # tails carry genuine signal, so a small absolute remainder is expected, not zero
    if r["collapse_delta_aicc"] is not None:
        assert r["collapse_delta_aicc"] < max(2.0, 0.4 * r["delta_aicc"])

def test_broad_real_transition_passes_via_width_scaling():
    r = _fit(*broad_transition())
    assert r["tc_determined"]

def test_broad_background_mismatch_rejected():
    # smooth bowl the wing poly slightly misfits EVERYWHERE (order-4 curvature, no peak):
    # any AICc win must persist after near-Tc removal -> rejected by collapse gate
    T = np.linspace(150.0, 260.0, 120)
    cp = debye_like(T) + 3e-7 * (T - 205.0) ** 4 + rng(1).normal(0, 0.05, T.size)
    r = _fit(T, cp)
    assert not r["tc_determined"]
