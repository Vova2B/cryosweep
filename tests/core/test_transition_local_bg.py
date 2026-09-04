import numpy as np
from cryosweep_core.fitting.transitions import local_window, wing_poly
from tests.core.synth_transitions import afm_like, narrow_window, debye_like

def test_local_window_scales_with_tc():
    T, cp = afm_like()
    w = local_window(T, cp, 203.0, wing_mask_k=2.0, wing_frac=0.03, span_mult=5.0)
    assert w["ok"]
    assert abs(w["W"] - 6.09) < 0.1                       # max(2.0, 0.03*203)
    assert w["T"].min() >= 203.0 - 5 * w["W"] - 1e-9 and w["T"].max() <= 203.0 + 5 * w["W"] + 1e-9

def test_local_window_narrow_group_all_in():
    T, cp = narrow_window()
    w = local_window(T, cp, 201.0, wing_mask_k=2.0, wing_frac=0.03, span_mult=5.0)
    assert w["ok"] and w["T"].size >= 50                  # ~all 62 points inside ±30 K

def test_wing_poly_cannot_bend_into_peak():
    T, cp = afm_like(noise=0.0)
    w = local_window(T, cp, 203.0, wing_mask_k=2.0, wing_frac=0.03, span_mult=5.0)
    coef = wing_poly(w["T"], w["cp"], w["inner"], order=2)
    resid = w["cp"] - np.polyval(coef, w["T"])
    assert resid[w["inner"]].max() > 5.0                  # peak stands ABOVE the wing poly
    assert np.abs(resid[~w["inner"]]).max() < 1.0         # wings well described

def test_local_window_insufficient_wings_declines():
    T = np.linspace(200.0, 206.0, 12)                     # everything inside the inner mask
    cp = debye_like(T)
    w = local_window(T, cp, 203.0, wing_mask_k=2.0, wing_frac=0.03, span_mult=5.0)
    assert not w["ok"] and "wing" in w["reason"]
