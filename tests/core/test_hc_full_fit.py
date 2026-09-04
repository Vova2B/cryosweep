import numpy as np
from cryosweep_core.fitting.heat_capacity import specific_heat_full, fit_full_range

_INIT = {"theta_D": 100.0, "n": 7.0, "gamma": 0.007,
         "theta_E1": 50.0, "theta_E2": 150.0, "m1": 1.0, "m2": 2.0}
_FREE = {k: False if k == "n" else False for k in _INIT}  # all free for recovery test

def test_recovers_planted_params():
    true = {"theta_D": 220.0, "n": 5.0, "gamma": 0.012, "theta_E1": 60.0,
            "theta_E2": 180.0, "m1": 1.0, "m2": 1.5}
    T = np.linspace(2.0, 300.0, 120)
    cp = specific_heat_full(T, **true)
    out = fit_full_range(T, cp, init=dict(_INIT), fixed=dict(_FREE))
    assert out["ok"]
    assert out["r2"] > 0.999
    assert out["params"]["theta_E1"] <= out["params"]["theta_E2"]  # canonical ordering
    assert abs(out["params"]["theta_D"] - 220.0) < 5.0

def test_fixed_param_is_unchanged():
    T = np.linspace(2.0, 300.0, 80)
    cp = specific_heat_full(T, 200.0, 3.0, 0.01, 50.0, 150.0, 1.0, 2.0)
    fixed = {k: (k == "n") for k in _INIT}            # n fixed
    out = fit_full_range(T, cp, init={**_INIT, "n": 3.0}, fixed=fixed)
    assert out["ok"]
    assert out["params"]["n"] == 3.0

def test_failure_returns_ok_false_no_raise():
    out = fit_full_range(np.array([2.0, 3.0]), np.array([1.0, 1.1]),
                         init=dict(_INIT), fixed=dict(_FREE))
    assert out["ok"] is False
    assert out["params"] == {}


def test_seed_does_not_override_fixed_param():
    """A fixed param must keep its init value even if seed supplies a different value."""
    T = np.linspace(2.0, 300.0, 80)
    cp = specific_heat_full(T, 220.0, 3.0, 0.01, 60.0, 180.0, 1.0, 1.5)
    init = {**_INIT, "theta_D": 300.0, "n": 3.0}
    fixed = {k: (k in ("theta_D", "n")) for k in _INIT}   # theta_D and n fixed
    seed = {"theta_D": 150.0, "gamma": 0.01}               # seed tries to push theta_D to 150
    out = fit_full_range(T, cp, init=init, fixed=fixed, seed=seed)
    assert out["ok"], f"fit failed: {out.get('reason')}"
    assert out["params"]["theta_D"] == 300.0, (
        f"Fixed theta_D should stay at init=300.0, got {out['params']['theta_D']}"
    )
