import numpy as np
from cryosweep_core.analyzers.hc import _build_field_groups, HCData
from cryosweep_core.config import HeatCapacityCfg

def _multifield(Tc_by_field):
    # build (T, C, F) with an injected lambda anomaly per field setpoint
    from cryosweep_core.fitting import transitions as tr
    Ts, Cs, Fs = [], [], []
    for field_oe, Tc in Tc_by_field.items():
        T = np.linspace(2.0, 30.0, 60)
        C = tr.background(T, 0.01, 2e-4) + tr.lambda_anomaly(T, Tc, 0.110, 0.05, 0.08)
        Ts.append(T); Cs.append(C); Fs.append(np.full_like(T, field_oe))
    return np.concatenate(Ts), np.concatenate(Cs), np.concatenate(Fs)

def test_disabled_no_transition_key_oracle():
    T, C, F = _multifield({0.0: 10.0, 10000.0: 12.0})
    cfg = HeatCapacityCfg()                       # transitions_enabled=False
    groups, _ = _build_field_groups(T, C, F, cfg, n_atoms=4.0, lo=None, hi=10.0)
    assert all("transition" not in g for g in groups)

def test_enabled_attaches_transition_and_locates():
    T, C, F = _multifield({0.0: 10.0, 10000.0: 12.0})
    cfg = HeatCapacityCfg(transitions_enabled=True, transition_universality="ising3d")
    groups, _ = _build_field_groups(T, C, F, cfg, n_atoms=4.0, lo=None, hi=10.0)
    ok = [g for g in groups if g["status"] == "ok"]
    assert ok and all("transition" in g for g in ok)
    assert any(g["transition"].get("tc_determined") for g in ok)

def test_field_bin_aliasing_one_group_per_setpoint():
    T, C, F = _multifield({0.0: 10.0, 10000.0: 12.0, 30000.0: 15.0})
    cfg = HeatCapacityCfg(transitions_enabled=True)
    groups, _ = _build_field_groups(T, C, F, cfg, n_atoms=4.0, lo=None, hi=10.0)
    assert len([g for g in groups if g["status"] == "ok"]) == 3   # no two setpoints merged
