import numpy as np
import pandas as pd
from types import SimpleNamespace
from cryosweep_core.analyzers.hc import HCAnalyzer
from cryosweep_core.config import RunConfig
from tests.core.synth_transitions import narrow_window, afm_like, debye_like


class _Hdr:
    title = "s"; app_version = None; n_atoms = 3.0


def _raw(rows):
    # minimal HC RawTable stand-in (same duck-typed pattern as test_export_hc.py):
    # rows = list of (T_K, field_Oe, Cp_J_per_molK); analyzer expects mJ/(mol*K).
    T = np.array([r[0] for r in rows], float)
    F = np.array([r[1] for r in rows], float)
    C = np.array([r[2] for r in rows], float)
    df = pd.DataFrame({"Sample Temp (Kelvin)": T, "Samp HC (mJ/mole-K)": C * 1e3,
                       "Field (Oe)": F})
    return SimpleNamespace(df=df, header=_Hdr(), path=None)


def _zero_field_rows():
    # afm_like alone has only ~3 points below 10 K — not enough for the analyzer's primary
    # low-T fit (needs >=5) — so densify the zero-field low-T region with smooth points.
    T0, C0 = afm_like()
    Tlow = np.linspace(2.0, 9.5, 16)
    rows = [(t, 0.0, c) for t, c in zip(Tlow, debye_like(Tlow))]
    rows += [(t, 0.0, c) for t, c in zip(T0, C0)]
    return rows


def test_high_t_only_group_gets_transition_attempt():
    T2, C2 = narrow_window(Tc=199.0)          # 2 T, high-T window only (no low-T points)
    rt = _raw(_zero_field_rows() + [(t, 20000.0, c) for t, c in zip(T2, C2)])
    cfg = RunConfig(heatcapacity={"transitions_enabled": True})
    res = HCAnalyzer().analyze(rt, cfg)
    groups = res.data.field_groups if hasattr(res.data, "field_groups") else res.data["field_groups"]
    g2 = next(g for g in groups if abs(g["field_oe"] - 20000.0) < 1500)
    assert g2["status"] == "insufficient"          # low-T outputs still gated
    assert g2.get("transition", {}).get("attempted")  # but the transition WAS attempted
    tch = res.data.tc_h if hasattr(res.data, "tc_h") else res.data["tc_h"]
    assert any(abs(p["field_oe"] - 20000.0) < 1500 for p in tch)


def test_flag_off_no_transition_key():
    rt = _raw(_zero_field_rows() + [(t, 20000.0, c) for t, c in zip(*narrow_window(Tc=199.0))])
    res = HCAnalyzer().analyze(rt, RunConfig())
    groups = res.data.field_groups if hasattr(res.data, "field_groups") else res.data["field_groups"]
    assert all("transition" not in g for g in groups)


def test_new_config_knob_defaults():
    from cryosweep_core.config import HeatCapacityCfg
    c = HeatCapacityCfg()
    assert c.transition_wing_frac == 0.03
    assert c.transition_span_mult == 5.0
    assert c.transition_wing_order == 3
    assert c.transition_prominence_n == 4.0
    assert c.transition_collapse_margin == 2.0
    assert c.transition_amp_max_frac == 1.0
