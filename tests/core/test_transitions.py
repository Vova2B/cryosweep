import numpy as np
import pytest
from cryosweep_core.fitting import transitions as tr

def test_universality_table():
    assert tr.UNIVERSALITY["mean_field"] == 0.0
    assert tr.UNIVERSALITY["ising3d"] == pytest.approx(0.110)
    assert tr.UNIVERSALITY["xy3d"] == pytest.approx(-0.013)

def test_lambda_anomaly_powerlaw_diverges_toward_Tc():
    T = np.array([8.0, 9.0, 9.9])
    c = tr.lambda_anomaly(T, Tc=10.0, alpha=0.110, Aplus=1.0, Aminus=1.0)
    # closer to Tc from below => larger |t|^{-alpha}
    assert c[2] > c[1] > c[0] > 0

def test_lambda_anomaly_log_branch_at_alpha0():
    T = np.array([8.0, 9.5])
    c = tr.lambda_anomaly(T, Tc=10.0, alpha=0.0, Aplus=1.0, Aminus=1.0)
    # -A ln|t|: |t| small => ln negative => -A ln|t| positive and larger nearer Tc
    assert c[1] > c[0] > 0

def test_lambda_anomaly_branch_asymmetry():
    Tb = np.array([9.0]); Ta = np.array([11.0])
    cb = tr.lambda_anomaly(Tb, 10.0, 0.110, Aplus=1.0, Aminus=3.0)
    ca = tr.lambda_anomaly(Ta, 10.0, 0.110, Aplus=1.0, Aminus=3.0)
    # same |t|=0.1 both sides; below uses Aminus=3, above uses Aplus=1
    assert cb[0] == pytest.approx(3.0 * ca[0], rel=1e-9)

def test_lambda_anomaly_finite_at_Tc():
    c = tr.lambda_anomaly(np.array([10.0]), 10.0, 0.110, 1.0, 1.0)
    assert np.isfinite(c[0])   # |t|=0 must be regularized, not inf/nan

def test_background_monotone_and_t5():
    T = np.linspace(1, 20, 50)
    b = tr.background(T, gamma=0.01, beta=1e-4, delta=1e-7, lattice_t5=True)
    assert np.all(np.diff(b) > 0)                       # monotone increasing for positive coeffs
    b2 = tr.background(T, 0.01, 1e-4, 1e-7, lattice_t5=False)
    assert b2[-1] < b[-1]                               # delta term ignored when lattice_t5=False

def test_jump_step():
    T = np.array([5.0, 10.0, 15.0])
    s = tr.jump_step(T, Tc=10.0, dC=2.0)
    assert s[0] == pytest.approx(2.0)   # below
    assert s[2] == pytest.approx(0.0)   # above
    assert s[1] == pytest.approx(1.0)   # at Tc => 0.5*dC


def _synthetic_lambda(Tc=10.0, alpha=0.110, n=41):
    T = np.linspace(2.0, 20.0, n)
    cp = tr.background(T, 0.01, 2e-4) + tr.lambda_anomaly(T, Tc, alpha, 0.05, 0.08)
    return T, cp

def test_locate_lambda_finds_interior_peak_near_Tc():
    T, cp = _synthetic_lambda(Tc=10.0)
    loc = tr.locate_lambda(T, cp)
    assert loc["interior"] is True
    assert abs(loc["Tc_seed"] - 10.0) < 1.5

def test_locate_lambda_rejects_monotone_tail():
    T = np.linspace(2.0, 20.0, 40)
    cp = tr.background(T, 0.01, 3e-4)          # smooth, no anomaly, rising tail
    loc = tr.locate_lambda(T, cp)
    assert loc["interior"] is False           # no interior bump

def test_locate_jump_finds_step():
    T = np.linspace(2.0, 20.0, 40)
    cp = tr.background(T, 0.01, 2e-4) + tr.jump_step(T, Tc=12.0, dC=0.5)
    loc = tr.locate_jump(T, cp)
    assert loc["interior"] is True
    assert abs(loc["Tc_seed"] - 12.0) < 1.5

def test_locate_jump_no_step_low_stat():
    T = np.linspace(2.0, 20.0, 40)
    cp = tr.background(T, 0.01, 2e-4)
    loc_step = tr.locate_jump(T, cp + tr.jump_step(T, 12.0, 0.5))
    loc_flat = tr.locate_jump(T, cp)
    assert loc_step["stat"] > loc_flat["stat"]   # step statistic discriminates


def test_artifact_filter_drops_lone_spike_keeps_cluster():
    T = np.linspace(2.0, 20.0, 40)
    cp = tr.background(T, 0.01, 2e-4)
    cp_spike = cp.copy(); cp_spike[20] += 5.0                 # lone spike
    r = tr.artifact_filter(T, cp_spike)
    assert r["dropped"] >= 1
    assert any("spike" in a.lower() or "artifact" in a.lower() for a in r["advisories"])
    # a real transition cluster (several elevated neighbors) must survive:
    cp_cluster = cp.copy(); cp_cluster[18:23] += 2.0
    r2 = tr.artifact_filter(T, cp_cluster)
    assert r2["dropped"] == 0

def test_artifact_filter_drops_duplicate_T_multivalued():
    T = np.array([2.0, 4.0, 6.0, 6.0, 8.0, 10.0, 12.0])       # duplicate at 6.0
    cp = np.array([0.1, 0.2, 0.3, 9.9, 0.5, 0.6, 0.7])        # wildly different at the dup
    r = tr.artifact_filter(T, cp)
    assert r["dropped"] >= 1


def test_artifact_filter_dupT_keeps_cluster_point():
    # Smooth background with a genuine 5-point elevated transition cluster.
    # (Elevation of 4.0, not the minimal 2.0, so the gap clears the MAD-based
    # threshold with margin -- MAD is robust to the 5/41-point cluster, so a
    # too-small elevation would not even register as a duplicate-T artifact.)
    T = np.linspace(2.0, 20.0, 40)
    cp = tr.background(T, 0.01, 2e-4)
    cp = cp.copy()
    cp[18:23] += 4.0                                   # real transition cluster (elevated)
    elevated_val = cp[20]
    baseline_val = float(tr.background(T, 0.01, 2e-4)[20])  # what the point would be w/o the cluster
    # Duplicate the T of the cluster point (index 20) with a near-background spurious reading,
    # inserted immediately after it so the pair is (cluster_point, spurious_duplicate).
    T_dup = np.insert(T, 21, T[20])
    cp_dup = np.insert(cp, 21, baseline_val)
    r = tr.artifact_filter(T_dup, cp_dup)
    assert r["dropped"] >= 1
    # the elevated (cluster) value at that T must survive; the spurious baseline dup must be dropped
    at_T = r["cp"][np.isclose(r["T"], T[20])]
    assert at_T.size == 1
    assert at_T[0] == pytest.approx(elevated_val)


def test_artifact_filter_drops_downward_dip():
    T = np.linspace(2.0, 20.0, 40)
    cp = tr.background(T, 0.01, 2e-4)
    cp_dip = cp.copy(); cp_dip[20] -= 5.0               # lone dip
    r = tr.artifact_filter(T, cp_dip)
    assert r["dropped"] >= 1
    assert any("dip" in a.lower() or "artifact" in a.lower() for a in r["advisories"])


def _models_via_window(T, cp, Tc_seed, form, alpha, order=3):
    # _fit_transition_models now operates on a local window with a wing-fixed poly:
    # reproduce fit_transition's wiring for direct model-level tests.
    win = tr.local_window(T, cp, Tc_seed, wing_mask_k=2.0, wing_frac=0.03, span_mult=5.0)
    return tr._fit_transition_models(win["T"], win["cp"], form=form, alpha=alpha,
                                     Tc_seed=Tc_seed, W=win["W"], inner=win["inner"],
                                     order=order, amp_max_frac=1.0, aicc_margin=2.0)

def test_fit_models_lambda_beats_background_on_anomaly():
    T, cp = _synthetic_lambda(Tc=10.0, alpha=0.110)
    loc = tr.locate_lambda(T, cp)
    res = _models_via_window(T, cp, loc["Tc_seed"], form="lambda", alpha=0.110)
    assert res["chosen_key"] == "lambda"
    assert abs(res["models"]["lambda"]["params"]["Tc"] - 10.0) < 1.0

def test_fit_models_lambda_declines_on_smooth_data():
    T = np.linspace(2.0, 20.0, 41)
    cp = tr.background(T, 0.01, 3e-4)                          # no anomaly
    res = _models_via_window(T, cp, 10.0, form="lambda", alpha=0.110)
    assert res["chosen_key"] == "background"                   # AICc margin not beaten

def test_fit_models_jump_recovers_Tc_and_dC():
    T = np.linspace(2.0, 20.0, 41)
    cp = tr.background(T, 0.01, 2e-4) + tr.jump_step(T, Tc=12.0, dC=0.5)
    loc = tr.locate_jump(T, cp)
    res = _models_via_window(T, cp, loc["Tc_seed"], form="jump", alpha=0.0)
    assert res["chosen_key"] == "jump"
    p = res["models"]["jump"]["params"]
    assert abs(p["Tc"] - 12.0) < 1.0 and abs(p["dC"] - 0.5) < 0.2

def test_fit_models_deterministic():
    T, cp = _synthetic_lambda(Tc=10.0)
    loc = tr.locate_lambda(T, cp)
    a = _models_via_window(T, cp, loc["Tc_seed"], form="lambda", alpha=0.110)
    b = _models_via_window(T, cp, loc["Tc_seed"], form="lambda", alpha=0.110)
    assert a["models"]["lambda"]["params"] == b["models"]["lambda"]["params"]


def test_fit_transition_lambda_determined_on_anomaly():
    T, cp = _synthetic_lambda(Tc=10.0, alpha=0.110)
    g = tr.fit_transition(T, cp, form="lambda", universality="ising3d")
    assert g["attempted"] and g["tc_determined"] is True
    assert abs(g["Tc"] - 10.0) < 1.0
    assert g["delta_aicc"] >= 2.0

def test_fit_transition_declines_on_smooth_real_like_data():
    # REAL-curvature featureless null (Debye-like + noise) — harder than the old
    # background-drawn cubic, which the wing-poly architecture fits trivially.
    from tests.core.synth_transitions import wide_null
    g = tr.fit_transition(*wide_null(), form="lambda", universality="mean_field")
    assert g["tc_determined"] is False                        # NULL case: no fabricated Tc

def test_fit_transition_refuses_edge_transition():
    # anomaly jammed against the low-T edge cannot be bracketed -> must be refused
    T = np.linspace(2.0, 20.0, 40)
    cp = tr.background(T, 0.01, 2e-4) + tr.lambda_anomaly(T, 2.3, 0.110, 0.05, 0.05)
    g = tr.fit_transition(T, cp, form="lambda", universality="ising3d")
    assert g["tc_determined"] is False

def test_fit_transition_jump_determined():
    T = np.linspace(2.0, 20.0, 41)
    cp = tr.background(T, 0.01, 2e-4) + tr.jump_step(T, 12.0, 0.5)
    g = tr.fit_transition(T, cp, form="jump", universality="mean_field")
    assert g["tc_determined"] is True and abs(g["Tc"] - 12.0) < 1.0

def test_fit_transition_json_safe():
    import json, math
    T, cp = _synthetic_lambda(Tc=10.0)
    g = tr.fit_transition(T, cp, form="lambda", universality="ising3d")
    json.dumps(g)                                             # must not raise (no nan/inf leak)
    assert g["Tc_sigma"] is None or math.isfinite(g["Tc_sigma"])


def test_fit_transition_jump_null_declines():
    # featureless REAL-curvature data, form="jump": the jump locator returns interior=True
    # even on flat data (no genuine step to reject a candidate on), so this proves the
    # AICc/prominence gates -- not the locator -- are what decline here.
    from tests.core.synth_transitions import wide_null
    g = tr.fit_transition(*wide_null(), form="jump", universality="mean_field")
    assert g["tc_determined"] is False

def test_fit_transition_broad_feature_unresolved_advisory():
    # a broad weak bump on SPARSE noisy real curvature: the locator finds an interior
    # candidate (a genuine local maximum away from both endpoints), but with so few points
    # the 3-extra-parameter lambda model cannot beat the AICc margin over background ->
    # must decline AND emit the "broad feature unresolved" advisory rather than silently
    # vanishing. (The old fixture, a noiseless 0.003-amplitude bump on a drawn background,
    # is meaningless under the wing-poly architecture: with zero noise ANY interior
    # feature is infinitely significant.)
    from tests.core.synth_transitions import debye_like, rng
    T = np.linspace(100.0, 200.0, 24)
    cp = debye_like(T) + 0.6 * np.exp(-0.5 * ((T - 150.0) / 12.0) ** 2) \
         + rng(2).normal(0, 0.4, T.size)
    g = tr.fit_transition(T, cp, form="lambda", universality="mean_field")
    assert g["tc_determined"] is False
    assert any("broad feature unresolved" in a for a in g["advisories"])

def test_fit_transition_sloped_background_labels_cp_peak():
    # lambda anomaly riding a sloped (gamma dominant) background -- guards that the Cp-peak
    # convention survives a sloped background. If the determined Tc drifts, that is the honest
    # limitation, so allow up to ~2 K.
    T = np.linspace(2.0, 20.0, 41)
    cp = tr.background(T, 0.05, 2e-4) + tr.lambda_anomaly(T, 10.0, 0.110, 0.05, 0.08)
    g = tr.fit_transition(T, cp, form="lambda", universality="ising3d")
    assert g["tc_determined"] is True
    assert abs(g["Tc"] - 10.0) < 2.0


def test_compare_forms_indistinguishable_when_aicc_close(monkeypatch):
    # Drive delta_aicc directly via a fit_transition stub so the verdict logic is tested
    # honestly and deterministically, instead of hoping some synthetic curve happens to
    # produce a small |delta_aicc|. compare_transition_forms calls the module-level
    # `fit_transition(T, cp, form=..., **kw)` -- monkeypatching the module attribute
    # intercepts both calls (form="lambda" then form="jump").
    T = np.linspace(2.0, 20.0, 41)
    cp = tr.background(T, 0.01, 2e-4)   # content is irrelevant; fit_transition is stubbed

    def fake_close(T, cp, *, form, **kwargs):
        aicc = 10.0 if form == "lambda" else 11.0   # |delta_aicc| = 1.0 < default band 2.0
        return {"aicc": aicc, "attempted": True, "tc_determined": True, "form": form}

    monkeypatch.setattr(tr, "fit_transition", fake_close)
    c = tr.compare_transition_forms(T, cp, universality="mean_field")
    assert c["verdict"] == "indistinguishable on this data"   # DEFAULT band, genuine closeness

    def fake_decisive(T, cp, *, form, **kwargs):
        aicc = 10.0 if form == "lambda" else 20.0   # |delta_aicc| = 10.0, lambda clearly lower
        return {"aicc": aicc, "attempted": True, "tc_determined": True, "form": form}

    monkeypatch.setattr(tr, "fit_transition", fake_decisive)
    c2 = tr.compare_transition_forms(T, cp, universality="mean_field")
    assert c2["verdict"] == "lambda"   # lower aicc wins when the gap is decisive

def test_compare_forms_picks_jump_on_clear_step():
    T = np.linspace(2.0, 20.0, 41)
    cp = tr.background(T, 0.01, 2e-4) + tr.jump_step(T, 12.0, 0.6)
    c = tr.compare_transition_forms(T, cp, universality="ising3d",
                                    indistinguishable_band=2.0)
    assert c["verdict"] == "jump"   # real fixture gives a decisive delta_aicc (~-83)
    assert "lambda" in c and "jump" in c


def test_config_transition_defaults():
    from cryosweep_core.config import HeatCapacityCfg
    c = HeatCapacityCfg()
    assert c.transitions_enabled is False
    assert c.transition_form == "lambda"
    assert c.transition_universality == "mean_field"
    assert c.transition_lattice_t5 is False
    assert c.transition_wing_mask_k == 2.0
    assert c.transition_aicc_margin == 2.0
    assert c.transition_compare_forms is False
    assert c.transition_indistinguishable_band == 2.0
