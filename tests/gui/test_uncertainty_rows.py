"""U3 honest-uncertainty GUI rows (2026-08-10 spec §7 / Task 12).

Copy contract: no ± glyph on a window-sensitive number; statistical sigma named σ_stat,
qualified, last; instrument sigma ALWAYS labeled as instrument noise, never conflatable
with a fit sigma.
"""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest                                                    # noqa: E402

from cryosweep_gui.output_panel import flatten_rows                   # noqa: E402


def _vsm_data(window_sensitive):
    flags = ["window_sensitive"] if window_sensitive else []
    return {
        "probe": "vsm",
        "fit": {"params": {"C": 1.5, "theta": -50.27, "mu_eff": 4.499},
                "sigma": {"C": 0.01, "theta": 0.99, "mu_eff": 0.01},
                "r2": 0.996515, "quality_flags": flags},
        "cw_ladder": [
            {"tmin_k": 25.0, "theta_k": -42.24, "sigma_theta_k": 0.13, "mu_eff": 4.427,
             "sigma_mu_eff": 0.002, "r2": 0.99995, "n_points": 138},
            {"tmin_k": 200.0, "theta_k": -37.55, "sigma_theta_k": 0.60, "mu_eff": 4.392,
             "sigma_mu_eff": 0.005, "r2": 0.99979, "n_points": 50},
        ],
        "theta_spread_k": 12.72, "mu_eff_spread": 0.107,
    }


def test_window_sensitive_cw_row_follows_u3():
    rows = dict(flatten_rows(_vsm_data(True)))
    row = rows["Curie-Weiss θ"]
    assert "WINDOW-SENSITIVE" in row
    assert "σ_stat" in row and "fit scatter only" in row
    # no ± anywhere before the sigma clause — the U3 rule: never "θ ± spread"
    assert "±" not in row
    # F3: the HEADLINE is the shipped fit's theta (-50.3), NOT the T>=25 K rung (-42.2).
    # Spec §1.3 (authoritative over §7's draft copy): the published number must not move.
    plain = row.replace("−", "-")
    assert plain.startswith("θ = -50.3 K (full-window fit — REPORTED)")
    assert "-42.2" not in plain.split("WINDOW-SENSITIVE")[0]
    # the ladder drift and the rung are present, as subordinate context
    assert "T≥25" in row and "-37.5" in plain and "-42.2" in plain
    # σ_stat is the LAST clause
    assert row.rstrip().endswith("(fit scatter only, not the uncertainty on θ)")


def test_clean_cw_row_unchanged_from_today():
    # Captured BEFORE the Task-12 edit: a clean fit emits ONLY the generic fit.* rows,
    # exactly as today (str(param)); no dedicated CW row is added.
    rows = dict(flatten_rows(_vsm_data(False)))
    assert "Curie-Weiss θ" not in rows
    assert rows["fit.theta"] == "-50.27"


def test_hall_point_rows_sigma_labels():
    data = {"probe": "hall", "points": [
        {"temperature": 2.0, "R_H": -7.2e-9, "r_h_sigma": 1.5e-11, "sigma_zero_dof": False},
        {"temperature": 300.0, "R_H": -7.2e-4, "r_h_sigma": None, "sigma_zero_dof": True},
    ]}
    rows = dict(flatten_rows(data))
    assert "±" in rows["R_H@2.0K"] and "σ residual" in rows["R_H@2.0K"]
    assert "(no σ — 2-point method)" in rows["R_H@300.0K"]


def test_hall_tdep_instrument_sigma_row_labeled():
    data = {"probe": "hall_tdep", "points": [
        {"temperature": float(t), "R_H": 1.3e-11, "r_h_sigma": None,
         "r_h_sigma_instrument": 1.6e-11, "sigma_zero_dof": False}
        for t in range(5)
    ]}
    rows = dict(flatten_rows(data))
    inst = rows["R_H(T) σ_inst"]
    assert "σ_inst" in inst and "instrument noise, not fit quality" in inst
    res = rows["R_H(T) σ (residual)"]
    assert "0/5" in res


def test_resistivity_powerlaw_row_tto_idiom_and_rrr_pm():
    pl = {"params": {"rho0": 1e-5, "A": 2e-7, "n": 0.769}, "sigma": {"n": 0.013},
          "r2": 0.992, "quality_flags": ["window_sensitive"], "fit_range": [2.0, 30.0]}
    data = {"probe": "resistivity", "bridges": [{
        "channel": 1, "rrr": 18.52, "rrr_std": 0.349, "power_law": pl,
        "power_law_n_spread": 0.269,
        "power_law_ladder": [
            {"cutoff_k": 10.0, "n": 0.500, "sigma": 0.01, "r2": 0.99, "n_points": 40},
            {"cutoff_k": 30.0, "n": 0.769, "sigma": 0.013, "r2": 0.992, "n_points": 90}],
        "rho_h_curves": [],
    }]}
    rows = dict(flatten_rows(data))
    # F13 (final-review): rrr_std is instrument-derived, so U5/O4 require it labeled as
    # instrument noise — a bare ± was the one unqualified error bar left in the slice.
    assert rows["ch1.RRR"] == ("18.52 ± 0.35 (σ_inst — instrument noise; "
                               "excludes ramp/endpoint choice)")
    row = rows["ch1.ρ = ρ₀ + A·Tⁿ"]
    assert row.startswith("n ≈ 0.8")                     # 1 dp when window-sensitive
    assert "WINDOW-SENSITIVE" in row
    assert "flags: window_sensitive" in row
    assert "σ_stat 0.013 (fit scatter only, not the uncertainty on n)" in row
    assert row.rstrip().endswith("not the uncertainty on n)")
    assert "±" not in row


def test_entropy_rln_verdict_rows():
    unmatched = {"probe": "heatcapacity",
                 "entropy_rln_suggestion": {"j": 0.5, "value": 5.76, "label": "R ln2",
                                            "distance": 6.49, "rel_err": 1.127,
                                            "matched": False, "tol": 0.25}}
    rows = dict(flatten_rows(unmatched))
    assert rows["Rln suggestion"] == "R ln2 (NOT matched — S_mag saturation is 113% away)"
    matched = {"probe": "heatcapacity",
               "entropy_rln_suggestion": {"j": 1.0, "value": 9.13, "label": "R ln3",
                                          "distance": 0.18, "rel_err": 0.02,
                                          "matched": True, "tol": 0.25}}
    rows = dict(flatten_rows(matched))
    assert rows["Rln suggestion"] == "R ln3 (matched, 2% off)"
