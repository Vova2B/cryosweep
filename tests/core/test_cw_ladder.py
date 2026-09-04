"""CW fit-window ladder (spec 2026-08-10 §1). Pure-array tests + does-not-cry-wolf fixture."""
import numpy as np
import pytest

from cryosweep_core.fitting.models import CurieWeissModel, fit_cw_ladder


def _exact_cw(theta=-40.0, C=1.5, n=150):
    T = np.linspace(5.0, 300.0, n)
    return T, (T - theta) / C


def test_exact_cw_every_rung_recovers_theta_and_no_flag():
    T, ic = _exact_cw()
    primary, ladder, th_spread, mu_spread = fit_cw_ladder(T, ic)
    full = CurieWeissModel().fit(T, ic)
    # U1: primary params byte-identical to the plain full-window fit
    assert primary.params == full.params
    assert primary.sigma == full.sigma
    assert primary.r2 == full.r2
    for e in ladder:
        assert e["theta_k"] == pytest.approx(-40.0, abs=1e-4)   # >= 4 dp recovery
    # does-not-cry-wolf: spread is convergence noise, << the 2.0 K floor; NOT pinned
    assert th_spread is not None and th_spread < 2.0
    assert "window_sensitive" not in primary.quality_flags
    assert primary.quality_flags == full.quality_flags


def test_ladder_shape_contract():
    T, ic = _exact_cw()
    _, ladder, _, _ = fit_cw_ladder(T, ic)
    keys = {"tmin_k", "theta_k", "sigma_theta_k", "mu_eff", "sigma_mu_eff", "r2", "n_points"}
    assert ladder, "at least the 25/50/100/150/200 rungs fit on a 5-300 K exact grid"
    for e in ladder:
        assert set(e) == keys
        assert all(np.isfinite(v) for k, v in e.items() if k != "n_points")
    assert [e["tmin_k"] for e in ladder] == sorted(e["tmin_k"] for e in ladder)
    # margin rule: no rung at cutoff >= T.max() - 20 (here 280) — all five fit on this grid
    assert [e["tmin_k"] for e in ladder] == [25.0, 50.0, 100.0, 150.0, 200.0]


def test_contaminated_curve_fires_window_sensitive():
    # CW above 60 K, ordered/curved below: mimics the measured MPMS shape
    T = np.linspace(5.0, 300.0, 200)
    ic = (T + 42.0) / 1.5
    low = T < 60.0
    ic[low] = ic[low] - 8.0 * (60.0 - T[low]) / 60.0     # low-T pull, ~the 8 K theta bias
    primary, ladder, th_spread, _ = fit_cw_ladder(T, ic)
    assert th_spread is not None and th_spread > 2.0
    assert "window_sensitive" in primary.quality_flags


import dataclasses
import json
import pathlib

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry

VSM_SYNTH = pathlib.Path("tests/core/fixtures/vsm_synth.dat")

# Same values as tests/core/test_mpms_realdata.py (molar computed from formula; mass PROVISIONAL)
_MOLAR, _MASS_MG = 683.22, 12.0


def _analyze(path, **cfg_overrides):
    cfg = RunConfig.load()
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    return analyze_file(load_dat(str(path)), cfg, build_default_registry())


@pytest.fixture
def mpms_real(mpms_real_path):
    """The analyzed real MPMS result, masses patched — same loading path as
    tests/core/test_mpms_realdata.py (skips off the owner machine via require_real)."""
    rt = load_dat(str(mpms_real_path))
    rt2 = dataclasses.replace(
        rt, header=dataclasses.replace(rt.header, molar_mass=_MOLAR, mass_mg=_MASS_MG))
    return analyze_file(rt2, RunConfig.load(unit_system="CGS"), build_default_registry())


def test_vsm_synth_gains_additive_ladder_fields_and_stays_ok():
    r = _analyze(VSM_SYNTH)
    assert r.status == "ok"                                  # unchanged in THIS task
    d = r.data
    assert "cw_ladder" in d and "theta_spread_k" in d and "mu_eff_spread" in d
    # additive-only: fit params identical to a direct model fit is pinned by existing
    # tests; here pin JSON-safety + determinism of the new fields
    json.dumps(d, allow_nan=False)
    r2 = _analyze(VSM_SYNTH)
    assert json.dumps(r.data, sort_keys=True) == json.dumps(r2.data, sort_keys=True)


def test_real_mpms_ladder_oracles(mpms_real):
    r = mpms_real
    d = r.data
    # full fit byte-identical (spec §8): re-measured through the shipped path, pinned 4 s.f.
    assert d["fit"]["params"]["theta"] == pytest.approx(-50.27, abs=0.01)
    assert d["fit"]["r2"] == pytest.approx(0.996515, abs=1e-5)
    rung25 = next(e for e in d["cw_ladder"] if e["tmin_k"] == 25.0)
    assert rung25["theta_k"] == pytest.approx(-42.24, abs=0.01)
    assert d["theta_spread_k"] == pytest.approx(12.72, abs=0.05)
    assert "window_sensitive" in d["fit"]["quality_flags"]
    assert any("extends below |theta|" in w for w in r.warnings)   # T_min 3.0 < 50.3


def test_short_grid_yields_none_spread_not_zero():
    T = np.linspace(5.0, 40.0, 60)                        # only the 25 K rung can fit,
    ic = (T + 40.0) / 1.5                                 # and 25 >= 40-20 -> skipped too
    primary, ladder, th_spread, mu_spread = fit_cw_ladder(T, ic)
    assert ladder == []
    assert th_spread is None and mu_spread is None        # U2: None, NEVER 0.0
    assert "window_sensitive" not in primary.quality_flags


def test_confidence_clamps_and_penalizes():
    # unit-level: drive the confidence law through the analyzer on the synth file by
    # asserting the published parts; edge values through direct computation
    r = _analyze(VSM_SYNTH)
    assert 0.0 <= r.confidence <= 1.0
    assert r.confidence_parts["fit"] == r.data["fit"]["r2"]      # raw r2 kept (transparency)


def test_confidence_edge_r2_zero_is_zero_not_half():
    # exactly-r2=0 must NOT outrank r2=0.3 (the falsy-0.0 defect). Construct via the law
    # itself — mirrored here so a silent formula change fails loudly:
    from cryosweep_core.analyzers.mag import _cw_confidence
    class _F:  # minimal stand-in with the two consumed attrs
        def __init__(self, r2, flags): self.r2, self.quality_flags = r2, flags
    assert _cw_confidence(_F(0.0, [])) == 0.0
    assert _cw_confidence(_F(-2.0, [])) == 0.0               # negative r2 clamps, not negative conf
    assert _cw_confidence(_F(None, [])) == 0.0
    assert _cw_confidence(_F(0.9965, ["window_sensitive"])) == pytest.approx(0.49825)
    assert _cw_confidence(_F(0.9965, [])) == pytest.approx(0.9965)


def test_real_mpms_flips_to_low_confidence(mpms_real):
    assert mpms_real.status == "low_confidence"              # closed O1: accepted flip
    assert mpms_real.confidence == pytest.approx(0.498, abs=0.002)


def test_window_sensitive_threshold_is_strictly_greater(monkeypatch):
    """F12 (final-review): mutation M15 (`>` -> `>=` at models.py:114) SURVIVED. Float
    equality on `theta_spread == max(3 sigma, floor)` is measure-zero on real data, so
    nothing could hit it by accident — but the rule "spread must EXCEED the floor, not
    merely reach it" is a deliberate choice and was pinned by nothing. Driven here by
    setting the floor to the measured spread exactly."""
    import cryosweep_core.fitting.models as M
    T, ic = _exact_cw()
    _, _, spread, _ = fit_cw_ladder(T, ic)
    assert spread is not None
    # floor set EXACTLY at the spread: strict > must NOT fire
    monkeypatch.setattr(M, "_CW_SPREAD_FLOOR_K", spread)
    primary, _, _, _ = fit_cw_ladder(T, ic)
    assert "window_sensitive" not in primary.quality_flags
    # a hair below it must fire — the threshold is live, not dead
    monkeypatch.setattr(M, "_CW_SPREAD_FLOOR_K", spread * (1 - 1e-9))
    primary, _, _, _ = fit_cw_ladder(T, ic)
    assert "window_sensitive" in primary.quality_flags
