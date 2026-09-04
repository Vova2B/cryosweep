"""Resistivity power-law cutoff ladder (2026-08-10 spec §3) — structural port of
fit_kappa_ph_ladder. Pure-array tests; analyzer wiring + O7 tests appended by Task 7."""
import numpy as np
import pytest

from cryosweep_core.fitting.transport import PowerLawRhoModel, fit_rho_powerlaw_ladder


def _exact_grid(n=200):
    T = np.linspace(2.0, 35.0, n)
    return T, 1e-4 + 2e-6 * T ** 2


def test_exact_powerlaw_every_rung_recovers_n2_and_no_flag():
    T, rho = _exact_grid()
    primary, ladder, n_spread = fit_rho_powerlaw_ladder(T, rho)
    # U1 byte-identity: primary IS the shipped <=30 K fit
    direct = PowerLawRhoModel().fit(T[T <= 30.0], rho[T <= 30.0])
    assert primary.params == direct.params
    assert primary.sigma == direct.sigma
    assert primary.r2 == direct.r2
    for e in ladder:
        assert e["n"] == pytest.approx(2.0, abs=1e-3)     # >= 3 dp recovery
    assert n_spread is not None and n_spread < 0.05
    assert "window_sensitive" not in primary.quality_flags


def test_two_regime_curve_fires_window_sensitive():
    # n = 3 below 15 K, sub-linear above (continuous): the physics reading flips on cutoff
    T = np.linspace(2.0, 35.0, 220)
    rho = np.where(T <= 15.0,
                   1e-4 + 5e-7 * T ** 3,
                   1e-4 + 5e-7 * 15.0 ** 3 + 2.5e-4 * (np.maximum(T - 15.0, 0.0)) ** 0.6)
    primary, ladder, n_spread = fit_rho_powerlaw_ladder(T, rho)
    assert n_spread is not None and n_spread > 0.05
    assert "window_sensitive" in primary.quality_flags


def test_short_grid_single_rung_yields_none_spread_no_flag():
    # 5 points at 22-30 K: T <= 20 has < 4 points, so ONLY the 30 K (primary) rung fits
    T = np.linspace(22.0, 30.0, 5)
    rho = 1e-4 + 2e-6 * T ** 2
    primary, ladder, n_spread = fit_rho_powerlaw_ladder(T, rho)
    assert len(ladder) == 1 and ladder[0]["cutoff_k"] == 30.0
    assert n_spread is None                     # U2: None, NEVER 0.0
    assert "window_sensitive" not in primary.quality_flags


def test_ladder_key_set_and_ordering():
    T, rho = _exact_grid()
    _, ladder, _ = fit_rho_powerlaw_ladder(T, rho)
    # `at_bound` appended 2026-09-01: a rung pinned at a search bound is not a measurement
    # and is excluded from n_spread, so each rung must carry whether it was.
    keys = {"cutoff_k", "n", "sigma", "r2", "n_points", "at_bound"}
    assert [e["cutoff_k"] for e in ladder] == [10.0, 15.0, 20.0, 30.0]
    for e in ladder:
        assert set(e) == keys
        assert isinstance(e["at_bound"], bool)
        assert all(np.isfinite(v) for k, v in e.items() if k != "at_bound")


# ---------------- Task 7: analyzer wiring + closed O7 -----------------------
import json
import pathlib

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.analyzers.resistivity import BridgeResult, _capabilities, _HONESTY_FLAGS
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry
from cryosweep_core.result import FitResult

FIX = pathlib.Path("tests/core/fixtures")


def _analyze(path):
    return analyze_file(load_dat(str(path)), RunConfig.load(), build_default_registry())


def _fit_with_flags(flags):
    return FitResult(model="power_law_rho", params={"rho0": 1e-6, "A": 1e-8, "n": 2.0},
                     sigma={"rho0": 0.0, "A": 0.0, "n": 0.01}, covariance=[], r2=0.99,
                     n_points=20, fit_range=[2.0, 30.0], units={}, quality_flags=flags)


def test_analyzer_gains_additive_ladder_fields_json_safe_deterministic():
    r = _analyze(FIX / "rho_sc_synth.dat")
    bridges = r.data["bridges"]
    for b in bridges:
        assert "power_law_ladder" in b and "power_law_n_spread" in b
    json.dumps(r.data, allow_nan=False)
    r2 = _analyze(FIX / "rho_sc_synth.dat")
    assert json.dumps(r.data, sort_keys=True) == json.dumps(r2.data, sort_keys=True)


def test_o7_window_sensitive_does_not_revoke_power_law_fit():
    def cap(flags):
        br = BridgeResult(channel=1, rho_source="instrument_column",
                          power_law=_fit_with_flags(flags))
        caps = {c.name: c for c in _capabilities([br], [], [])}
        return caps["power_law_fit"].applicable
    assert cap(["window_sensitive"]) is True      # honesty annotation -> capability stays
    assert cap(["rho0_unresolved"]) is False      # a real fit-quality flag still revokes
    assert "window_sensitive" in _HONESTY_FLAGS


def test_real_dc_rho_ladder_oracles(dc_rho_path):
    r = _analyze(dc_rho_path)
    br = next(b for b in r.data["bridges"] if b["power_law"] is not None)
    # shipped power_law byte-identical: n = 0.769 +- 0.013 (3 s.f., re-measured)
    assert br["power_law"]["params"]["n"] == pytest.approx(0.769, abs=0.001)
    assert br["power_law"]["sigma"]["n"] == pytest.approx(0.013, abs=0.001)
    # Re-measured 2026-09-01. Was 0.269, taken from the 10 K rung — but that rung is pinned
    # at the n = 0.5 search BOUND (n = 0.5000 exactly, 166 pts, r2 = 0.957), so 0.269 was the
    # distance to a bound, not a measured drift. Bound-pinned rungs no longer enter the
    # spread: the resolved rungs run 0.5729 (15 K) -> 0.7686 (30 K) = 0.1957. The physics
    # reading is unchanged — the spread is still ~15x the pcov sigma 0.0131 and still fires
    # window_sensitive; only the overstated magnitude is gone. The 10 K rung is still IN the
    # ladder, carrying at_bound=True, so nothing is hidden from the reader.
    assert br["power_law_n_spread"] == pytest.approx(0.1957, abs=0.005)
    assert [e["at_bound"] for e in br["power_law_ladder"]] == [True, False, False, False]
    assert "window_sensitive" in br["power_law"]["quality_flags"]
    caps = {c["name"]: c for c in r.data["capabilities"]}
    assert caps["power_law_fit"]["applicable"] is True     # closed O7
    json.dumps(r.data, allow_nan=False)


def test_real_act_ladder_oracles(act_real_path):
    r = _analyze(act_real_path)
    br = next(b for b in r.data["bridges"]
              if b["channel"] == 2 and b["power_law"] is not None)
    # rungs 10-30 K only (the ladder deliberately omits the spec table's 50 K rung):
    # spread = 3.667 - 2.895 ~= 0.772, re-measured through the shipped path
    assert br["power_law_n_spread"] == pytest.approx(0.772, abs=0.01)
    assert "window_sensitive" in br["power_law"]["quality_flags"]
    caps = {c["name"]: c for c in r.data["capabilities"]}
    assert caps["power_law_fit"]["applicable"] is True     # closed O7


def test_window_sensitive_threshold_is_strictly_greater(monkeypatch):
    """F12 (final-review): the sibling of models.py:114 at transport.py:147 — neither
    boundary had a test. `>` (exceed) not `>=` (reach) is the deliberate rule."""
    import cryosweep_core.fitting.transport as TR
    T, rho = _exact_grid()
    _, _, spread = fit_rho_powerlaw_ladder(T, rho)
    assert spread is not None
    monkeypatch.setattr(TR, "_N_SPREAD_FLOOR", spread)
    primary, _, _ = fit_rho_powerlaw_ladder(T, rho)
    assert "window_sensitive" not in primary.quality_flags
    monkeypatch.setattr(TR, "_N_SPREAD_FLOOR", spread * (1 - 1e-9))
    primary, _, _ = fit_rho_powerlaw_ladder(T, rho)
    assert "window_sensitive" in primary.quality_flags
