"""kappa_ph = B*T^n window-ladder fit (spec §1). Pure-array tests: no .dat, no analyzer.

The oracles here are computed on the SAME pinned grid the Task-3 fixture uses
(np.linspace(30.0, 2.0, 150), kappa_ph = 1.0e-3 * T**3), so an n that is right here and wrong
on the fixture means the analyzer wiring is at fault, not the fit."""
import numpy as np
import pytest

from cryosweep_core.fitting.thermal import KappaPhPowerModel, fit_kappa_ph_ladder

L0 = 2.443e-8


def _exact_grid():
    """The pinned fixture grid: kappa_ph is an EXACT power law with n = 3.0, B = 1e-3."""
    T = np.linspace(30.0, 2.0, 150)
    return T, 1.0e-3 * T ** 3


def test_model_identity_and_param_names():
    m = KappaPhPowerModel()
    assert m.key == "kappa_ph_power"
    assert m.params == ["B", "n"]


def test_single_window_fit_recovers_the_exact_exponent_and_prefactor():
    T, kph = _exact_grid()
    fr = KappaPhPowerModel().fit(T, kph, cutoff=10.0)
    assert fr.model == "kappa_ph_power"
    assert fr.params["n"] == pytest.approx(3.0, abs=1e-6)
    assert fr.params["B"] == pytest.approx(1.0e-3, rel=1e-6)
    assert fr.r2 == pytest.approx(1.0, abs=1e-7)
    assert fr.n_points == 43                       # 43 grid points at T <= 10 K
    assert fr.fit_range == [pytest.approx(2.0), 10.0]
    assert fr.units == {"B": "W/(K^(1+n) m)", "n": ""}
    assert set(fr.sigma) == {"B", "n"}


def test_fit_is_scale_invariant_in_n_because_y_is_normalised():
    # PowerLawRhoModel's reason, re-pinned: curve_fit is NOT scale-invariant, so a kappa_ph
    # at a small absolute magnitude would trap the optimizer at p0 without the mean-normalise.
    T, kph = _exact_grid()
    a = KappaPhPowerModel().fit(T, kph, cutoff=10.0)
    b = KappaPhPowerModel().fit(T, kph * 1e-9, cutoff=10.0)
    assert b.params["n"] == pytest.approx(a.params["n"], abs=1e-6)
    assert b.params["B"] == pytest.approx(a.params["B"] * 1e-9, rel=1e-6)


def test_sigma_b_carries_the_same_magnitude_rescale_as_b():
    # sigma_B must be rescaled by `* s` exactly as B is -- otherwise the reported uncertainty
    # belongs to the NORMALISED prefactor and sits next to a physical B, orders of magnitude off.
    # Deliberately NOT the exact grid: there sigma_B is ~5e-16 convergence noise and every
    # comparison falls under pytest.approx's 1e-12 absolute floor, which is vacuous. A 2 %-noise
    # fixture gives a genuine sigma_B ~ 2.9e-5, and the ratios below use abs=0.0.
    rng = np.random.default_rng(7)
    T = np.linspace(30.0, 2.0, 150)
    kph = 1.0e-3 * T ** 3 * (1.0 + 0.02 * rng.standard_normal(T.size))
    a = KappaPhPowerModel().fit(T, kph, cutoff=10.0)
    b = KappaPhPowerModel().fit(T, kph * 1e-9, cutoff=10.0)
    assert a.sigma["B"] > 1e-6                                     # non-vacuous by construction
    assert b.sigma["B"] == pytest.approx(a.sigma["B"] * 1e-9, rel=1e-6, abs=0.0)
    assert b.params["B"] == pytest.approx(a.params["B"] * 1e-9, rel=1e-6, abs=0.0)
    assert b.sigma["n"] == pytest.approx(a.sigma["n"], rel=1e-6, abs=0.0)   # n is scale-free


def test_fit_declines_when_the_prefactor_rescale_overflows():
    # I1 (DELIBERATE DEVIATION from the plan's pinned block): kappa_ph near the float ceiling
    # makes `popt[0] * s` overflow to +inf, which previously escaped as params["B"] = inf,
    # sigma["B"] = inf and NO quality flag -- json.dumps(..., allow_nan=False) then raises
    # downstream. The fit must decline at the source instead.
    T = np.linspace(2.0, 10.0, 40)
    with pytest.raises(ValueError, match="non-finite prefactor"):
        KappaPhPowerModel().fit(T, np.full(T.size, 1.0e307), cutoff=10.0)
    with pytest.raises(ValueError):                # and the ladder propagates the primary decline
        fit_kappa_ph_ladder(T, np.full(T.size, 1.0e307))


def test_fit_raises_below_ten_points():
    T = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    with pytest.raises(ValueError):
        KappaPhPowerModel().fit(T, 1e-3 * T ** 3, cutoff=10.0)


def test_fit_raises_when_every_point_is_non_physical():
    # An all-negative kappa_ph empties the (k > 0) mask, so it is the POINT-COUNT guard that
    # raises, not the positive-mean one -- asserted on the message so the distinction is pinned.
    # The `if not (s > 0)` branch is unreachable through this API (the mask guarantees a
    # positive mean) and is marked defensive-only in the module; it is NOT covered here.
    T = np.linspace(2.0, 10.0, 40)
    with pytest.raises(ValueError, match=r"needs >=10 finite"):
        KappaPhPowerModel().fit(T, np.full(T.size, -1.0), cutoff=10.0)


def test_ladder_shape_is_the_pinned_key_set_and_ordering():
    T, kph = _exact_grid()
    primary, ladder = fit_kappa_ph_ladder(T, kph)
    assert len(ladder) == 5                                     # 4 curve_fit rungs + 1 loglog
    for e in ladder:
        assert set(e) == {"cutoff_k", "method", "n", "sigma", "r2", "n_points"}
        assert isinstance(e["n_points"], int)
        assert all(np.isfinite(e[k]) for k in ("cutoff_k", "n", "sigma", "r2"))
    assert [e["method"] for e in ladder] == ["curve_fit"] * 4 + ["loglog"]
    assert [e["cutoff_k"] for e in ladder] == [10.0, 15.0, 20.0, 30.0, 10.0]
    assert [e["n_points"] for e in ladder[:4]] == [43, 70, 96, 150]


def test_exact_power_law_recovers_n_at_every_rung_and_does_not_cry_wolf():
    # THE point of the 0.05 floor (C4): under a ratio-only rule (n_spread > 3*sigma_n) this
    # fixture measures n_spread 7.37e-9 vs 3*sigma 3.87e-9 -> the flag FIRES on data where n is
    # recovered to 9 decimals. With the floor it cannot.
    T, kph = _exact_grid()
    primary, ladder = fit_kappa_ph_ladder(T, kph)
    for e in ladder:
        assert e["n"] == pytest.approx(3.0, abs=1e-6), e
    assert primary.quality_flags == []
    cf = [e["n"] for e in ladder if e["method"] == "curve_fit"]
    assert (max(cf) - min(cf)) < 0.05


def test_loglog_rung_sits_at_the_primary_cutoff_and_is_excluded_from_the_spread():
    # Non-vacuous, and the claim is MEASURED: on a curved (saturating) kappa_ph the two methods
    # genuinely diverge on the SAME <=10 K window -- curve_fit 2.1256 vs log-log 2.4189 -- so
    # folding the loglog rung into the spread would inflate it 0.7571 -> 1.0504 (+39 %). On the
    # earlier exact-power-law fixture both methods returned 3.0 and this test proved nothing.
    T = np.linspace(30.0, 2.0, 400)
    kph = 1.0e-3 * T ** 3 / (1.0 + (T / 8.0) ** 2)
    primary, ladder = fit_kappa_ph_ladder(T, kph)
    ll = [e for e in ladder if e["method"] == "loglog"]
    assert len(ll) == 1
    assert ll[0]["cutoff_k"] == 10.0
    assert ll[0]["n_points"] == 115
    assert ll[0]["n"] == pytest.approx(2.4189, abs=1e-3)
    cf = [e for e in ladder if e["method"] == "curve_fit"]
    assert len(cf) == 4
    assert len(ladder) == len(cf) + 1
    assert cf[0]["cutoff_k"] == 10.0                      # same window, different method
    assert cf[0]["n"] == pytest.approx(2.1256, abs=1e-3)
    cf_n = [e["n"] for e in cf]
    assert max(cf_n) - min(cf_n) == pytest.approx(0.75711, abs=1e-3)          # curve_fit only
    with_ll = cf_n + [ll[0]["n"]]
    assert max(with_ll) - min(with_ll) == pytest.approx(1.05041, abs=1e-3)    # if it leaked in


def test_the_spread_is_max_minus_min_across_rungs_not_a_standard_deviation():
    # The spread IS the honest error bar on n, so its definition is load-bearing. This fixture
    # separates the two candidate definitions against the 0.05 floor: max - min = 0.0907 (fires
    # window_sensitive), np.std of the same four rungs = 0.0335 (would NOT fire). 3*sigma_n is
    # 0.0016 here, so the floor -- not the ratio -- is the active threshold either way.
    T = np.linspace(30.0, 2.0, 400)
    kph = 1.0e-3 * T ** 3 * np.exp(-0.006 * T)
    primary, ladder = fit_kappa_ph_ladder(T, kph)
    cf = [e["n"] for e in ladder if e["method"] == "curve_fit"]
    assert len(cf) == 4
    assert max(cf) - min(cf) == pytest.approx(0.09067, abs=1e-3)
    assert float(np.std(cf)) == pytest.approx(0.03352, abs=1e-3)
    assert float(np.std(cf)) < 0.05 < max(cf) - min(cf)   # the definitions straddle the floor
    assert 3.0 * primary.sigma["n"] < 0.05
    assert "window_sensitive" in primary.quality_flags


def test_window_sensitive_fires_when_n_genuinely_drifts_across_windows():
    # Two-regime kappa_ph: T^3 below 12 K, flattening above -> the fitted n must fall with the
    # window, by far more than 0.05. This is the gate file's behaviour in miniature.
    T = np.linspace(30.0, 2.0, 400)
    kph = np.where(T <= 12.0, 1.0e-3 * T ** 3, 1.0e-3 * 12.0 ** 3 * (T / 12.0) ** 0.8)
    primary, ladder = fit_kappa_ph_ladder(T, kph)
    cf = [e["n"] for e in ladder if e["method"] == "curve_fit"]
    assert (max(cf) - min(cf)) > 0.05
    assert "window_sensitive" in primary.quality_flags
    assert "ladder_incomplete" not in primary.quality_flags


def test_ladder_incomplete_when_only_one_rung_fits_and_spread_is_never_zero():
    # I1: a file topping out at 12 K fits only the 10 K rung. Emitting n_spread = 0.0 there
    # would assert "n is stable across windows" when it was never measured.
    T = np.linspace(12.0, 2.0, 120)
    primary, ladder = fit_kappa_ph_ladder(T, 1.0e-3 * T ** 3)
    cf = [e for e in ladder if e["method"] == "curve_fit"]
    assert [e["cutoff_k"] for e in cf] == [10.0]
    assert "ladder_incomplete" in primary.quality_flags
    assert "window_sensitive" not in primary.quality_flags


def test_ladder_raises_when_the_primary_rung_declines():
    T = np.linspace(30.0, 20.0, 100)               # nothing at all below 10 K
    with pytest.raises(ValueError):
        fit_kappa_ph_ladder(T, 1.0e-3 * T ** 3)


def test_kappa_e_dominant_is_gated_on_the_median_over_the_primary_window():
    T, kph = _exact_grid()
    quiet = L0 * T / 1.0e-5                        # the Task-3 fixture's kappa_e: median 6.5 %
    loud = 9.0 * kph                               # kappa_e/kappa = 0.9 everywhere
    assert "kappa_e_dominant" not in fit_kappa_ph_ladder(T, kph, kappa_e=quiet)[0].quality_flags
    assert "kappa_e_dominant" in fit_kappa_ph_ladder(T, kph, kappa_e=loud)[0].quality_flags


def test_n_at_bound_flag_fires_on_a_flat_curve():
    # A constant kappa_ph drives n to its 0.5 lower bound; the flag is what says "don't read n".
    T = np.linspace(30.0, 2.0, 150)
    primary, _ = fit_kappa_ph_ladder(T, np.full(T.size, 1.0))
    assert "n_at_bound" in primary.quality_flags


def test_n_at_bound_flag_fires_at_the_upper_bound_too():
    # The UPPER bound is 6.0, not something larger: a T^9 curve cannot be fitted, n is pinned at
    # 6.0 and the flag must say so. Raising the bound would let a fabricated n = 8 through clean.
    T = np.linspace(30.0, 2.0, 150)
    fr = KappaPhPowerModel().fit(T, T ** 9.0, cutoff=10.0)
    assert fr.params["n"] == pytest.approx(6.0, abs=1e-6)
    assert "n_at_bound" in fr.quality_flags


@pytest.mark.parametrize("n_true, flagged", [(0.505, True), (0.55, False),
                                             (5.995, True), (5.9, False)])
def test_n_at_bound_tolerance_is_one_percent_of_an_exponent(n_true, flagged):
    # Pins the 1e-2 tolerance from BOTH sides with exponents that sit near-but-not-at the bound:
    # 0.005 away flags, 0.05 away does not. The flat-curve test alone drives n exactly onto the
    # bound, so any tolerance -- 1e-2 or 1e-9 -- would have passed it.
    T = np.linspace(30.0, 2.0, 150)
    fr = KappaPhPowerModel().fit(T, 1.0e-3 * T ** n_true, cutoff=10.0)
    assert fr.params["n"] == pytest.approx(n_true, abs=1e-6)
    assert ("n_at_bound" in fr.quality_flags) is flagged


def test_degenerate_window_is_flagged_and_n_is_exactly_the_initial_guess():
    # A window with a single distinct T fits NOTHING: curve_fit returns p0 untouched, so n comes
    # back as exactly 2.0 with sigma_n ~ 0 and r2 = 0.0. Consumers must not have to infer that
    # from r2 == 0.0 -- and this also pins p0's n-guess, which is otherwise invisible.
    T = np.concatenate([np.full(20, 5.0), np.linspace(20.0, 30.0, 50)])
    fr = KappaPhPowerModel().fit(T, 1.0e-3 * T ** 3, cutoff=10.0)
    assert fr.n_points == 20
    assert fr.r2 == 0.0
    assert fr.params["n"] == 2.0                   # exactly p0[1]; never fitted
    assert fr.sigma["n"] == pytest.approx(0.0, abs=1e-6)
    assert "degenerate_window" in fr.quality_flags
    # and it survives the ladder merge onto the primary result
    primary, _ = fit_kappa_ph_ladder(T, 1.0e-3 * T ** 3)
    assert "degenerate_window" in primary.quality_flags
