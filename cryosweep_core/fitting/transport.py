from __future__ import annotations
import numpy as np
from scipy.stats import linregress
from scipy.optimize import curve_fit
from cryosweep_core.result import FitResult


class LinearFitModel:
    """Generic linear fit y = slope*x + intercept (slope/intercept/r^2 + sigma)."""
    key = "linear"
    params = ["slope", "intercept"]

    def fit(self, x, y, xunit="K", yunit="Ohm*cm") -> FitResult:
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 2 or np.ptp(x) == 0:
            raise ValueError("linear fit needs >=2 points with distinct x")
        r = linregress(x, y)
        return FitResult(
            model="linear",
            params={"slope": float(r.slope), "intercept": float(r.intercept)},
            sigma={"slope": float(r.stderr), "intercept": float(r.intercept_stderr)},
            covariance=[], r2=float(r.rvalue ** 2), n_points=int(x.size),
            fit_range=[float(x.min()), float(x.max())],
            units={"slope": f"{yunit}/{xunit}", "intercept": yunit},
            quality_flags=[])


def _power_law(T, rho0, A, n):
    return rho0 + A * np.power(T, n)


class PowerLawRhoModel:
    """Resistivity power law rho = rho0 + A*T^n (n~2 Fermi liquid, n~5 phonon)."""
    key = "power_law_rho"
    params = ["rho0", "A", "n"]

    def fit(self, T, rho, yunit="Ohm*cm") -> FitResult:
        T = np.asarray(T, float); rho = np.asarray(rho, float)
        m = np.isfinite(T) & np.isfinite(rho) & (T > 0) & (rho > 0)
        T, rho = T[m], rho[m]
        if T.size < 4:
            raise ValueError("power-law fit needs >=4 physical points")
        # Scale-robustness: curve_fit (Levenberg-Marquardt) is not scale-invariant, so
        # geometry-recomputed rho at a small absolute magnitude can trap the optimizer at
        # its initial guess. Fit on rho normalized by its mean, then rescale rho0 and A
        # back (n and r2 are scale-invariant).
        s = float(np.mean(rho))
        if not (s > 0):
            raise ValueError("power-law fit needs positive-mean rho")
        y = rho / s
        span = max(y.max() - y.min(), 1e-30)
        p0 = [float(y.min()), float(span / (T.max() ** 2)), 2.0]
        popt, pcov = curve_fit(_power_law, T, y, p0=p0, maxfev=20000,
                               bounds=([0.0, 0.0, 0.5], [np.inf, np.inf, 6.0]))
        resid = y - _power_law(T, *popt)
        ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        sig = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
        n_val = float(popt[2])
        rho0_val = float(popt[0] * s)
        rho_min = float(np.min(rho))                 # rho is the masked, physical (>0) array
        sig_n = float(sig[2])
        flags = []
        if abs(n_val - 0.5) < 1e-2 or abs(n_val - 6.0) < 1e-2:
            flags.append("n_at_bound")
        if np.isfinite(sig_n) and abs(n_val) > 0 and sig_n >= abs(n_val):
            # sigma_n swamps n itself -> the exponent is not resolved by this window. Same
            # "pinned/unresolved is not a measurement" idea as rho0_unresolved below, and the
            # same decline discipline as the TTO kappa_ph fit (CLAUDE.md, thermal.py C4).
            # Measured motivation: the SC example's 9-30 K window gives n = 0.618 +/- 1.969
            # at r2 = 0.43 and, before this flag, reported it as a CLEAN fit.
            flags.append("n_unresolved")
        if rho0_val < 1e-3 * rho_min:                # rho0 pinned to ~0 -> residual not resolved
            flags.append("rho0_unresolved")
        return FitResult(
            model="power_law_rho",
            params={"rho0": float(popt[0] * s), "A": float(popt[1] * s), "n": float(popt[2])},
            sigma={"rho0": float(sig[0] * s), "A": float(sig[1] * s), "n": float(sig[2])},
            covariance=[], r2=float(r2), n_points=int(T.size),
            fit_range=[float(T.min()), float(T.max())],
            units={"rho0": yunit, "A": f"{yunit}/K^n", "n": ""},
            quality_flags=flags)


class RhoT2FermiLiquidModel:
    """Forced-n=2 Fermi-liquid fit rho = rho0 + beta*T^2 via linregress on (T^2, rho).
    Distinct from LinearFitModel (which is linear in T, not T^2). fit_range is stored in K
    (not K^2) so renderers can reconstruct the fit window on either a T or T^2 axis."""
    key = "rho_t2_linear"
    params = ["rho0", "beta"]

    def fit(self, T, rho, yunit="Ohm*cm") -> FitResult:
        T = np.asarray(T, float); rho = np.asarray(rho, float)
        m = np.isfinite(T) & np.isfinite(rho) & (T > 0) & (rho > 0)
        T, rho = T[m], rho[m]
        if T.size < 2 or np.ptp(T) == 0:
            raise ValueError("rho-T^2 fit needs >=2 physical points with distinct T")
        r = linregress(T ** 2, rho)
        return FitResult(
            model="rho_t2_linear",
            params={"rho0": float(r.intercept), "beta": float(r.slope)},
            sigma={"rho0": float(r.intercept_stderr), "beta": float(r.stderr)},
            covariance=[], r2=float(r.rvalue ** 2), n_points=int(T.size),
            fit_range=[float(T.min()), float(T.max())],
            units={"rho0": yunit, "beta": f"{yunit}/K^2"},
            quality_flags=[])


POWER_LAW_DECLINE_FLAGS = frozenset({"n_at_bound", "n_unresolved"})
"""Flags that make the EXPONENT not a measurement: it is a search bound, or its sigma swamps
it. Every surface (annotation, fit line, CSV cells, GUI row) declines on these and shows the
reason instead - the TTO kappa_ph rule, applied to resistivity.

Deliberately does NOT include `rho0_unresolved`, which says the RESIDUAL is pinned to ~0 and
leaves n a real (if window-sensitive) measurement: the QD example file carries it with
n = 0.649 +- 0.247 at r2 = 0.957, and a test pins that n still reaches the CSV beside its
spread and flag. Only the fit LINE and rho0 itself are withheld there."""

NO_FIT_LINE_FLAGS = POWER_LAW_DECLINE_FLAGS | {"rho0_unresolved"}
"""Fit line suppressed: either the exponent is not a measurement, or rho0 is unresolved so
the drawn curve would not be the one the parameters describe."""

_RHO_LADDER_RUNGS = (10.0, 15.0, 20.0, 30.0)
_RHO_LADDER_MIN_PTS = 4
_N_SPREAD_FLOOR = 0.05   # reused verbatim from thermal.py C4: the smallest exponent drift
                         # worth a reader's attention, and load-bearing armor against
                         # convergence-noise flag inversion on exact fixtures (measured
                         # there: spread 7.37e-9 vs 3-sigma 3.87e-9 would fire without it)


def fit_rho_powerlaw_ladder(T, rho, yunit="Ohm*cm",
                            rungs=_RHO_LADDER_RUNGS, primary=30.0):
    """(primary FitResult with ladder flags attached, ladder, n_spread) — spec 2026-08-10 §3,
    structural port of fit_kappa_ph_ladder (thermal.py:105).

    Measured motivation: dc rho UTiHx ladder 10->30 K moves n by 0.269 (~20 sigma of the
    shipped pcov sigma 0.013); ACT ch2 spread 0.772; QD example ch1 2.456 — every rung
    r2 >= 0.936, and the physics reading (n~3 phonon vs n~0.6 sub-linear) flips on the
    cutoff alone. The PRIMARY is the existing shipped <=30 K fit (byte-identical, U1).
    No log-log method rung: additive rho0 breaks the transform (deferred §10).
    Raises only what the primary fit raises; non-primary rung failures are omitted."""
    T = np.asarray(T, float); rho = np.asarray(rho, float)
    model = PowerLawRhoModel()
    mp = np.isfinite(T) & np.isfinite(rho) & (T <= primary)
    primary_res = model.fit(T[mp], rho[mp], yunit=yunit)   # may raise -> caller declines
    # NB: unlike fit_kappa_ph_ladder there is NO cut>tmax skip rule — the analyzer hands in
    # the already-<=30 K-masked arrays, whose tmax is typically just under 30 K, and the
    # spec §8 oracles (dc rho spread 0.269, ACT 0.772) INCLUDE the 30 K primary rung.
    ladder: list[dict] = []
    for cut in sorted(rungs):
        m = np.isfinite(T) & np.isfinite(rho) & (T <= cut)
        if int(m.sum()) < _RHO_LADDER_MIN_PTS:
            continue
        try:
            fr = primary_res if cut == primary else model.fit(T[m], rho[m], yunit=yunit)
        except (ValueError, RuntimeError):
            continue
        n_i, s_i, r2_i = fr.params["n"], fr.sigma["n"], fr.r2
        if not all(np.isfinite(float(v)) for v in (n_i, s_i, r2_i)):
            continue                                       # a non-finite rung is a failed rung
        ladder.append({"cutoff_k": float(cut), "n": float(n_i), "sigma": float(s_i),
                       "r2": float(r2_i), "n_points": int(fr.n_points),
                       "at_bound": bool("n_at_bound" in fr.quality_flags)})
    # A rung pinned at a search bound is not a measurement, so it must not enter the spread:
    # several rungs pinned at the SAME bound agree exactly and fake a window-stable exponent.
    # Measured on the SC example (onset 8.8 K): rungs 10/15/20 K all pin at 0.5, so the old
    # all-rung spread was 0.118 and window_sensitive stayed silent while n truly ran 0.5->0.99.
    ns = [e["n"] for e in ladder if not e["at_bound"]]
    n_spread = float(max(ns) - min(ns)) if len(ns) >= 2 else None       # None, NEVER 0.0
    flags = list(primary_res.quality_flags)
    if n_spread is None and "ladder_incomplete" not in flags:
        flags.append("ladder_incomplete")   # fewer than two RESOLVED rungs -> no spread claim
    if (n_spread is not None
            and n_spread > max(3.0 * float(primary_res.sigma["n"]), _N_SPREAD_FLOOR)):
        flags.append("window_sensitive")
    if flags != list(primary_res.quality_flags):           # FitResult is frozen: copy
        primary_res = primary_res.model_copy(update={"quality_flags": flags})
    return primary_res, ladder, n_spread


# --------------------------- activated transport (Arrhenius) ---------------------------

KB_MEV_PER_K = 8.617333262e-2   # Boltzmann constant, meV/K (CODATA exact)

ARRHENIUS_SPREAD_FLOOR_MEV = 1.0
"""window_sensitive floor for the E_a ladder spread (meV): the smallest activation-energy
drift worth a reader's attention. Deliberately loose, NOT finely tuned — the verdict on all
three validation datasets is identical for any floor from 0.5 to 2 meV (pinned by test); if
that ever narrows, the noise/physics separation has collapsed and the rule needs rethinking,
not retuning (the 1/chi-guard convention)."""

ARRHENIUS_DECLINE_FLAGS = frozenset({"insufficient_rho_span", "ea_unresolved"})
"""Flags that make E_a not a measurement. `insufficient_rho_span`: under one e-fold of rho
change an exponential deviates from its chord by < ~12% — indistinguishable from a straight
line at instrument scatter (measured on the corpus' ONE real insulating channel: x1.3 change
over 3-340 K gives r2 = 0.10 and E_a drifting 0.054 -> 7.96 meV purely with the window).
`ea_unresolved`: sigma swamps the slope. Every surface (fit line, CSV cells, GUI row,
capability) declines on these and shows the reason instead.

`window_sensitive` deliberately does NOT decline: on non-Arrhenius insulators (Mott VRH) the
fit still measures a slope on every window — the drift across windows IS the finding, and
hiding the number would hide it."""


def _arrhenius_one(T, rho):
    """OLS of ln rho vs 1/T -> E_a in meV. Reports the ACTIVATION ENERGY as measured and
    never silently converts to a gap: for intrinsic conduction rho ~ exp(+E_g/2k_BT) so
    E_g = 2*E_a, but for extrinsic conduction the activation is a donor/acceptor level and
    the factor is 1 (or 1/2 under compensation) — TRANSPORT ALONE CANNOT TELL THE REGIMES
    APART. The only gap field is therefore named `e_g_assuming_intrinsic_mev`, so the
    assumption travels in the name everywhere the number goes."""
    T = np.asarray(T, float); rho = np.asarray(rho, float)
    m = np.isfinite(T) & np.isfinite(rho) & (T > 0) & (rho > 0)
    T, rho = T[m], rho[m]
    if T.size < 4 or float(np.ptp(T)) == 0.0:
        raise ValueError("arrhenius fit needs >= 4 points spanning a temperature range")
    y = np.log(rho)
    r = linregress(1.0 / T, y)
    ea = KB_MEV_PER_K * float(r.slope)
    sigma = KB_MEV_PER_K * float(r.stderr)
    flags: list[str] = []
    if float(np.ptp(y)) < 1.0:
        flags.append("insufficient_rho_span")
    if sigma >= abs(ea):
        flags.append("ea_unresolved")
    return FitResult(
        model="arrhenius_rho",
        params={"e_a_mev": ea, "ln_rho0": float(r.intercept),
                "e_g_assuming_intrinsic_mev": 2.0 * ea},
        sigma={"e_a_mev": sigma, "e_g_assuming_intrinsic_mev": 2.0 * sigma},
        covariance=[], r2=float(r.rvalue ** 2), n_points=int(T.size),
        fit_range=[float(T.min()), float(T.max())],
        units={"e_a_mev": "meV", "ln_rho0": "", "e_g_assuming_intrinsic_mev": "meV"},
        quality_flags=flags)


def fit_arrhenius_ladder(T, rho):
    """(primary FitResult with ladder flags attached, ladder, ea_spread_mev, alt_models).

    Ladder rungs at T >= quantile {0, 25, 50, 75}% of the fitted window — activation is a
    whole-window statement, so rungs shrink from the low-T side (where extrinsic freeze-out
    and VRH curvature live). Rungs carrying a decline flag are excluded from the spread (a
    non-measurement cannot vouch for window stability). `window_sensitive` fires when the
    spread exceeds max(3 sigma, ARRHENIUS_SPREAD_FLOOR_MEV).

    alt_models: r2 of ln rho against 1/T (Arrhenius), T^-1/4 (Mott 3D VRH) and T^-1/2
    (Efros-Shklovskii) on the SAME window — reported with the note that r2 over one window
    cannot select a conduction mechanism, precisely so nobody can pretend it can."""
    T = np.asarray(T, float); rho = np.asarray(rho, float)
    primary = _arrhenius_one(T, rho)
    ladder: list[dict] = []
    eas: list[float] = []
    for q in (0.0, 0.25, 0.50, 0.75):
        tmin = float(np.quantile(T, q))
        m = T >= tmin
        try:
            fr = primary if q == 0.0 else _arrhenius_one(T[m], rho[m])
        except ValueError:
            continue
        ladder.append({"t_min_k": float(fr.fit_range[0]), "n_points": int(fr.n_points),
                       "e_a_mev": float(fr.params["e_a_mev"]),
                       "sigma_mev": float(fr.sigma["e_a_mev"]),
                       "r2": float(fr.r2), "flags": list(fr.quality_flags)})
        if not fr.quality_flags:
            eas.append(float(fr.params["e_a_mev"]))
    spread = float(max(eas) - min(eas)) if len(eas) >= 2 else None      # None, NEVER 0.0
    flags = list(primary.quality_flags)
    declined = bool(set(flags) & ARRHENIUS_DECLINE_FLAGS)
    if spread is None and not declined:
        flags.append("ladder_incomplete")
    if (spread is not None
            and spread > max(3.0 * float(primary.sigma["e_a_mev"]),
                             ARRHENIUS_SPREAD_FLOOR_MEV)):
        flags.append("window_sensitive")
    if flags != list(primary.quality_flags):               # FitResult is frozen: copy
        primary = primary.model_copy(update={"quality_flags": flags})
    y = np.log(rho[np.isfinite(rho) & (rho > 0) & np.isfinite(T) & (T > 0)])
    Tm = T[np.isfinite(rho) & (rho > 0) & np.isfinite(T) & (T > 0)]
    alt = {"models": [], "note": ("r2 over one temperature window cannot select a conduction "
                                  "mechanism: Arrhenius, Mott VRH and Efros-Shklovskii all "
                                  "fit a smooth insulating rho(T) comparably well there; "
                                  "discrimination needs a wide range or an independent probe "
                                  "(Hall, thermopower).")}
    for name, x in (("arrhenius", 1.0 / Tm), ("mott_vrh_3d", Tm ** -0.25),
                    ("efros_shklovskii", Tm ** -0.5)):
        rr = linregress(x, y)
        alt["models"].append({"model": name, "r2": float(rr.rvalue ** 2)})
    return primary, ladder, spread, alt
