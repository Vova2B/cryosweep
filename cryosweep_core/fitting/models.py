from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from cryosweep_core.result import FitResult

_K = {"CGS": 2.827, "SI": 797.8}
# C = chi*(T-theta). chi_molar (CGS) is emu/(mol*Oe), so C is emu*K/(mol*Oe) — the
# physically-correct string (reconciled with v1 main.py:4455; the old "emu*K/mol" dropped the
# per-Oe). SI chi is m^3/mol -> C is m^3*K/mol. Consumers of the old CGS string were updated.
_C_UNIT = {"CGS": "emu*K/(mol*Oe)", "SI": "m^3*K/mol"}
_CHI0_UNIT = {"CGS": "emu/(mol*Oe)", "SI": "m^3/mol"}

class CurieWeissModel:
    key = "curie_weiss"
    params = ["C", "theta", "mu_eff"]

    def fit(self, T, inv_chi, unit_system="CGS", modified=False) -> FitResult:
        T = np.asarray(T, float); inv_chi = np.asarray(inv_chi, float)
        m = np.isfinite(T) & np.isfinite(inv_chi)
        T, inv_chi = T[m], inv_chi[m]
        if T.size < 3:                       # Bug 6: need >=3 points for a meaningful fit + covariance
            raise ValueError(f"Curie-Weiss fit needs >=3 finite points, got {T.size}")
        k = _K[unit_system]
        if not modified:
            p, cov = np.polyfit(T, inv_chi, 1, cov=True)
            slope, intercept = float(p[0]), float(p[1])
            if slope == 0.0:                 # Bug 2: flat inv_chi -> C = 1/slope would crash
                raise ValueError("Curie-Weiss fit failed: zero slope (1/chi is constant; "
                                 "check for zero-field or non-physical susceptibility)")
            C = 1.0 / slope
            theta = -intercept / slope
            s_sl = float(np.sqrt(cov[0, 0])); s_int = float(np.sqrt(cov[1, 1])); c_si = float(cov[0, 1])
            sC = s_sl / slope ** 2
            sTheta = np.sqrt(s_int ** 2 / slope ** 2 + (intercept ** 2 / slope ** 4) * s_sl ** 2
                             - 2.0 * (intercept / slope ** 3) * c_si)
            # mu_eff = k sqrt(C) has no value for C <= 0 (flagged C_nonpositive by the caller):
            # null, never NaN in the JSON.
            mu_eff = float(k * np.sqrt(C)) if C > 0 else None
            sMu = abs(k / (2.0 * np.sqrt(C)) * sC) if C > 0 else None
            fit_line = (T - theta) / C
            r2 = _r2(inv_chi, fit_line)
            return FitResult(model="curie_weiss", params={"C": C, "theta": theta, "mu_eff": mu_eff},
                             sigma={"C": abs(sC), "theta": float(sTheta), "mu_eff": sMu},
                             covariance=cov.tolist(), r2=r2, n_points=int(T.size),
                             fit_range=[float(T.min()), float(T.max())],
                             units={"C": _C_UNIT[unit_system], "theta": "K", "mu_eff": "mu_B"}, quality_flags=[])
        # modified: chi = chi0 + C/(T - theta)
        # Fitted on chi / median|chi|, then C and chi0 (and their covariance) are scaled back.
        # Without it the fit depended on the unit system: SI chi is ~1e-5 of CGS chi, the
        # optimizer's steps were sized for O(1) parameters, and chi0 stuck at its 0.0 start
        # (measured: theta -173 K in CGS vs -67 K in SI on one file).
        chi = 1.0 / inv_chi
        s = float(np.median(np.abs(chi)))
        s = s if np.isfinite(s) and s > 0 else 1.0
        chi_n = chi / s
        lin = np.polyfit(T, inv_chi * s, 1)
        p0 = [abs(1.0 / lin[0]) if lin[0] else 1.0, min(-lin[1] / lin[0] if lin[0] else 0.0, T.min() - 1.0), 0.0]
        lower = [1e-12, -np.inf, -np.inf]; upper = [np.inf, T.min() - 1e-6, np.inf]
        popt, pcov = curve_fit(lambda t, C, th, c0: c0 + C / (t - th), T, chi_n, p0=p0, bounds=(lower, upper), maxfev=10000)
        scale = np.array([s, 1.0, s])
        c_pinned = bool(popt[0] <= _MOD_C_PINNED_N)          # normalised units, before rescale
        popt = popt * scale
        pcov = pcov * np.outer(scale, scale)
        C, theta, chi0 = map(float, popt)
        mu_eff = k * np.sqrt(C)
        sig = np.sqrt(np.diag(pcov))
        fit_line = 1.0 / (chi0 + C / (T - theta))
        # Which parameters are measurements. Values stay here (the caller still needs the
        # fit's own prediction for its curve); decline_modified() nulls the flagged ones.
        ub = float(T.min()) - 1e-6
        at_bound = {"C": c_pinned,
                    "theta": (ub - theta) <= 1e-6 * max(1.0, abs(ub)) + 1e-6,
                    "chi0": False}
        flags = []
        for name, v, sg in zip(("C", "theta", "chi0"), (C, theta, chi0), sig):
            if at_bound[name]:
                flags.append(f"{name}_at_bound")
            elif not np.isfinite(sg) or sg == 0.0 or sg >= abs(v):
                flags.append(f"{name}_unresolved")
        return FitResult(model="curie_weiss_modified",
                         params={"C": C, "theta": theta, "chi0": chi0, "mu_eff": mu_eff},
                         sigma={"C": float(sig[0]), "theta": float(sig[1]), "chi0": float(sig[2]),
                                "mu_eff": float(k / (2 * np.sqrt(C)) * sig[0])},
                         covariance=pcov.tolist(), r2=_r2(inv_chi, fit_line), n_points=int(T.size),
                         fit_range=[float(T.min()), float(T.max())],
                         units={"C": _C_UNIT[unit_system], "theta": "K",
                                "chi0": _CHI0_UNIT[unit_system], "mu_eff": "mu_B"},
                         quality_flags=flags)

#: The modified fit's C lower bound is 1e-12 on chi / median|chi|. A real Curie term there
#: is O(10..1000) (chi ~ 1 over T - theta of tens to hundreds of K); one within three
#: decades of the bound is pinned to it, not measured (measured: 4e-11 on a 2-20 K window
#: of a real file, against >= 0.5 on every unwindowed example and real file).
_MOD_C_PINNED_N = 1e-9

#: Modified-fit flags that remove a parameter from the report.
MOD_DECLINE_FLAGS = frozenset(f"{p}_{w}" for p in ("C", "theta", "chi0")
                              for w in ("unresolved", "at_bound"))
#: ... and those that also remove its curve: without theta or C there is no prediction.
MOD_NO_CURVE_FLAGS = frozenset(f for f in MOD_DECLINE_FLAGS if not f.startswith("chi0"))


def decline_modified(fr):
    """Null every modified-fit parameter whose flag declines it -- a pinned or unresolved
    parameter is not a measurement (same discipline as the resistivity power law and the
    TTO kappa_ph fit). mu_eff = k sqrt(C) goes with C. Sigma goes with its value."""
    flags = set(fr.quality_flags)
    gone = {p for p in ("C", "theta", "chi0")
            if f"{p}_unresolved" in flags or f"{p}_at_bound" in flags}
    if "C" in gone:
        gone.add("mu_eff")
    if not gone:
        return fr
    return fr.model_copy(update={
        "params": {k: (None if k in gone else v) for k, v in fr.params.items()},
        "sigma": {k: (None if k in gone else v) for k, v in fr.sigma.items()}})


def _r2(y, fit):
    ss_res = float(np.sum((y - fit) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


_CW_RUNGS = (25.0, 50.0, 100.0, 150.0, 200.0)   # closed O3: uniform, for reproducibility
_CW_MIN_RUNG_PTS = 10
_CW_SPREAD_FLOOR_K = 2.0    # closed O2: NOT physics-derived; convergence-noise guard, tunable
_CW_RUNG_MARGIN_K = 20.0    # a rung needs a window, not a sliver (code constant, not physics)


def fit_cw_ladder(T, inv_chi, unit_system="CGS", rungs=_CW_RUNGS, windowed=False):
    """(primary full-window FitResult with ladder flags attached, ladder, theta_spread_k,
    mu_eff_spread). Mirrors fit_kappa_ph_ladder (thermal.py:105).

    Measured motivation (spec §1.1): on the MPMS real file the full-window theta is
    -50.27 +- 0.99 while every T>=25 K rung sits at -42.2 .. -37.5 (spread 12.72 K, 13x the
    statistical sigma) and r2 >= 0.9965 everywhere — r2 cannot warn. The spread INCLUDES the
    full fit: its displacement from the rungs is the signal.
    `windowed=True` (the points ARE a user fit window): a rung at or below their lower edge
    would refit exactly the primary's points, so it is skipped, and one surviving rung
    already gives a spread (primary + rung = two fits). Without a user window the ladder is
    exactly the original one: no skip, a spread from >= 2 rungs.
    Raises only what the primary CurieWeissModel().fit raises; rung failures are omitted."""
    T = np.asarray(T, float); inv_chi = np.asarray(inv_chi, float)
    primary = CurieWeissModel().fit(T, inv_chi, unit_system=unit_system)
    tmax = float(np.nanmax(T))
    tmin = float(np.nanmin(T))
    ladder: list[dict] = []
    for cutoff in sorted(rungs):
        if cutoff >= tmax - _CW_RUNG_MARGIN_K:
            continue
        if windowed and cutoff <= tmin:      # a copy of the primary, not a rung
            continue
        m = np.isfinite(T) & np.isfinite(inv_chi) & (T >= cutoff)
        if int(m.sum()) < _CW_MIN_RUNG_PTS:
            continue
        try:
            fr = CurieWeissModel().fit(T[m], inv_chi[m], unit_system=unit_system)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        ladder.append({"tmin_k": float(cutoff),
                       "theta_k": float(fr.params["theta"]),
                       "sigma_theta_k": float(fr.sigma["theta"]),
                       "mu_eff": fr.params["mu_eff"],          # None where C <= 0
                       "sigma_mu_eff": fr.sigma["mu_eff"],
                       "r2": float(fr.r2), "n_points": int(fr.n_points)})
    if len(ladder) >= (1 if windowed else 2):
        thetas = [e["theta_k"] for e in ladder] + [float(primary.params["theta"])]
        mus = [m for m in [e["mu_eff"] for e in ladder] + [primary.params["mu_eff"]]
               if m is not None]
        theta_spread = float(max(thetas) - min(thetas))
        mu_spread = float(max(mus) - min(mus)) if len(mus) >= 2 else None
    else:
        theta_spread = mu_spread = None       # U2: None, never 0.0
    flags = list(primary.quality_flags)
    if (theta_spread is not None
            and theta_spread > max(3.0 * float(primary.sigma["theta"]), _CW_SPREAD_FLOOR_K)):
        flags.append("window_sensitive")
    if flags != list(primary.quality_flags):  # FitResult is frozen: copy, don't mutate
        primary = primary.model_copy(update={"quality_flags": flags})
    return primary, ladder, theta_spread, mu_spread


#: Samples across a drawn Curie-Weiss curve (before the exact window edges are inserted).
_CW_CURVE_POINTS = 400


def _positive_run(g, y, inside):
    """The longest contiguous run of finite, positive 1/chi -- a window-only curve never
    draws negative 1/chi and never joins two sides of a pole."""
    ok = np.isfinite(y) & (y > 0)
    best, i, n = (0, 0), 0, ok.size
    while i < n:
        if not ok[i]:
            i += 1
            continue
        j = i
        while j < n and ok[j]:
            j += 1
        if j - i > best[1] - best[0]:
            best = (i, j)
        i = j
    a, b = best
    return g[a:b], y[a:b], inside[a:b]


def cw_curve(fit_params, model, fit_lo, fit_hi, top):
    """The drawn Curie-Weiss curve, 1/chi against T, built once in the core.

    `fit_lo`/`fit_hi` are the fit's `fit_range` (the fitted points), `top` the requested upper
    end (never below `fit_hi`). The curve starts where it reaches 1/chi = 0 -- at its own
    theta, which may be negative -- and runs to `top` as one line:

      * curie_weiss: 1/chi = (T - theta)/C. Extended only when C > 0 and theta < fit_lo;
        otherwise the line is <= 0 somewhere inside its own window, and drawing it out would
        draw negative 1/chi ("C_nonpositive", "theta_in_window"). Nor when theta < -fit_hi
        ("theta_out_of_range", as for the modified fit below).
      * curie_weiss_modified: 1/chi = (T - theta)/(chi0 (T - theta) + C), written so that
        T = theta gives exactly 0. With chi0 < 0 the denominator vanishes at
        T* = theta - C/chi0 > theta and 1/chi is negative beyond it, so the curve stops short
        of T*; a pole inside the window declines the extension ("pole_in_window"). A theta
        below -fit_hi means T >> |theta| is never reached, so the curve is not carried to its
        theta either ("theta_out_of_range").

    A declined extension leaves the window only, clipped to finite 1/chi > 0, with
    `zero_crossing: False` and the `reason`. Both window edges are exact grid points, so
    `in_fit_window` flips AT the bound. Non-finite values never reach the result."""
    fit_lo, fit_hi = float(fit_lo), float(fit_hi)
    top = max(float(top), fit_hi)
    C = float(fit_params["C"]); th = float(fit_params["theta"])
    if model == "curie_weiss":
        def f(t):
            return (t - th) / C
        reason = ("C_nonpositive" if not C > 0 else
                  "theta_in_window" if not th < fit_lo else
                  "theta_out_of_range" if th < -fit_hi else None)
    else:
        c0 = float(fit_params["chi0"])

        def f(t):
            d = t - th
            with np.errstate(divide="ignore", invalid="ignore"):
                return d / (c0 * d + C)
        tstar = th - C / c0 if c0 < 0 else np.inf
        reason = ("C_nonpositive" if not C > 0 else
                  "theta_in_window" if not th < fit_lo else
                  "theta_out_of_range" if th < -fit_hi else
                  "pole_in_window" if tstar <= fit_hi else None)
    if reason is None:
        g = np.unique(np.concatenate([np.linspace(th, top, _CW_CURVE_POINTS), [fit_lo, fit_hi]]))
        if model != "curie_weiss" and np.isfinite(tstar):
            g = g[g < tstar]                  # analytic cut: the pole is never sampled
        y = np.asarray(f(g), float)
        y[0] = 0.0 if g[0] == th else y[0]
        inside = (g >= fit_lo) & (g <= fit_hi)
        keep = np.isfinite(y) & ((y > 0) | (g == th))
        g, y, inside = g[keep], y[keep], inside[keep]
    else:
        g = np.unique(np.concatenate([np.linspace(fit_lo, fit_hi, _CW_CURVE_POINTS),
                                      [fit_lo, fit_hi]]))
        y = np.asarray(f(g), float)
        g, y, inside = _positive_run(g, y, np.ones_like(g, bool))
    return {"t_grid": g.tolist(), "inv_chi_fit": y.tolist(),
            "in_fit_window": [bool(v) for v in inside],
            "zero_crossing": reason is None, "reason": reason,
            "fit_range": [fit_lo, fit_hi]}
