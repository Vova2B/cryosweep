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
            mu_eff = k * np.sqrt(C)
            s_sl = float(np.sqrt(cov[0, 0])); s_int = float(np.sqrt(cov[1, 1])); c_si = float(cov[0, 1])
            sC = s_sl / slope ** 2
            sTheta = np.sqrt(s_int ** 2 / slope ** 2 + (intercept ** 2 / slope ** 4) * s_sl ** 2
                             - 2.0 * (intercept / slope ** 3) * c_si)
            sMu = k / (2.0 * np.sqrt(C)) * sC
            fit_line = (T - theta) / C
            r2 = _r2(inv_chi, fit_line)
            return FitResult(model="curie_weiss", params={"C": C, "theta": theta, "mu_eff": mu_eff},
                             sigma={"C": abs(sC), "theta": float(sTheta), "mu_eff": abs(sMu)},
                             covariance=cov.tolist(), r2=r2, n_points=int(T.size),
                             fit_range=[float(T.min()), float(T.max())],
                             units={"C": _C_UNIT[unit_system], "theta": "K", "mu_eff": "mu_B"}, quality_flags=[])
        # modified: chi = chi0 + C/(T - theta)
        chi = 1.0 / inv_chi
        lin = np.polyfit(T, inv_chi, 1)
        p0 = [abs(1.0 / lin[0]) if lin[0] else 1.0, min(-lin[1] / lin[0] if lin[0] else 0.0, T.min() - 1.0), 0.0]
        lower = [1e-12, -np.inf, -np.inf]; upper = [np.inf, T.min() - 1e-6, np.inf]
        popt, pcov = curve_fit(lambda t, C, th, c0: c0 + C / (t - th), T, chi, p0=p0, bounds=(lower, upper), maxfev=10000)
        C, theta, chi0 = map(float, popt)
        mu_eff = k * np.sqrt(C)
        sig = np.sqrt(np.diag(pcov))
        fit_line = 1.0 / (chi0 + C / (T - theta))
        return FitResult(model="curie_weiss_modified",
                         params={"C": C, "theta": theta, "chi0": chi0, "mu_eff": mu_eff},
                         sigma={"C": float(sig[0]), "theta": float(sig[1]), "chi0": float(sig[2]),
                                "mu_eff": float(k / (2 * np.sqrt(C)) * sig[0])},
                         covariance=pcov.tolist(), r2=_r2(inv_chi, fit_line), n_points=int(T.size),
                         fit_range=[float(T.min()), float(T.max())],
                         units={"C": _C_UNIT[unit_system], "theta": "K",
                                "chi0": _CHI0_UNIT[unit_system], "mu_eff": "mu_B"},
                         quality_flags=[])

def _r2(y, fit):
    ss_res = float(np.sum((y - fit) ** 2)); ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0


_CW_RUNGS = (25.0, 50.0, 100.0, 150.0, 200.0)   # closed O3: uniform, for reproducibility
_CW_MIN_RUNG_PTS = 10
_CW_SPREAD_FLOOR_K = 2.0    # closed O2: NOT physics-derived; convergence-noise guard, tunable
_CW_RUNG_MARGIN_K = 20.0    # a rung needs a window, not a sliver (code constant, not physics)


def fit_cw_ladder(T, inv_chi, unit_system="CGS", rungs=_CW_RUNGS):
    """(primary full-window FitResult with ladder flags attached, ladder, theta_spread_k,
    mu_eff_spread). Mirrors fit_kappa_ph_ladder (thermal.py:105).

    Measured motivation (spec §1.1): on the MPMS real file the full-window theta is
    -50.27 +- 0.99 while every T>=25 K rung sits at -42.2 .. -37.5 (spread 12.72 K, 13x the
    statistical sigma) and r2 >= 0.9965 everywhere — r2 cannot warn. The spread INCLUDES the
    full fit: its displacement from the rungs is the signal.
    Raises only what the primary CurieWeissModel().fit raises; rung failures are omitted."""
    T = np.asarray(T, float); inv_chi = np.asarray(inv_chi, float)
    primary = CurieWeissModel().fit(T, inv_chi, unit_system=unit_system)
    tmax = float(np.nanmax(T))
    ladder: list[dict] = []
    for cutoff in sorted(rungs):
        if cutoff >= tmax - _CW_RUNG_MARGIN_K:
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
                       "mu_eff": float(fr.params["mu_eff"]),
                       "sigma_mu_eff": float(fr.sigma["mu_eff"]),
                       "r2": float(fr.r2), "n_points": int(fr.n_points)})
    if len(ladder) >= 2:
        thetas = [e["theta_k"] for e in ladder] + [float(primary.params["theta"])]
        mus = [e["mu_eff"] for e in ladder] + [float(primary.params["mu_eff"])]
        theta_spread = float(max(thetas) - min(thetas))
        mu_spread = float(max(mus) - min(mus))
    else:
        theta_spread = mu_spread = None       # U2: None, never 0.0
    flags = list(primary.quality_flags)
    if (theta_spread is not None
            and theta_spread > max(3.0 * float(primary.sigma["theta"]), _CW_SPREAD_FLOOR_K)):
        flags.append("window_sensitive")
    if flags != list(primary.quality_flags):  # FitResult is frozen: copy, don't mutate
        primary = primary.model_copy(update={"quality_flags": flags})
    return primary, ladder, theta_spread, mu_spread
