from __future__ import annotations
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from cryosweep_core.result import FitResult

_KAPPA_PH_MIN_PTS = 10       # denser data than the rho fit's >=4 -- TTO curves carry hundreds
_N_SPREAD_FLOOR = 0.05       # absolute floor under the window_sensitive rule (C4) -- see below


def _kappa_ph_power(T, B, n):
    return B * np.power(T, n)


class KappaPhPowerModel:
    """Phonon thermal conductivity power law kappa_ph = B*T^n over a low-T window.

    n is reported WITH its window-ladder spread AND its method delta, because on real data
    both move n far more than the statistical sigma does (measured on the gate file:
    2.03 -> 1.31 over 10 -> 30 K, and 2.0266 curve_fit vs 2.0078 log-log on the same
    <=10 K window, against sigma 0.006). Never report the statistical sigma alone.

    `window_sensitive` is EXPECTED on real data -- no real kappa(T) is one power law from
    10 to 30 K. Its ABSENCE is the informative case."""
    key = "kappa_ph_power"
    params = ["B", "n"]

    def fit(self, T, kappa_ph, cutoff=10.0) -> FitResult:
        T = np.asarray(T, float); k = np.asarray(kappa_ph, float)
        m = np.isfinite(T) & np.isfinite(k) & (T > 0) & (k > 0) & (T <= cutoff)
        T, k = T[m], k[m]
        if T.size < _KAPPA_PH_MIN_PTS:
            raise ValueError(f"kappa_ph power-law fit needs >={_KAPPA_PH_MIN_PTS} finite "
                             f"kappa_ph > 0 points at T <= {cutoff}")
        # curve_fit (Levenberg-Marquardt) is NOT scale-invariant: kappa_ph at a small absolute
        # magnitude traps the optimizer at its initial guess. Fit on kappa_ph normalized by its
        # mean, then rescale B (n, sigma_n and r2 are scale-invariant). Same as PowerLawRhoModel.
        s = float(np.mean(k))
        if not (s > 0):
            # DEFENSIVE ONLY, unreachable through this API: the (k > 0) mask above already
            # guarantees mean(k) > 0 for a non-empty array, and an empty array is caught by the
            # point-count guard. Kept for parity with PowerLawRhoModel (transport.py:49-50).
            raise ValueError("kappa_ph power-law fit needs positive-mean kappa_ph")
        y = k / s
        # p0's n-guess is deliberately 2.0 and is PINNED by the degenerate-window test below:
        # on a window with a single distinct T nothing is fitted and n comes back as exactly p0.
        p0 = [float(y.min() / T.max() ** 2), 2.0]
        popt, pcov = curve_fit(_kappa_ph_power, T, y, p0=p0, maxfev=20000,
                               bounds=([0.0, 0.5], [np.inf, 6.0]))
        resid = y - _kappa_ph_power(T, *popt)
        ss_res = float(np.sum(resid ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
        sig = np.sqrt(np.clip(np.diag(pcov), 0.0, np.inf))
        n_val = float(popt[1])
        b_val = float(popt[0] * s)
        sigma_b = float(sig[0] * s)
        # DEVIATION from the plan's pinned code block (deliberate, review finding I1, 2026-08-03):
        # the `* s` rescale can OVERFLOW to +inf on extreme-magnitude kappa_ph (measured: a flat
        # kappa_ph ~ 1e307 returns B = sigma_B = inf with no flag at all), and a non-finite B
        # would then reach JSON/CSV, which the global "never emit non-finite" constraint forbids
        # and which json.dumps(allow_nan=False) turns into a hard raise downstream. Declining at
        # the source is consistent with how fit_kappa_ph_ladder already treats a bad rung: it
        # catches ValueError and drops that rung, and lets a bad PRIMARY rung propagate.
        # (n / sigma_n / r2 are scale-invariant and are guarded in the ladder at line ~120.)
        if not (np.isfinite(b_val) and np.isfinite(sigma_b)):
            raise ValueError("kappa_ph power-law fit produced a non-finite prefactor B "
                             "(magnitude rescale overflowed) -- declining rather than reporting")
        flags = []
        if abs(n_val - 0.5) < 1e-2 or abs(n_val - 6.0) < 1e-2:
            flags.append("n_at_bound")
        if np.unique(T).size < 2:
            # A window with a single distinct T fits NOTHING: n comes back as exactly p0 (2.0),
            # sigma_n ~ 0 and r2 = 0.0. Without this flag a consumer would have to infer the
            # degeneracy from r2 == 0.0, which is not a safe signal. n here is not a measurement.
            flags.append("degenerate_window")
        return FitResult(
            model="kappa_ph_power",
            params={"B": b_val, "n": n_val},
            sigma={"B": sigma_b, "n": float(sig[1])},
            covariance=[], r2=float(r2), n_points=int(T.size),
            fit_range=[float(T.min()), float(cutoff)],
            units={"B": "W/(K^(1+n) m)", "n": ""},
            quality_flags=flags)


def _loglog_rung(T, kappa_ph, primary):
    """(n, sigma, r2, n_points) from an OLS fit of log(kappa_ph) on log(T) over the primary
    mask, or None. The METHOD is a free choice the spec proves matters (0.154 = ~19 sigma on
    the gate file's <=15 K window), so it is reported rather than silently made."""
    T = np.asarray(T, float); k = np.asarray(kappa_ph, float)
    m = np.isfinite(T) & np.isfinite(k) & (T > 0) & (k > 0) & (T <= primary)
    T, k = T[m], k[m]
    if T.size < _KAPPA_PH_MIN_PTS:
        return None
    try:
        r = linregress(np.log(T), np.log(k))
    except ValueError:
        return None
    n, sg, r2 = float(r.slope), float(r.stderr), float(r.rvalue ** 2)
    if not all(np.isfinite(v) for v in (n, sg, r2)):
        return None
    return n, sg, r2, int(T.size)


def fit_kappa_ph_ladder(T, kappa_ph, kappa_e=None,
                        rungs=(10.0, 15.0, 20.0, 30.0),
                        primary=10.0) -> tuple[FitResult, list[dict]]:
    """(primary FitResult with ladder flags attached, full ladder).

    The window choice moves n by 0.71 on the gate file, the method by 0.15, and the
    statistical sigma is 0.008 -- so reporting "n = 1.80 +- 0.008" would be a textbook
    over-claim, and r2 cannot warn you (every rung is >= 0.99). Raises ValueError when the
    PRIMARY rung declines; non-primary rungs are caught here and omitted from the ladder."""
    T = np.asarray(T, float); k = np.asarray(kappa_ph, float)
    model = KappaPhPowerModel()
    primary_res = model.fit(T, k, cutoff=primary)          # may raise -> caller declines
    finite_T = T[np.isfinite(T)]
    tmax = float(finite_T.max()) if finite_T.size else float("nan")
    ladder: list[dict] = []
    for cut in sorted(rungs):
        if np.isfinite(tmax) and cut > tmax:               # skipped, not failed
            continue
        try:
            fr = primary_res if cut == primary else model.fit(T, k, cutoff=cut)
        except (ValueError, RuntimeError):
            continue
        n_i, s_i, r2_i = fr.params["n"], fr.sigma["n"], fr.r2
        if not all(np.isfinite(float(v)) for v in (n_i, s_i, r2_i)):
            continue                                       # a non-finite rung is a failed rung
        ladder.append({"cutoff_k": float(cut), "method": "curve_fit", "n": float(n_i),
                       "sigma": float(s_i), "r2": float(r2_i), "n_points": int(fr.n_points)})
    ll = _loglog_rung(T, k, primary)
    if ll is not None:                                     # appended LAST (M8 ordering)
        ladder.append({"cutoff_k": float(primary), "method": "loglog", "n": ll[0],
                       "sigma": ll[1], "r2": ll[2], "n_points": ll[3]})

    cf = [e["n"] for e in ladder if e["method"] == "curve_fit"]
    n_spread = float(max(cf) - min(cf)) if len(cf) >= 2 else None   # None, NEVER 0.0 (I1)
    sigma_n = float(primary_res.sigma["n"])
    flags = list(primary_res.quality_flags)
    if n_spread is None:
        flags.append("ladder_incomplete")
    elif n_spread > max(3.0 * sigma_n, _N_SPREAD_FLOOR):
        # The 0.05 floor is LOAD-BEARING: without it both sides collapse to curve_fit's
        # convergence noise on exact data (7.37e-9 vs 3.87e-9 -> fires), inverting the
        # "does not cry wolf" oracle. 0.05 is the smallest phonon-exponent drift worth a
        # reader's attention. On real files this flag is EXPECTED to fire; its absence is
        # the informative case.
        flags.append("window_sensitive")
    if kappa_e is not None:
        ke = np.asarray(kappa_e, float)
        m = (np.isfinite(T) & np.isfinite(k) & (T > 0) & (k > 0) & (T <= primary)
             & np.isfinite(ke))
        if m.any():
            tot = k[m] + ke[m]
            good = tot > 0
            if good.any() and float(np.median(ke[m][good] / tot[good])) > 0.5:
                flags.append("kappa_e_dominant")
    return primary_res.model_copy(update={"quality_flags": flags}), ladder
