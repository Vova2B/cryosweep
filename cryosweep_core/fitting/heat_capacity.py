from __future__ import annotations
import warnings
import numpy as np
from scipy.integrate import quad
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import linregress
from cryosweep_core.result import FitResult

R = 8.314462618  # J/(mol*K)

SCHOTTKY_ZPEAK = 2.39936      # z=Delta/T at the r=1 Schottky maximum (T_peak = Delta/ZPEAK ~ 0.417*Delta)
SCHOTTKY_CMAX_R1 = 0.43922    # z^2 e^z/(1+e^z)^2 at ZPEAK -> C_max = f*R*CMAX per mole of TLS
MU_B_OVER_KB = 0.6717         # Bohr magneton / Boltzmann, K/T (Zeeman: Delta = g*MU_B_OVER_KB*B)

def schottky_two_level(T, f, Delta, r=1.0):
    """Molar two-level Schottky heat capacity, J/(mol*K). z=Delta/T, r=g0/g1.
    Prefactor is f*R (f = number of two-level systems per formula unit) -- n_atoms
    (a phonon-counting quantity) must NOT enter here. Overflow-safe: evaluated via
    u=e^{-z} so denom = r/u + 2 + u/r = (1+r*e^z)^2/(r*e^z) stays finite for z up to ~700."""
    T = np.asarray(T, float)
    z = Delta / T
    u = np.exp(-np.clip(z, 0.0, None))            # e^{-z} in (0,1]; underflows to 0 at z~745 -> denom=inf -> C=0
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        denom = r / u + 2.0 + u / r               # = (1+a)^2/a with a=r*e^z -> C = f*R*r*z^2*e^z/(1+r*e^z)^2
        C = f * R * z ** 2 / denom
    return C


def nuclear_tail(T, alphaN):
    """Nuclear Schottky high-T tail, J/(mol*K): alphaN / T^2 (alphaN grows ~ H^2)."""
    T = np.asarray(T, float)
    return alphaN / T ** 2


def _schottky_seed_peak(T, cp, gamma0, beta0, fit_max_k):
    T = np.asarray(T, float); cp = np.asarray(cp, float)
    m = np.isfinite(T) & np.isfinite(cp) & (T > 0) & (T <= fit_max_k)
    T, cp = T[m], cp[m]
    order = np.argsort(T); T, cp = T[order], cp[order]
    resid = cp - (gamma0 * T + beta0 * T ** 3)
    i = int(np.argmax(resid))
    # I3: an interior maximum that actually stands above BOTH window endpoints -- rejects a
    # monotonic rising tail (max at the last point) and flat/noise (no prominent interior bump).
    interior = 0 < i < (T.size - 1) and resid[i] >= resid[0] and resid[i] >= resid[-1]
    T_star = float(T[i]) if interior else None
    if interior:
        Delta_seed = SCHOTTKY_ZPEAK * T_star
        f_seed = max(resid[i] / (SCHOTTKY_CMAX_R1 * R), 1e-6)
    else:
        Delta_seed = max(2.5 * float(T.max()), 1.0)       # peak above window -> rough seed, pre-flag
        f_seed = 0.05
    # C2: stash the debye_t3-derived seeds so the fit's background p0 starts from THEM (not constants)
    return {"T": T, "cp": cp, "gamma0": float(gamma0), "beta0": float(beta0),
            "Delta_seed": float(Delta_seed), "f_seed": float(f_seed),
            "peak_covered": bool(interior), "T_star": T_star}


def _fit_one(model_func, T, cp, p0, bounds, param_names):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", OptimizeWarning)
        popt, pcov = curve_fit(model_func, T, cp, p0=p0, bounds=bounds, maxfev=20000)
    yhat = model_func(T, *popt)
    rss = float(np.sum((cp - yhat) ** 2))
    n = int(T.size); k = len(param_names)
    _, _, aicc = _information_criteria(rss, n, k)
    params = {nm: float(v) for nm, v in zip(param_names, popt)}
    perr = np.sqrt(np.diag(pcov))
    sigma = {nm: float(s) for nm, s in zip(param_names, perr)}
    return {"params": params, "sigma": sigma, "pcov": pcov.tolist(), "rss": rss,
            "n": n, "k": k, "aicc": aicc, "r2": _r2(cp, yhat),
            "param_names": list(param_names),                 # order == pcov row order (I1/corr)
            "bounds": [list(bounds[0]), list(bounds[1])]}     # for _identifiability (I1/I2)


def _fit_schottky_models(T, cp, seed, *, r, lattice_t5, include_nuclear,
                         nuclear_max_tmin_k, delta_max_k, f_max, aicc_margin=2.0):
    T, cp = seed["T"], seed["cp"]
    g0, b0, d0 = seed["gamma0"], seed["beta0"], 1e-8     # C2: seed background p0 from the debye_t3 fit
    # --- assemble background param layout (delta optional) ---
    bg_names = ["gamma", "beta"] + (["delta"] if lattice_t5 else [])
    def _bg(Tx, gamma, beta, *rest):
        out = gamma * Tx + beta * Tx ** 3
        if lattice_t5:
            out = out + rest[0] * Tx ** 5
        return out
    INF = np.inf
    bg_lo = [-INF, -INF] + ([-INF] if lattice_t5 else [])
    bg_hi = [INF, INF] + ([INF] if lattice_t5 else [])
    bg_p0 = [g0, b0] + ([d0] if lattice_t5 else [])

    models = {}
    # M0 background-only
    models["background"] = _fit_one(_bg, T, cp, bg_p0, (bg_lo, bg_hi), bg_names)

    # M1 +Schottky : params = bg + [f, Delta]
    sch_names = bg_names + ["f", "Delta"]
    def _m1(Tx, *v):
        nb = len(bg_names)
        bg = _bg(Tx, *v[:nb]); f, D = v[nb], v[nb + 1]
        return bg + schottky_two_level(Tx, f, D, r)
    m1_p0 = bg_p0 + [seed["f_seed"], seed["Delta_seed"]]
    m1_lo = bg_lo + [0.0, 1e-3]; m1_hi = bg_hi + [f_max, delta_max_k]
    m1_p0 = [min(max(v, lo), hi) for v, lo, hi in zip(m1_p0, m1_lo, m1_hi)]
    try:
        models["schottky"] = _fit_one(_m1, T, cp, m1_p0, (m1_lo, m1_hi), sch_names)
    except Exception:
        pass

    # M2 +Schottky+nuclear (gated)
    if include_nuclear and float(np.min(T)) <= nuclear_max_tmin_k:
        nuc_names = sch_names + ["alphaN"]
        def _m2(Tx, *v):
            return _m1(Tx, *v[:-1]) + nuclear_tail(Tx, v[-1])
        m2_p0 = m1_p0 + [max((cp[0] - _bg(T[0], *bg_p0)) * T[0] ** 2, 1e-6)]
        m2_lo = m1_lo + [0.0]; m2_hi = m1_hi + [INF]
        try:
            models["schottky_nuclear"] = _fit_one(_m2, T, cp, m2_p0, (m2_lo, m2_hi), nuc_names)
        except Exception:
            pass

    # --- within-basis AICc selection (same response y=cp), gated by a parsimony margin: a
    # non-background model may only be chosen if it beats background by >= aicc_margin ---
    finite = {kk: v for kk, v in models.items() if v["aicc"] is not None and np.isfinite(v["aicc"])}
    bg = finite.get("background", {}).get("aicc")
    if bg is not None:
        finite = {kk: v for kk, v in finite.items()
                  if kk == "background" or (bg - v["aicc"]) >= aicc_margin}
    chosen_key = min(finite, key=lambda kk: finite[kk]["aicc"]) if finite else "background"

    # chosen-model curve for the overlay plot
    grid = np.linspace(float(T.min()), float(T.max()), 200)
    ch = models[chosen_key]; cp_fit = _bg(grid, *[ch["params"][nm] for nm in bg_names])
    if chosen_key in ("schottky", "schottky_nuclear"):
        cp_fit = cp_fit + schottky_two_level(grid, ch["params"]["f"], ch["params"]["Delta"], r)
    if chosen_key == "schottky_nuclear":
        cp_fit = cp_fit + nuclear_tail(grid, ch["params"]["alphaN"])
    return {"models": models, "chosen_key": chosen_key, "seed": {"gamma": g0, "beta": b0},
            "t_grid": grid.tolist(), "cp_fit": np.asarray(cp_fit, float).tolist()}


def fit_schottky(T, cp, *, gamma0, beta0, r=1.0, lattice_t5=False, include_nuclear=False,
                 nuclear_max_tmin_k=2.5, fit_max_k=15.0, delta_max_k=100.0, f_max=5.0,
                 rel_sigma=1.0, bound_rail_frac=0.01, peak_corr_max=0.95, is_lowest_field=False,
                 aicc_margin=2.0):
    """Joint per-field low-T Schottky fit in the Cp-vs-T basis. Returns the g['schottky'] dict.
    Off-path (schottky_enabled) is handled by the caller; this always attempts a fit."""
    seed = _schottky_seed_peak(T, cp, gamma0, beta0, fit_max_k)   # stashes gamma0/beta0 for seeding
    T_w = seed["T"]
    empty = {"attempted": False, "chosen_key": "background", "params": {}, "sigma": {}, "r": r,
             "r2": None, "aicc": {}, "identifiability": {}, "delta_determined": False,
             "peak_covered": seed["peak_covered"], "reason": "too few points",
             "t_grid": [], "cp_fit": [], "t_data": [], "cp_data": [], "warnings": []}
    if T_w.size < 6:
        return empty
    res = _fit_schottky_models(T_w, seed["cp"], seed, r=r, lattice_t5=lattice_t5,
                               include_nuclear=include_nuclear, nuclear_max_tmin_k=nuclear_max_tmin_k,
                               delta_max_k=delta_max_k, f_max=f_max, aicc_margin=aicc_margin)
    ck = res["chosen_key"]; ch = res["models"][ck]
    warnings_out = []
    has_schottky = ck in ("schottky", "schottky_nuclear")
    delta_determined = False; reason = ""; ident = {}
    fitted_peak_covered = seed["peak_covered"]
    if has_schottky:
        # I1: populate per-param identifiability via the reused slice-2 machinery
        ident = _identifiability(ch["param_names"], ch["params"], ch["sigma"],
                                 (ch["bounds"][0], ch["bounds"][1]), rel_sigma, bound_rail_frac)
        D = ch["params"]["Delta"]; sD = ch["sigma"].get("Delta", np.inf)
        railed = ident.get("Delta", {}).get("railed", False)   # I2: reuse the computed rail flag
        rel = abs(sD / D) if D else np.inf
        # corr(f, Delta) from pcov (param_names order == pcov row order)
        names = ch["param_names"]; pc = np.asarray(ch["pcov"], float)
        try:
            i, j = names.index("f"), names.index("Delta")
            corr = abs(pc[i, j] / np.sqrt(pc[i, i] * pc[j, j]))
        except Exception:
            corr = 1.0
        # post-fit peak coverage: the FITTED Schottky component's maximum must lie strictly
        # inside the fit window (a peak above/below the window => Delta is a bound, not resolved).
        Tw = np.asarray(seed["T"], float)
        _grid = np.linspace(float(Tw.min()), float(Tw.max()), 200)
        _csch = schottky_two_level(_grid, ch["params"]["f"], ch["params"]["Delta"], r)
        _ipk = int(np.argmax(_csch))
        fitted_peak_covered = 0 < _ipk < (_grid.size - 1)
        # f railed against its upper bound => amplitude non-identifiable
        f_val = ch["params"].get("f", 0.0)
        f_railed = abs(f_val - f_max) <= bound_rail_frac * max(f_max, 1e-30)
        checks = {"peak": fitted_peak_covered, "delta_not_railed": not railed,
                  "f_not_railed": not f_railed, "rel_sigma": rel < rel_sigma, "corr": corr < peak_corr_max}
        delta_determined = all(checks.values())
        if not delta_determined:
            failed = [k for k, v in checks.items() if not v]
            reason = "Δ lower-bound only / not determined (" + ", ".join(failed) + ")"
        if D > 1.0 and ch["params"].get("f", 0) > 1.0:
            warnings_out.append("f > 1: TLS count per formula unit exceeds 1 (or misfit)")
        # beta cannibalization vs background-only
        b_bg = res["models"]["background"]["params"]["beta"]; b_ch = ch["params"]["beta"]
        if b_bg and abs(b_ch - b_bg) / abs(b_bg) > 0.30:
            warnings_out.append("β shifted >30% when adding Schottky (lattice/anomaly cannibalization)")
    if is_lowest_field:
        warnings_out.append("lowest-field group: a Kramers doublet has Δ→0 (Schottky ill-defined here)")
    return {"attempted": True, "chosen_key": ck, "params": dict(ch["params"]),
            "sigma": dict(ch["sigma"]), "r": r, "r2": ch["r2"],
            "aicc": {k: v["aicc"] for k, v in res["models"].items()},
            "identifiability": ident, "delta_determined": bool(delta_determined),
            "peak_covered": bool(fitted_peak_covered), "reason": reason,
            "t_grid": res["t_grid"], "cp_fit": res["cp_fit"],
            "t_data": T_w.tolist(), "cp_data": np.asarray(seed["cp"], float).tolist(),
            "warnings": warnings_out}


def _debye_integrand(x):
    if x < 1e-8:
        return x * x
    if x > 50.0:
        return x**4 * np.exp(-x)
    ex = np.exp(x)
    return x**4 * ex / (ex - 1.0) ** 2


def debye_heat_capacity(T, theta_D, n=1.0):
    if theta_D <= 0:
        raise ValueError("theta_D must be positive")
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    if np.any(T_arr <= 0):
        raise ValueError("temperature must be positive")
    out = np.empty_like(T_arr)
    for i, Ti in enumerate(T_arr):
        yi = theta_D / Ti
        if yi < 0.1:
            # High-T Debye expansion Cv = 3nR(1 - yi^2/20 + O(yi^4)); the 2-term form is
            # accurate to ~yi^4 (< 1e-5 at yi=0.1) and matches the quad integral, so we use
            # it instead of a bare 3nR (which is only correct as yi->0).
            out[i] = 3.0 * n * R * (1.0 - yi * yi / 20.0)
        elif yi > 700.0:
            out[i] = (12.0/5.0) * np.pi**4 * n * R * (Ti/theta_D)**3
        else:
            integral, _ = quad(_debye_integrand, 0.0, yi, limit=200)
            out[i] = 9.0 * n * R * (Ti/theta_D)**3 * integral
    return float(out[0]) if np.ndim(T) == 0 else out

def einstein_heat_capacity(T, theta_E, m):
    T = np.asarray(T, float)
    x = theta_E / T
    x_safe = np.clip(x, 0, 30)
    ex = np.exp(x_safe)
    denom = np.where(ex - 1 == 0, 1e-10, ex - 1)
    cv = m * 3 * R * x_safe**2 * ex / denom**2
    return np.where(x > 30, 0.0, cv)

def specific_heat_full(T, theta_D, n, gamma, theta_E1, theta_E2, m1, m2):
    """Debye-Einstein Cp(T) in J/(mol*K). gamma in J/(mol*K^2); electronic term = gamma*T.
    Ported from SpHeat_Sept2024_V6.py specific_heat_scalar (the /1000 there is because its g
    slider is in mJ; we keep gamma in J so it is apples-to-apples with the low-T fit)."""
    T = np.asarray(T, float)
    return (gamma * T
            + debye_heat_capacity(T, theta_D, n)
            + einstein_heat_capacity(T, theta_E1, m1)
            + einstein_heat_capacity(T, theta_E2, m2))

_FULL_PARAMS = ("theta_D", "n", "gamma", "theta_E1", "theta_E2", "m1", "m2")
_FULL_BOUNDS = {"theta_D": (1.0, 1000.0), "n": (0.0, 50.0), "gamma": (0.0, 1.0),
                "theta_E1": (1.0, 1000.0), "theta_E2": (1.0, 1000.0),
                "m1": (0.0, 50.0), "m2": (0.0, 50.0)}
_FULL_UNITS = {"theta_D": "K", "n": "", "gamma": "J/(mol*K^2)", "theta_E1": "K",
               "theta_E2": "K", "m1": "", "m2": ""}


def fit_full_range(T, cp, *, init, fixed, fit_min_k=None, fit_max_k=None, seed=None):
    """Config-driven full-range Debye-Einstein fit. Returns a plain dict (NOT FitResult)."""
    base = {k: float(init[k]) for k in _FULL_PARAMS}
    fixed = {k: bool(fixed.get(k, False)) for k in _FULL_PARAMS}
    if seed:
        for k in ("gamma", "theta_D"):
            if k in seed and seed[k] is not None and np.isfinite(seed[k]) and not fixed[k]:
                base[k] = float(seed[k])
    T = np.asarray(T, float); cp = np.asarray(cp, float)
    m = np.isfinite(T) & np.isfinite(cp) & (T > 0)
    if fit_min_k is not None: m &= T >= fit_min_k
    if fit_max_k is not None: m &= T <= fit_max_k
    T, cp = T[m], cp[m]
    free = [k for k in _FULL_PARAMS if not fixed[k]]
    fail = {"ok": False, "reason": "", "params": {}, "fixed": fixed, "r2": None,
            "n_points": int(T.size), "fit_range": [], "units": dict(_FULL_UNITS),
            "t_grid": [], "cp_fit": []}
    if T.size < len(free) + 2:
        fail["reason"] = "too few points for the free-parameter count"; return fail
    if not free:
        fail["reason"] = "no free parameters"; return fail

    def model(Tx, *vals):
        p = dict(base); p.update(zip(free, vals))
        return specific_heat_full(Tx, **{k: p[k] for k in _FULL_PARAMS})

    p0 = [base[k] for k in free]
    lo = [_FULL_BOUNDS[k][0] for k in free]; hi = [_FULL_BOUNDS[k][1] for k in free]
    p0 = [min(max(v, l), h) for v, l, h in zip(p0, lo, hi)]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            popt, _ = curve_fit(model, T, cp, p0=p0, bounds=(lo, hi),
                                method="trf", maxfev=20000)
    except Exception as e:
        fail["reason"] = f"curve_fit failed: {type(e).__name__}"; return fail
    p = dict(base); p.update(zip(free, (float(v) for v in popt)))
    if p["theta_E1"] > p["theta_E2"]:                 # canonical ordering
        p["theta_E1"], p["theta_E2"] = p["theta_E2"], p["theta_E1"]
        p["m1"], p["m2"] = p["m2"], p["m1"]
    yhat = specific_heat_full(T, **{k: p[k] for k in _FULL_PARAMS})
    ss_res = float(np.sum((cp - yhat) ** 2)); ss_tot = float(np.sum((cp - cp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    grid = np.linspace(float(T.min()), float(T.max()), 300)
    return {"ok": True, "reason": "", "params": p, "fixed": fixed, "r2": r2,
            "n_points": int(T.size), "fit_range": [float(T.min()), float(T.max())],
            "units": dict(_FULL_UNITS),
            "t_grid": grid.tolist(),
            "cp_fit": specific_heat_full(grid, **{k: p[k] for k in _FULL_PARAMS}).tolist()}


def debye_temp_from_beta(beta, n_atoms=1):
    if beta <= 0:
        return float("nan")
    return (12 * np.pi**4 * n_atoms * R / (5 * beta)) ** (1.0/3.0)

class DebyeLowTModel:
    key = "debye_t3"
    params = ["gamma", "beta", "theta_D"]

    def __init__(self, n_atoms=1.0):
        self.n_atoms = n_atoms

    def fit(self, T, cp) -> FitResult:
        T = np.asarray(T, float); cp = np.asarray(cp, float)
        m = np.isfinite(T) & np.isfinite(cp) & (T > 0)
        T, cp = T[m], cp[m]
        T2 = T**2; y = cp / T                         # Cp/T = gamma + beta*T^2
        r = linregress(T2, y)
        gamma, beta = float(r.intercept), float(r.slope)
        theta = debye_temp_from_beta(beta, self.n_atoms)
        # sigma on theta_D via delta method: d theta/d beta = -theta/(3 beta)
        s_theta = abs(theta / (3 * beta)) * float(r.stderr) if beta > 0 and np.isfinite(theta) else float("nan")
        fit_line = gamma + beta * T2
        ss_res = float(np.sum((y - fit_line)**2)); ss_tot = float(np.sum((y - y.mean())**2))
        r2 = 1.0 - ss_res/ss_tot if ss_tot else 0.0
        return FitResult(model="debye_t3",
                         params={"gamma": gamma, "beta": beta, "theta_D": theta},
                         sigma={"gamma": float(r.intercept_stderr), "beta": float(r.stderr), "theta_D": s_theta},
                         covariance=[], r2=r2, n_points=int(T.size),
                         fit_range=[float(T.min()), float(T.max())],
                         units={"gamma": "J/(mol*K^2)", "beta": "J/(mol*K^4)", "theta_D": "K"},
                         quality_flags=[])


# --- Low-T model functions in Cp/T vs T^2 space (x = T^2). Ported verbatim from the owner's
#     SpHeat_Sept2024_V6.py low_temperature_fitting() (lines 468-491). ---
def _lowt_debye_t3_t5(x, gamma, beta, delta):
    return gamma + beta * x + delta * x * x                       # gamma + beta T^2 + delta T^4

def _lowt_spin_noninteracting(x, gamma, beta, A, T0):
    T = np.sqrt(x)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_term = np.where(T > 0, np.log(T0 / np.where(T > 0, T, 1.0)), 0.0)
    return gamma + beta * x + A * x * log_term                    # + A T^2 ln(T0/T)

def _lowt_spin_weak(x, gamma, beta, A, T0):
    T = np.sqrt(x)
    return gamma + beta * x + A * x * (1.0 + (T / T0) ** 2)       # + A T^2 (1 + (T/T0)^2)

# Complexity order (parsimony iterates this list). debye_t3 is special-cased to the analytic
# DebyeLowTModel; the rest use curve_fit with the reference p0/bounds.
_LOWT_MODELS = [
    {"key": "debye_t3", "label": "Debye T³", "param_names": ["gamma", "beta"]},
    {"key": "debye_t3_t5", "label": "Debye T³+T⁵", "func": _lowt_debye_t3_t5,
     "p0": [0.01, 1e-4, 1e-6], "bounds": None, "param_names": ["gamma", "beta", "delta"]},
    {"key": "spin_fluct_noninteracting", "label": "spin-fl non-int", "func": _lowt_spin_noninteracting,
     "p0": [0.01, 1e-4, 1e-4, 10.0], "bounds": ([-np.inf, -np.inf, -np.inf, 1.0], [np.inf, np.inf, np.inf, 500.0]),
     "param_names": ["gamma", "beta", "A", "T0"]},
    {"key": "spin_fluct_weak", "label": "spin-fl weak", "func": _lowt_spin_weak,
     "p0": [0.01, 1e-4, 1e-4, 10.0], "bounds": ([-np.inf, -np.inf, -np.inf, 1.0], [np.inf, np.inf, np.inf, 500.0]),
     "param_names": ["gamma", "beta", "A", "T0"]},
]


# Direct evaluators (Cp/T in the x = T^2 basis) for tail extrapolation of the entropy integral.
_LOWT_FUNCS = {
    "debye_t3": lambda x, p: p["gamma"] + p["beta"] * x,
    "debye_t3_t5": lambda x, p: _lowt_debye_t3_t5(x, p["gamma"], p["beta"], p["delta"]),
    "spin_fluct_noninteracting": lambda x, p: _lowt_spin_noninteracting(x, p["gamma"], p["beta"], p["A"], p["T0"]),
    "spin_fluct_weak": lambda x, p: _lowt_spin_weak(x, p["gamma"], p["beta"], p["A"], p["T0"]),
}


def eval_lowt_cp_over_t(model_key, params, T):
    """Cp/T for the chosen low-T model at temperatures T (x = T^2 basis). Raises KeyError on unknown key."""
    T = np.asarray(T, float)
    return np.asarray(_LOWT_FUNCS[model_key](T ** 2, params), float)


def _fin(x):
    """Finite float or None (non-finite must never reach JSON/CSV)."""
    return float(x) if (x is not None and np.isfinite(x)) else None


def _identifiability(param_names, values, sigma, bounds, rel_sigma, bound_rail_frac):
    """Per-free-param identifiability. bounds: (lo_list, hi_list) or None."""
    out = {}
    lo, hi = (bounds if bounds is not None else (None, None))
    for i, nm in enumerate(param_names):
        v = float(values[nm]); s = float(sigma.get(nm, np.inf))
        rel = abs(s / v) if v != 0 else np.inf
        railed = False
        if lo is not None:
            for bnd in (lo[i], hi[i]):
                if np.isfinite(bnd) and abs(v - bnd) <= bound_rail_frac * max(abs(bnd), 1e-30):
                    railed = True
        ok = np.isfinite(s) and (rel < rel_sigma) and (not railed)
        out[nm] = {"sigma": s, "rel_sigma": float(rel), "railed": bool(railed), "ok": bool(ok)}
    return out


def _r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot else 0.0

def _adj_r2(r2, n, p):
    return 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else float("-inf")

def _units_for(param_names):
    u = {"gamma": "J/(mol*K^2)", "beta": "J/(mol*K^4)", "delta": "J/(mol*K^6)",
         "A": "J/(mol*K^4)", "T0": "K", "theta_D": "K"}
    return {k: u.get(k, "") for k in list(param_names) + ["theta_D"]}

def _failed(spec):
    return {"key": spec["key"], "label": spec["label"], "ok": False, "r2": float("-inf"),
            "adj_r2": float("-inf"), "params": {}, "theta_D": float("nan"),
            "n_params": len(spec["param_names"]), "t2_grid": [], "cp_over_t_fit": [],
            "fitresult": None}

def _information_criteria(rss, n, k):
    """AIC/BIC/AICc for least squares. rss clamped to avoid log(0). AICc None if n-k-2<=0."""
    rss_safe = rss if rss > 1e-300 else 1e-300
    ll_term = n * np.log(rss_safe / n)
    aic = ll_term + 2.0 * (k + 1)
    bic = ll_term + (k + 1) * np.log(n)
    aicc = aic + 2.0 * (k + 1) * (k + 2) / (n - k - 2) if (n - k - 2) > 0 else None
    return float(aic), float(bic), (float(aicc) if aicc is not None else None)


def fit_delta_h_overlay(fields_oe, deltas_k, model="zeeman"):
    """Plot-time Zeeman/ZFS Δ(H) overlay fit.

    Interprets per-field Δ(H) points as either:
      - "zeeman": Δ = g·MU_B_OVER_KB·B  (linear through origin)
      - "zfs":    Δ = √(Δ₀² + (g·MU_B_OVER_KB·B)²)  (zero-field splitting)
      - "none":   no fit (pass-through, ok=False)

    Fields in Oe are converted to Tesla via /1e4.
    Requires ≥3 finite points; returns ok=False otherwise.

    Returns dict: {"model", "g_factor", "Delta0", "r2", "n_points", "ok"}.
    """
    B = np.asarray(fields_oe, float) / 1e4                # Oe -> Tesla
    D = np.asarray(deltas_k, float)
    m = np.isfinite(B) & np.isfinite(D)
    B, D = B[m], D[m]
    out = {"model": model, "g_factor": None, "Delta0": None, "r2": None,
           "n_points": int(B.size), "ok": False}
    if model == "none" or B.size < 3:
        return out
    try:
        if model == "zeeman":
            r = linregress(B, D)                          # D = (g*MU)*B  (through ~origin)
            g = float(r.slope) / MU_B_OVER_KB
            yhat = r.slope * B + r.intercept
            out.update(g_factor=g, Delta0=float(r.intercept), r2=float(r.rvalue ** 2), ok=True)
        elif model == "zfs":
            def f(Bx, g, D0): return np.sqrt(D0 ** 2 + (g * MU_B_OVER_KB * Bx) ** 2)
            popt, _ = curve_fit(f, B, D, p0=[2.0, max(float(D.min()), 0.1)], maxfev=20000)
            yhat = f(B, *popt)
            ss = 1.0 - np.sum((D - yhat) ** 2) / np.sum((D - D.mean()) ** 2)
            out.update(g_factor=float(abs(popt[0])), Delta0=float(abs(popt[1])), r2=float(ss), ok=True)
    except Exception:
        return out
    return out


def fit_lowt_models(T, cp, n_atoms=1.0, parsimony_r2=0.99, extended=False,
                    rel_sigma=1.0, bound_rail_frac=0.01, corr_warn=0.99):
    """Fit all four low-T models on Cp/T vs T^2; return every result + the parsimony-chosen one.
    debye_t3 uses the analytic DebyeLowTModel (linregress); the rest use curve_fit.

    When extended=True each dict in fits[] additionally carries:
      sigma, aic, bic, aicc, max_abs_corr.
    When extended=False the return is byte-identical to the pre-Task-2 result."""
    T = np.asarray(T, float); cp = np.asarray(cp, float)
    m = np.isfinite(T) & np.isfinite(cp) & (T > 0)
    T, cp = T[m], cp[m]
    x = T ** 2; y = cp / T; n = int(x.size)
    grid = np.linspace(float(x.min()), float(x.max()), 200) if n else np.array([])
    fits = []
    for spec in _LOWT_MODELS:
        p = len(spec["param_names"])
        if n < p + 2:
            fits.append(_failed(spec)); continue
        try:
            if spec["key"] == "debye_t3":
                fr = DebyeLowTModel(n_atoms=n_atoms).fit(T, cp)
                params = {"gamma": fr.params["gamma"], "beta": fr.params["beta"]}
                theta = fr.params["theta_D"]; r2 = float(fr.r2)
                curve = params["gamma"] + params["beta"] * grid
                pcov = None  # not used for debye_t3 (analytic path)
            else:
                kw = {"p0": spec["p0"], "maxfev": 10000}
                if spec["bounds"] is not None:
                    kw["bounds"] = spec["bounds"]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    popt, pcov = curve_fit(spec["func"], x, y, **kw)
                params = dict(zip(spec["param_names"], (float(v) for v in popt)))
                theta = debye_temp_from_beta(params["beta"], n_atoms)
                r2 = _r2(y, spec["func"](x, *popt))
                curve = spec["func"](grid, *popt)
            theta_pub = float(theta)
            ext = {}
            if extended:
                is_lattice = spec["key"] in ("debye_t3", "debye_t3_t5")
                if not is_lattice:
                    theta_pub = float("nan")         # C4: spin-fluct beta non-lattice -> theta_D meaningless
                if spec["key"] == "debye_t3":
                    sigma_raw = {"gamma": float(fr.sigma["gamma"]), "beta": float(fr.sigma["beta"])}
                    if np.isfinite(theta):           # theta_D sigma (delta method, from FitResult)
                        sigma_raw["theta_D"] = float(fr.sigma["theta_D"])
                    max_corr = None                  # 2-param lattice; not the degeneracy concern
                    rss = float(np.sum((y - (params["gamma"] + params["beta"] * x)) ** 2))
                else:
                    perr = np.sqrt(np.diag(pcov))
                    sigma_raw = {nm: float(s) for nm, s in zip(spec["param_names"], perr)}
                    d = np.sqrt(np.diag(pcov))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        corr = pcov / np.outer(d, d)
                    iu = np.triu_indices(len(d), k=1)
                    with np.errstate(all="ignore"):
                        vals = np.abs(corr[iu])
                    max_corr = (float(np.nanmax(vals)) if iu[0].size and not np.all(np.isnan(vals)) else None)
                    rss = float(np.sum((y - spec["func"](x, *popt)) ** 2))
                aic, bic, aicc = _information_criteria(rss, n, p)
                bnds = spec.get("bounds")
                ident = _identifiability(spec["param_names"], params, sigma_raw, bnds,
                                         rel_sigma, bound_rail_frac)
                fit_ok = all(pp["ok"] for pp in ident.values()) and \
                         (max_corr is None or max_corr < corr_warn)
                for nm in ident:
                    ident[nm]["sigma"] = _fin(ident[nm]["sigma"])
                    ident[nm]["rel_sigma"] = _fin(ident[nm]["rel_sigma"])
                ext = {"sigma": {nm: _fin(s) for nm, s in sigma_raw.items()},
                       "aic": aic, "bic": bic, "aicc": aicc,
                       "max_abs_corr": (_fin(max_corr) if max_corr is not None else None),
                       "identifiable": bool(fit_ok), "identifiability": ident}
            fr_params = dict(params); fr_params["theta_D"] = theta_pub
            # gamma < 0 is unphysical (negative Sommerfeld coefficient) but it IS the
            # measured value: flag it machine-readably rather than blanking it, so every
            # surface that prints gamma (the figure annotation included) can say so.
            qflags = ["gamma_negative"] if fr_params.get("gamma", 0.0) < 0 else []
            fitresult = FitResult(model=spec["key"], params=fr_params, r2=r2, n_points=n,
                                  fit_range=[float(T.min()), float(T.max())],
                                  units=_units_for(spec["param_names"]),
                                  quality_flags=qflags)
            fits.append({"key": spec["key"], "label": spec["label"], "ok": True, "r2": r2,
                         "adj_r2": _adj_r2(r2, n, p), "params": fr_params, "theta_D": theta_pub,
                         "n_params": p, "t2_grid": grid.tolist(),
                         "cp_over_t_fit": np.asarray(curve, float).tolist(), "fitresult": fitresult,
                         **ext})
        except Exception:
            fits.append(_failed(spec))
    chosen = next((f for f in fits if f["ok"] and f["r2"] >= parsimony_r2), None)
    if chosen is None:
        ok = [f for f in fits if f["ok"]]
        chosen = max(ok, key=lambda f: f["adj_r2"]) if ok else None
    _extra = ("sigma", "aic", "bic", "aicc", "max_abs_corr", "identifiable", "identifiability")
    public = [{**{k: f[k] for k in ("key", "label", "ok", "r2", "adj_r2", "params", "theta_D",
                                    "n_params", "t2_grid", "cp_over_t_fit")},
               **{k: f[k] for k in _extra if k in f}} for f in fits]
    return {"fits": public, "chosen": (chosen["fitresult"] if chosen else None),
            "chosen_key": (chosen["key"] if chosen else None)}
