from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import curve_fit, OptimizeWarning
from cryosweep_core.fitting.heat_capacity import _information_criteria, _r2, _identifiability, _fin

# Fixed critical exponents by universality class (alpha=0 => logarithmic branch).
UNIVERSALITY = {"mean_field": 0.0, "ising3d": 0.110, "xy3d": -0.013}

# Regularization: the power-law/log divergence at t=0 is never sampled; clamp |t|.
_T_EPS = 1e-3   # minimum |reduced temperature| used in the singular term

# Minimum raw data points required strictly below AND strictly above a fitted Tc for it to
# count as bracketed (see fit_transition). Below this, one branch is anchored by too few
# points to trust regardless of the nominal curve_fit sigma.
_MIN_BRACKET_PTS = 3

# Collapse gate (fit_transition HARD gate 2): fraction of the ORIGINAL AICc advantage a
# real localized anomaly may retain after near-Tc point removal (its tails carry genuine
# signal). Measured on synthetic fixtures: true anomalies retain <= 0.29x, smooth
# background inadequacy >= 0.45x.
_COLLAPSE_FRAC = 0.4


def background(T, gamma, beta, delta=0.0, lattice_t5=False):
    """Smooth monotone lattice+electronic background gamma*T + beta*T^3 [+ delta*T^5]."""
    T = np.asarray(T, float)
    out = gamma * T + beta * T ** 3
    if lattice_t5:
        out = out + delta * T ** 5
    return out


def lambda_anomaly(T, Tc, alpha, Aplus, Aminus):
    """Singular critical term only (NO background). t=(T-Tc)/Tc; branch-asymmetric amplitudes.
    alpha != 0: (A_pm/alpha)*|t|^{-alpha}. alpha == 0: -A_pm*ln|t| (logarithmic class).
    |t| is clamped to _T_EPS so the never-sampled divergence stays finite."""
    T = np.asarray(T, float)
    t = (T - Tc) / Tc
    at = np.clip(np.abs(t), _T_EPS, None)
    A = np.where(t >= 0.0, Aplus, Aminus)
    if alpha == 0.0:
        return -A * np.log(at)
    return (A / alpha) * at ** (-alpha)


def jump_step(T, Tc, dC):
    """Mean-field jump: dC below Tc, 0 above, 0.5*dC at Tc (Heaviside Theta(Tc-T))."""
    T = np.asarray(T, float)
    step = np.where(T < Tc, 1.0, np.where(T > Tc, 0.0, 0.5))
    return dC * step


def _sorted_finite(T, cp):
    T = np.asarray(T, float); cp = np.asarray(cp, float)
    m = np.isfinite(T) & np.isfinite(cp) & (T > 0)
    T, cp = T[m], cp[m]
    o = np.argsort(T)
    return T[o], cp[o]


def locate_lambda(T, cp):
    """Locate a candidate lambda-anomaly Tc on a QUARTIC DETREND of the data (no low-T
    seeds: a gamma*T+beta*T^3 seed background diverges at high T and buries any high-T
    peak — measured 30.6 K vs real 203 K on the real multi-field heat-capacity file). Order 4,
    not the quadratic locate_jump
    uses: over a wide range the Debye knee leaves a quadratic residual at the low-T
    endpoint (~14.6 on a Debye-like curve) that dominates a real high-T peak (~8) and
    fails the interior check; a quartic follows the knee while a peak still stands out.
    Tc_seed = prominence-weighted centroid of the argmax NEIGHBORHOOD (i±2) — a global
    top-k centroid mixes in far-away noise points and drifts (187 K vs 203 K measured).
    Seed only; does not gate reality."""
    T, cp = _sorted_finite(T, cp)
    out = {"Tc_seed": None, "interior": False, "resid": np.array([]), "T": T, "cp": cp}
    if T.size < 6:
        return out
    coef = np.polyfit(T, cp, 4)
    resid = cp - np.polyval(coef, T)
    out["resid"] = resid
    # Preferred seed: the raw Cp INTERIOR argmax (the lambda Cp-peak convention). On a
    # narrow group dominated by the anomaly (the real file's in-field 180-210 K windows) the quartic
    # bends INTO the peak and its residual argmax lands on a shoulder (measured 186.8 K vs
    # the true 199.8 K peak); the raw Cp maximum is exactly the peak there. Requires REAL
    # prominence over both endpoints (2x the detrend scatter) so a noise wiggle on a flat
    # plateau top cannot hijack the seed; monotone/featureless data declines to the
    # detrend fallback.
    j = int(np.argmax(cp))
    s = 1.4826 * _mad(resid)
    if 0 < j < (T.size - 1) and cp[j] - max(cp[0], cp[-1]) >= 2.0 * s:
        out["interior"] = True
        lo, hi = max(0, j - 2), min(T.size, j + 3)
        w = np.clip(cp[lo:hi] - cp.min(), 0.0, None)
        out["Tc_seed"] = float(T[j]) if w.sum() <= 0 else float(np.sum(T[lo:hi] * w) / np.sum(w))
        return out
    i = int(np.argmax(resid))
    interior = 0 < i < (T.size - 1) and resid[i] >= resid[0] and resid[i] >= resid[-1]
    out["interior"] = bool(interior)
    if not interior:
        return out
    lo, hi = max(0, i - 2), min(T.size, i + 3)
    w = np.clip(resid[lo:hi] - resid.min(), 0.0, None)
    out["Tc_seed"] = float(T[i]) if w.sum() <= 0 else float(np.sum(T[lo:hi] * w) / np.sum(w))
    return out


def locate_jump(T, cp):
    """Locate a candidate step-jump Tc via a robust two-segment change-point statistic:
    for each interior split index, the absolute difference of segment means (NOT a
    finite-difference derivative -- garbage on scattered data). Seed = midpoint between
    the two temperatures bracketing the best split. Seed only; does not gate reality.

    A raw mean-difference on the untouched data is dominated by any smooth monotone
    curvature in the underlying Cp(T) (e.g. the T^3 lattice background), which swamps
    a genuine jump statistic near the range edges. So: (1) quadratic-detrend first --
    deliberately lower order than a cubic background so the fit does not near-exactly
    cancel it (which would leave a numerically tiny, ill-conditioned residual and an
    unstable ratio); (2) restrict candidate splits away from the extreme edges, since
    2-3 point segments there can trivially match any local pattern and out-score a
    genuine mid-range step."""
    T, cp = _sorted_finite(T, cp)
    out = {"Tc_seed": None, "interior": False, "stat": 0.0, "T": T, "cp": cp}
    n = T.size
    if n < 6:
        return out
    coef = np.polyfit(T, cp, 2)
    resid = cp - np.polyval(coef, T)
    s = np.std(resid) + 1e-30
    margin = max(2, int(round(0.2 * n)))             # keep splits away from the edges
    lo_i, hi_i = margin, n - margin
    best_stat, best_i = -1.0, None
    for i in range(lo_i, hi_i):                       # interior splits only
        lo, hi = resid[:i], resid[i:]
        stat = abs(lo.mean() - hi.mean()) / s
        if stat > best_stat:
            best_stat, best_i = stat, i
    if best_i is not None:
        out["stat"] = float(best_stat)
        out["interior"] = True
        out["Tc_seed"] = float(0.5 * (T[best_i - 1] + T[best_i]))
    return out


def _mad(x):
    x = np.asarray(x, float)
    med = np.median(x)
    return float(np.median(np.abs(x - med))) or float(np.std(x)) or 1e-30


def artifact_filter(T, cp, t_res=0.05):
    """Narrow data-quality artifact filter -- NOT a transition veto. Drops only
    (a) duplicate-T multivalued points ( |dT| < t_res with a large Cp gap ) and
    (b) lone unsupported spikes/dips (deviates strongly from BOTH neighbors in the
    SAME direction). A real transition is a *cluster* of elevated neighboring points
    and always survives: a cluster member has at least one neighbor on its own
    elevated side, so it never satisfies "deviates from both neighbors" simultaneously."""
    T, cp = _sorted_finite(T, cp)
    n = T.size
    keep = np.ones(n, dtype=bool)
    advisories = []
    if n < 4:
        return {"T": T, "cp": cp, "dropped": 0, "advisories": advisories}
    scale = 5.0 * _mad(cp)
    # (a) duplicate-T multivaluedness: near-equal T with large Cp gap -> drop the outlier of the pair.
    # The reference must be LOCAL (nearest points outside the pair), not the whole-series median:
    # a genuine transition-cluster point sits far from the global baseline by design, so a
    # global-median reference would systematically drop the cluster member and keep the spurious
    # background duplicate. Using the immediate outside neighbors keeps the cluster member (close
    # to its cluster neighbors) and drops the one far from its local surroundings.
    for i in range(1, n):
        if abs(T[i] - T[i - 1]) < t_res and abs(cp[i] - cp[i - 1]) > scale:
            outside = []
            if i - 2 >= 0:
                outside.append(cp[i - 2])
            if i + 1 < n:
                outside.append(cp[i + 1])
            ref = float(np.mean(outside))
            drop = i if abs(cp[i] - ref) > abs(cp[i - 1] - ref) else i - 1
            keep[drop] = False
    # (b) lone spikes/dips: deviates strongly from BOTH neighbors in the SAME direction
    # (a cluster has neighbor support; a genuine monotone step has opposite-sign neighbor diffs).
    for i in range(1, n - 1):
        if not keep[i]:
            continue
        d_lo = cp[i] - cp[i - 1]; d_hi = cp[i] - cp[i + 1]
        up = d_lo > scale and d_hi > scale
        down = d_lo < -scale and d_hi < -scale
        if up or down:
            keep[i] = False
    dropped = int((~keep).sum())
    if dropped:
        advisories.append(f"artifact filter dropped {dropped} point(s) (duplicate-T / lone spike/dip)")
    return {"T": T[keep], "cp": cp[keep], "dropped": dropped, "advisories": advisories}


def _model_record(params, sigma, pcov, param_names, bounds, cp, yhat):
    rss = float(np.sum((cp - yhat) ** 2))
    # Floor RSS against float64 round-off on essentially-exact (noiseless) fits. Without this, an
    # over-parameterized model (e.g. background+lambda vs background-only) can shave RSS by mere
    # numerical round-off (~1e-30 on O(1) data) and spuriously "win" the AICc gate despite the extra
    # params carrying zero real signal -- exactly the manufactured-peak failure mode this gate exists
    # to prevent. The floor sits far below any realistic instrument noise (relative 1e-12 of the
    # data's own sum-of-squares) but far above double-precision round-off, so it only clips this
    # degenerate synthetic-data case and never touches genuinely noisy (real) data.
    rss = max(rss, 1e-12 * (float(np.sum(cp ** 2)) or 1.0))
    n = int(cp.size); k = len(param_names)
    _, _, aicc = _information_criteria(rss, n, k)
    return {"params": {nm: float(v) for nm, v in zip(param_names, params)},
            "sigma": {nm: float(s) for nm, s in zip(param_names, sigma)},
            "pcov": (np.asarray(pcov, float).tolist() if pcov is not None else None),
            "param_names": list(param_names),
            "bounds": [list(bounds[0]), list(bounds[1])],
            "rss": rss, "n": n, "k": k,
            "aicc": (float(aicc) if aicc is not None else float("inf")),
            "r2": _r2(cp, yhat)}


_MIN_WING_PTS_SIDE = 3   # wing points required on EACH side of the inner mask


def local_window(T, cp, Tc_seed, *, wing_mask_k, wing_frac, span_mult):
    """Slice a local window around the candidate. W = max(wing_mask_k, wing_frac*Tc_seed)
    (a 203 K anomaly has ~K-wide wings; a fixed ±2 K mask under-covers it). Window =
    |T - Tc_seed| <= span_mult*W. ok=False when either wing side has < _MIN_WING_PTS_SIDE
    points — decline rather than fit an unconstrained background."""
    T, cp = _sorted_finite(T, cp)
    W = max(float(wing_mask_k), float(wing_frac) * float(Tc_seed))
    m = np.abs(T - Tc_seed) <= span_mult * W
    Tl, cl = T[m], cp[m]
    inner = np.abs(Tl - Tc_seed) <= W
    lo_w = int(np.sum((~inner) & (Tl < Tc_seed))); hi_w = int(np.sum((~inner) & (Tl > Tc_seed)))
    ok = lo_w >= _MIN_WING_PTS_SIDE and hi_w >= _MIN_WING_PTS_SIDE
    reason = "" if ok else f"insufficient wing support ({lo_w} below / {hi_w} above)"
    return {"T": Tl, "cp": cl, "W": W, "inner": inner, "ok": ok, "reason": reason}


def wing_poly(T, cp, inner_mask, order):
    """Low-order polynomial fitted on the WINGS only (candidate window excluded) so it
    cannot bend into the anomaly. None when the wings cannot constrain it."""
    keep = ~np.asarray(inner_mask, bool)
    if int(keep.sum()) < order + 2:
        return None
    return np.polyfit(np.asarray(T, float)[keep], np.asarray(cp, float)[keep], order)


def _fit_background_only(Tl, cl, inner, order):
    coef = wing_poly(Tl, cl, inner, order)
    if coef is None:
        return None, None
    yhat = np.polyval(coef, Tl)
    names = [f"c{j}" for j in range(order + 1)]
    lo = [-np.inf] * len(names); hi = [np.inf] * len(names)
    sig = [float("nan")] * len(names)                 # wing-fixed: no covariance claimed
    rec = _model_record(list(coef), sig, None, names, (lo, hi), cl, yhat)
    return rec, coef


def _fit_transition_models(Tl, cl, *, form, alpha, Tc_seed, W, inner, order,
                           amp_max_frac=1.0, aicc_margin=2.0):
    """Joint fit on the LOCAL window against the wing-FIXED polynomial background: only the
    anomaly's parameters are free (Tc, Aplus, Aminus for lambda; Tc, dC for jump), so the
    anomaly can only explain the interior — the background cannot bend into the peak.
    Returns None when the wings cannot constrain the polynomial."""
    bg_rec, coef = _fit_background_only(Tl, cl, inner, order)
    if bg_rec is None:
        return None
    # The wing poly's coefficients are FIXED from the wings — they buy zero freedom on the
    # interior points the anomaly competes over, so the background counts k=0. (Counting
    # them free, k=order+1, inverts the gate: on equal RSS the transition model would win
    # by parsimony and fabricate on featureless data.)
    bg_rec["k"] = 0
    _, _, bg_aicc0 = _information_criteria(bg_rec["rss"], int(cl.size), 0)
    bg_rec["aicc"] = float(bg_aicc0) if bg_aicc0 is not None else float("inf")
    models = {"background": bg_rec}
    base = np.polyval(coef, Tl)
    amp_max = float(amp_max_frac) * float(np.ptp(cl) or 1.0)
    poly_names = [f"c{j}" for j in range(order + 1)]

    if form == "lambda":
        names = poly_names + ["Tc", "Aplus", "Aminus"]
        def _m(Tx, Tc, Ap, Am):
            return np.polyval(coef, Tx) + lambda_anomaly(Tx, Tc, alpha, Ap, Am)
        # Exclude only the clamp neighborhood |t| < 2*_T_EPS around the seed, where
        # lambda_anomaly is flat by regularization. A wider cut (the old 0.02) scales with
        # Tc and at ~200 K removes the ENTIRE +-4 K peak core, leaving Tc unconstrained
        # (measured: fit drifts to the seed-window bound and rails).
        t = (Tl - Tc_seed) / Tc_seed
        fit_m = np.abs(t) >= 2.0 * _T_EPS
        Tf, cf = (Tl[fit_m], cl[fit_m]) if fit_m.sum() >= 6 else (Tl, cl)
        p0 = [Tc_seed, 0.1 * amp_max, 0.1 * amp_max]
        # Amplitudes are NON-NEGATIVE: a lambda anomaly is an excess-Cp peak on both
        # branches. Allowing negative amplitudes lets one branch dip below the background
        # and the cusp center AWAY from the data peak (measured on the real file at 2 T: fitted Tc
        # 203.9 K vs the densely-sampled data peak at 199.8 K).
        lo = [Tc_seed - W, 0.0, 0.0]
        hi = [Tc_seed + W, amp_max, amp_max]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", OptimizeWarning)
                popt, pcov = curve_fit(_m, Tf, cf, p0=p0, bounds=(lo, hi), maxfev=20000)
            params = list(coef) + list(popt)
            sig = [float("nan")] * len(poly_names) + list(np.sqrt(np.diag(pcov)))
            full_lo = [-np.inf] * len(poly_names) + lo
            full_hi = [np.inf] * len(poly_names) + hi
            models["lambda"] = _model_record(params, sig, pcov, names, (full_lo, full_hi),
                                             cl, _m(Tl, *popt))
            models["lambda"]["k"] = 3                       # only anomaly params are free
            _, _, aicc = _information_criteria(models["lambda"]["rss"], int(cl.size), 3)
            models["lambda"]["aicc"] = float(aicc) if aicc is not None else float("inf")
        except Exception as e:
            models["lambda_error"] = repr(e)                # honest label (was except: pass)

    if form == "jump":
        names = poly_names + ["Tc", "dC"]
        # A mean-field jump offsets the ENTIRE below-Tc branch, wings included: a single
        # poly across both wings is mis-specified (it splits the offset and fakes wing
        # scatter). Fit poly+step JOINTLY on the wings — the step column is constant per
        # wing side for any candidate Tc inside the inner window, so this stays a linear
        # LS and the background still cannot bend into the interior.
        wings = ~np.asarray(inner, bool)
        step_col = np.where(Tl < Tc_seed, 1.0, 0.0)
        Xw = np.vstack([Tl ** j for j in range(order, -1, -1)] + [step_col]).T
        theta, *_ = np.linalg.lstsq(Xw[wings], cl[wings], rcond=None)
        coef_j = theta[:-1]
        base_j = np.polyval(coef_j, Tl)
        resid_j = cl - base_j
        # Candidate Tc's = MIDPOINTS between consecutive data points (within +-W of the
        # seed), not the data points themselves: a candidate AT a data point puts that
        # point exactly at the step (0.5*dC by convention) and every candidate carries a
        # half-step misfit, so even an exact synthetic step never fits cleanly.
        Tu = np.unique(Tl)
        mids = 0.5 * (Tu[:-1] + Tu[1:])
        cand = np.unique(mids[np.abs(mids - Tc_seed) <= W])
        best, rss_profile = None, []
        for Tc in cand:
            s = jump_step(Tl, Tc, 1.0)
            dC = float(np.dot(s, resid_j) / (np.dot(s, s) or 1.0))
            dC = float(np.clip(dC, -amp_max, amp_max))
            yhat = base_j + dC * s
            rss = float(np.sum((cl - yhat) ** 2))
            rss_profile.append((Tc, rss))
            if best is None or rss < best[1]:
                best = (Tc, rss, dC, yhat)
        if best is not None:
            Tc, rss, dC, yhat = best
            prof = np.array(rss_profile); within = prof[prof[:, 1] <= rss * 2.0, 0]
            if within.size >= 2:
                sig_Tc = float((within.max() - within.min()) / 2.0)
            elif cand.size >= 2:
                # profile drops to (numerically) zero at the single best candidate: Tc is
                # resolution-limited, sigma = half the local candidate spacing, not inf
                sig_Tc = float(np.min(np.abs(np.diff(cand))) / 2.0)
            else:
                sig_Tc = float("inf")
            params = list(coef_j) + [Tc, dC]
            sig = [float("nan")] * len(poly_names) + [sig_Tc, float("inf")]
            lo = [-np.inf] * len(poly_names) + [float(cand.min()), -amp_max]
            hi = [np.inf] * len(poly_names) + [float(cand.max()), amp_max]
            models["jump"] = _model_record(params, sig, None, names, (lo, hi), cl, yhat)
            models["jump"]["k"] = 2
            _, _, aicc = _information_criteria(models["jump"]["rss"], int(cl.size), 2)
            models["jump"]["aicc"] = float(aicc) if aicc is not None else float("inf")
            models["jump"]["coef"] = coef_j

    bg_aicc = models["background"]["aicc"]
    chosen_key = "background"
    if form in models and (bg_aicc - models[form]["aicc"]) >= aicc_margin:
        chosen_key = form
    grid = np.linspace(float(Tl.min()), float(Tl.max()), 200)
    cp_fit = np.polyval(coef, grid)
    if chosen_key == "lambda":
        p = models["lambda"]["params"]
        cp_fit = cp_fit + lambda_anomaly(grid, p["Tc"], alpha, p["Aplus"], p["Aminus"])
    elif chosen_key == "jump":
        p = models["jump"]["params"]
        cp_fit = np.polyval(models["jump"]["coef"], grid) + jump_step(grid, p["Tc"], p["dC"])
    return {"models": models, "chosen_key": chosen_key, "coef": coef,
            "t_grid": grid.tolist(), "cp_fit": np.asarray(cp_fit, float).tolist()}


def _anomaly_fwhm(grid, cp_fit, coef, Tc, W):
    """Half-max width of the fitted anomaly component (fit curve minus wing poly);
    fallback to W when degenerate."""
    g = np.asarray(grid, float); comp = np.asarray(cp_fit, float) - np.polyval(coef, g)
    pk = float(comp.max())
    if not np.isfinite(pk) or pk <= 0:
        return float(W)
    above = g[comp >= 0.5 * pk]
    return float(max(W, (above.max() - above.min()) / 2.0)) if above.size >= 2 else float(W)


def fit_transition(T, cp, *, form, universality, lattice_t5=False,
                   wing_mask_k=2.0, wing_frac=0.03, span_mult=5.0, wing_order=3,
                   prominence_n=4.0, collapse_margin=2.0, amp_max_frac=1.0,
                   aicc_margin=2.0, rel_sigma=1.0, bound_rail_frac=0.01):
    """Public entry: locate + fit a transition candidate on a LOCAL window against a
    wing-fixed polynomial background, then apply the tc_determined HARD gate. Produces the
    g["transition"] contract (spec Sec.6.1). Never fabricates a Tc: any failure of the
    interior / AICc-margin / rel-sigma / rail checks yields tc_determined=False with an
    explanatory advisory, so downstream plots/report/export render the point hollow rather
    than claiming a determined transition. lattice_t5 is accepted but inert (the wing
    polynomial replaces the global gammaT+betaT^3[+deltaT^5] background)."""
    alpha = UNIVERSALITY.get(universality, 0.0)
    af = artifact_filter(T, cp)
    Tw, cw = af["T"], af["cp"]
    advisories = list(af["advisories"])
    if lattice_t5:
        advisories.append("lattice_t5 ignored: local wing background")
    empty = {"attempted": False, "form": form, "universality": universality, "Tc": None,
             "Tc_sigma": None, "tc_determined": False, "params": {}, "sigmas": {},
             "aicc": None, "aicc_bg": None, "delta_aicc": None, "railed": [], "identifiable": False,
             "t_data": Tw.tolist(), "cp_data": cw.tolist(), "cp_fit": [], "grid": [],
             "resid_signal": [], "compare": None, "advisories": advisories,
             "prominence": None, "prominence_floor": None, "collapse_delta_aicc": None}
    if Tw.size < 8:
        empty["advisories"].append("too few points after artifact filter")
        return empty
    loc = locate_lambda(Tw, cw) if form == "lambda" else locate_jump(Tw, cw)
    if not loc.get("interior") or loc.get("Tc_seed") is None:
        empty["advisories"].append("no interior transition candidate located")
        return empty
    # Width-informed inner half-width: the Tc-scaled formula max(wing_mask_k, wing_frac*Tc)
    # under-covers a genuinely BROAD anomaly (width 12 K at Tc=40 -> W=2), putting the
    # "wings" inside the anomaly so the wing poly absorbs it. Estimate the anomaly's
    # half-max half-width from the locator's detrend residual around the seed and widen W
    # accordingly (capped to a quarter of the data span so the window stays local). Narrow
    # peaks and lone outliers give a sub-K estimate and are unaffected.
    W_est = 0.0
    r_loc, T_loc = np.asarray(loc.get("resid", [])), np.asarray(loc.get("T", []))
    if r_loc.size and r_loc.size == T_loc.size:
        k = int(np.argmin(np.abs(T_loc - loc["Tc_seed"])))
        pk = float(r_loc[k])
        if pk > 0:
            above = r_loc >= 0.5 * pk
            lo_i = k
            while lo_i > 0 and above[lo_i - 1]:
                lo_i -= 1
            hi_i = k
            while hi_i < r_loc.size - 1 and above[hi_i + 1]:
                hi_i += 1
            W_est = 0.5 * float(T_loc[hi_i] - T_loc[lo_i])
    W_base = max(float(wing_mask_k), min(W_est, 0.25 * float(Tw.max() - Tw.min())))
    win = local_window(Tw, cw, loc["Tc_seed"], wing_mask_k=W_base,
                       wing_frac=wing_frac, span_mult=span_mult)
    if not win["ok"]:
        empty["advisories"].append(f"local window: {win['reason']}")
        return empty
    Tl, cl, Wwin = win["T"], win["cp"], win["W"]
    res = _fit_transition_models(Tl, cl, form=form, alpha=alpha, Tc_seed=loc["Tc_seed"],
                                 W=Wwin, inner=win["inner"], order=wing_order,
                                 amp_max_frac=amp_max_frac, aicc_margin=aicc_margin)
    if res is None:
        empty["advisories"].append("wing polynomial unconstrained")
        return empty
    ck = res["chosen_key"]; ch = res["models"][ck]; bg = res["models"]["background"]
    coef = res["coef"]
    aicc_bg = _fin(bg["aicc"]); aicc_ch = _fin(ch["aicc"])
    delta_aicc = (aicc_bg - aicc_ch) if (aicc_bg is not None and aicc_ch is not None) else None
    _resid_arr0 = cl - np.polyval(coef, Tl)
    resid = np.where(np.isfinite(_resid_arr0), _resid_arr0, 0.0).tolist()   # JSON-safe

    tc_determined = False; Tc = None; Tc_sigma = None; railed = []; identifiable = False
    prominence = None; prominence_floor = None; collapse_daicc = None
    if ck == form:
        ident = _identifiability(ch["param_names"], ch["params"], ch["sigma"],
                                 (ch["bounds"][0], ch["bounds"][1]), rel_sigma, bound_rail_frac)
        Tc = _fin(ch["params"].get("Tc")); Tc_sigma = _fin(ch["sigma"].get("Tc"))
        interior = Tc is not None and (Tl.min() < Tc < Tl.max())
        # Bracket check: a Tc perched near the data edge with too few raw points on one side
        # (e.g. an anomaly jammed against the low-T boundary) cannot be genuinely bracketed --
        # curve_fit's covariance-based sigma can still read deceptively tight on noiseless data
        # even when only 1-2 points anchor one branch, so this is an independent, non-negotiable
        # check (not folded into rel-sigma) that both branches actually have support.
        # Counted on the LOCAL window points (all the fit ever saw).
        bracketed = (Tc is not None and int(np.sum(Tl < Tc)) >= _MIN_BRACKET_PTS
                    and int(np.sum(Tl > Tc)) >= _MIN_BRACKET_PTS)
        tc_ok = ident.get("Tc", {}).get("ok", False)
        amp_names = ["Aplus", "Aminus"] if form == "lambda" else ["dC"]
        amp_railed = any(ident.get(a, {}).get("railed", False) for a in amp_names)
        railed = [nm for nm in ch["param_names"] if ident.get(nm, {}).get("railed")]
        identifiable = bool(tc_ok)
        # HARD gate 1 — prominence: the anomaly must stand above the wing scatter.
        # Form-specific signal: for lambda the residual peak above the wing poly near Tc;
        # for jump the step height |dC| itself (a step has no residual "peak", and the
        # plain wing poly straddles the offset wings, so its wing residual is model
        # mis-specification, not scatter — use the jump model's OWN wing residuals).
        resid_arr = np.asarray(resid, float)          # cl - polyval(coef, Tl)
        near = np.abs(Tl - Tc) <= Wwin if Tc is not None else np.zeros(Tl.size, bool)
        if form == "jump":
            yhat_j = np.polyval(res["models"]["jump"]["coef"], Tl) \
                     + jump_step(Tl, Tc, ch["params"]["dC"]) if Tc is not None else None
            scatter_src = (cl - yhat_j)[~near] if yhat_j is not None else np.array([])
            prominence = abs(_fin(ch["params"].get("dC")) or 0.0)
        else:
            scatter_src = resid_arr[~near]
            # SECOND-highest near-Tc residual: a real anomaly is a CLUSTER of elevated
            # points, so its runner-up is nearly as high as its peak; a single stray point
            # (measured: 2-65K Cp @ 50 kOe, one point +0.018 at 6.42 K on a smooth curve,
            # invisible to the global-MAD artifact filter) has no runner-up and fails.
            if near.any():
                near_sorted = np.sort(resid_arr[near])
                prominence = float(near_sorted[-2]) if near_sorted.size >= 2 else float(near_sorted[-1])
            else:
                prominence = None
        wing_scatter = 1.4826 * _mad(scatter_src) if int((~near).sum()) >= 4 else float("inf")
        prominence_floor = float(prominence_n) * float(wing_scatter)
        prom_ok = prominence is not None and np.isfinite(prominence_floor) \
                  and prominence >= prominence_floor
        if not prom_ok:
            advisories.append(f"prominence {0.0 if prominence is None else prominence:.3g} "
                              f"below floor {prominence_floor:.3g}")
        tc_determined = bool(interior and bracketed and tc_ok and not amp_railed and prom_ok)
        if not tc_determined and prom_ok:
            advisories.append("Tc not determined (interior/bracket/rel-σ/rail gate)")
        # HARD gate 2 (lambda only) — localized-improvement collapse: refit with points
        # within ±delta of Tc removed; a REAL localized anomaly loses most of its evidence
        # (advantage collapses), background inadequacy keeps winning broadly (advantage
        # persists -> reject). Relative criterion: the residual advantage must fall below
        # max(collapse_margin, _COLLAPSE_FRAC * original advantage) — lambda tails carry
        # real signal past the FWHM (measured: true anomalies retain <=0.29x, background
        # inadequacy >=0.45x), so a purely absolute margin misfires on strong anomalies.
        # A jump's evidence is inherently NON-local (whole-branch offset), so point
        # removal around Tc cannot collapse it even when real — gate does not apply.
        if tc_determined and Tc is not None and form == "lambda":
            delta = _anomaly_fwhm(res["t_grid"], res["cp_fit"], coef, Tc, Wwin)
            keep = np.abs(Tl - Tc) > delta
            collapse_ok = True
            if int(np.sum(Tl[keep] < Tc)) >= _MIN_BRACKET_PTS and \
               int(np.sum(Tl[keep] > Tc)) >= _MIN_BRACKET_PTS and int(keep.sum()) >= 8:
                res2 = _fit_transition_models(Tl[keep], cl[keep], form=form, alpha=alpha,
                                              Tc_seed=Tc, W=Wwin, inner=np.abs(Tl[keep] - Tc) <= Wwin,
                                              order=wing_order, amp_max_frac=amp_max_frac,
                                              aicc_margin=aicc_margin)
                if res2 is not None and form in res2["models"]:
                    collapse_daicc = float(res2["models"]["background"]["aicc"]
                                           - res2["models"][form]["aicc"])
                    ceiling = max(float(collapse_margin),
                                  _COLLAPSE_FRAC * float(delta_aicc or 0.0))
                    collapse_ok = collapse_daicc < ceiling
            if not collapse_ok:
                advisories.append(f"advantage persists after near-Tc removal "
                                  f"(ΔAICc {collapse_daicc:.1f}) — background inadequacy, not a transition")
            tc_determined = tc_determined and collapse_ok
    elif loc.get("interior"):
        advisories.append("broad feature unresolved (located but AICc did not beat background)")

    return {"attempted": True, "form": form, "universality": universality,
            "Tc": Tc, "Tc_sigma": Tc_sigma, "tc_determined": tc_determined,
            "params": {k: _fin(v) for k, v in ch["params"].items()},
            "sigmas": {k: _fin(v) for k, v in ch["sigma"].items()},
            "aicc": aicc_ch, "aicc_bg": aicc_bg, "delta_aicc": _fin(delta_aicc),
            "railed": railed, "identifiable": identifiable,
            "t_data": Tl.tolist(), "cp_data": cl.tolist(),
            "cp_fit": np.where(np.isfinite(res["cp_fit"]), res["cp_fit"], 0.0).tolist(),
            "grid": res["t_grid"], "resid_signal": resid,
            "compare": None, "advisories": advisories,
            "prominence": _fin(prominence), "prominence_floor": _fin(prominence_floor),
            "collapse_delta_aicc": _fin(collapse_daicc)}


def compare_transition_forms(T, cp, *, universality, lattice_t5=False,
                             wing_mask_k=2.0, wing_frac=0.03, span_mult=5.0, wing_order=3,
                             prominence_n=4.0, collapse_margin=2.0, amp_max_frac=1.0,
                             aicc_margin=2.0, rel_sigma=1.0, bound_rail_frac=0.01,
                             indistinguishable_band=2.0):
    """Opt-in comparison: fit BOTH the lambda and jump forms via fit_transition and report
    which, if either, is preferred by AICc. On sparse relaxation-type data, lambda-vs-jump is
    often genuinely non-discriminable -- the honest verdict is "indistinguishable on this
    data" whenever neither form's AICc beats the other's by more than indistinguishable_band
    (or when either fit declined, i.e. aicc is None). Never a coin-flip winner."""
    kw = dict(universality=universality, lattice_t5=lattice_t5,
              wing_mask_k=wing_mask_k, wing_frac=wing_frac, span_mult=span_mult,
              wing_order=wing_order, prominence_n=prominence_n,
              collapse_margin=collapse_margin, amp_max_frac=amp_max_frac,
              aicc_margin=aicc_margin, rel_sigma=rel_sigma,
              bound_rail_frac=bound_rail_frac)
    gl = fit_transition(T, cp, form="lambda", **kw)
    gj = fit_transition(T, cp, form="jump", **kw)
    al, aj = gl.get("aicc"), gj.get("aicc")
    if al is None or aj is None:
        return {"lambda": gl, "jump": gj, "delta_aicc": None,
                "verdict": "indistinguishable on this data"}
    d = float(aj - al)   # >0 => lambda lower AICc (better)
    if abs(d) <= indistinguishable_band:
        verdict = "indistinguishable on this data"
    else:
        verdict = "lambda" if d > 0 else "jump"
    return {"lambda": gl, "jump": gj, "delta_aicc": d, "verdict": verdict}
