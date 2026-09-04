from __future__ import annotations
import hashlib, pathlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.io.loader import load_dat
from cryosweep_core.fitting.heat_capacity import fit_lowt_models, fit_full_range, fit_schottky, fit_delta_h_overlay
from cryosweep_core.fitting.transitions import fit_transition, compare_transition_forms
from cryosweep_core.result import Result, FitResult, Provenance, Gate
from cryosweep_core.registry import Need

_LOWT_MAX_K = 10.0

class HCData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    probe: str = "heatcapacity"
    temperature: list[float] = []
    cp: list[float] = []                  # J/(mol*K)
    cp_over_t: list[float] = []
    t_squared: list[float] = []
    field_setpoint: float | None = None
    fit: FitResult | None = None
    model: str | None = None                 # chosen low-T model key
    lowt_fits: list[dict] = []               # every fitted model: {key,label,ok,r2,params,theta_D,...}
    full_fit: dict | None = None
    full_fit_available: bool = False
    full_fit_reason: str = ""
    comparison: dict | None = None
    full_temperature: list[float] = []
    full_cp: list[float] = []
    field_groups: list[dict] = []
    schottky_enabled: bool = False
    schottky_overlay: dict | None = None
    transitions_enabled: bool = False
    tc_h: list[dict] = []
    n_atoms: float | None = None
    n_atoms_available: bool = False
    entropy_temperature: list[float] = []
    entropy_total: list[float] = []
    entropy_magnetic: list[float | None] | None = None
    entropy_available: bool = False
    entropy_reason: str = ""
    entropy_extrapolated: bool = False
    entropy_lattice_source: str | None = None
    entropy_rln_suggestion: dict | None = None

def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""


def _chosen_by(fits, key):
    ok = [f for f in fits if f.get("ok") and f.get(key) is not None and np.isfinite(f[key])]
    return min(ok, key=lambda f: f[key])["key"] if ok else None


def _chosen_aicc(fits, margin=2.0):
    """Identifiability-gated, parsimony-aware AICc pick (C5). On exact/degenerate data the overfit
    spin models are non-identifiable (beta<->A collinear) and are excluded, so the simplest *resolvable*
    model wins instead of a floating-point coin-flip. Within `margin` AICc of the best, prefer fewer params."""
    cand = [f for f in fits if f.get("ok") and f.get("aicc") is not None
            and np.isfinite(f["aicc"]) and f.get("identifiable", False)]
    if not cand:                                            # nothing resolvable -> fall back to any AICc
        cand = [f for f in fits if f.get("ok") and f.get("aicc") is not None and np.isfinite(f["aicc"])]
    if not cand:                                            # no finite AICc at all -> BIC
        return _chosen_by(fits, "bic")
    best = min(f["aicc"] for f in cand)
    near = [f for f in cand if f["aicc"] <= best + margin]
    return min(near, key=lambda f: f["n_params"])["key"]


def _lowt_upturn_warning(Tlow, Clow, fits):
    """Flag a rising Cp/T as T^2->0 (nuclear-Schottky signature) via the debye_t3 residual.
    C2: floored so it CANNOT fire on a near-exact fit (the multifield fixture's in-field groups are
    exact lattice lines -> rms ~ 1e-16; an unfloored `low_mean > rms` would coin-flip a false positive)."""
    d = next((f for f in fits if f["key"] == "debye_t3" and f.get("ok")), None)
    if d is None:
        return []
    g = d["params"]["gamma"]; b = d["params"]["beta"]
    x = np.asarray(Tlow, float) ** 2; y = np.asarray(Clow, float) / np.asarray(Tlow, float)
    resid = y - (g + b * x)
    rms = float(np.sqrt(np.mean(resid ** 2)))
    sig = float(np.mean(np.abs(y)))                        # scale of Cp/T
    if sig <= 0 or rms <= 1e-6 * sig:                      # near-exact fit -> no meaningful upturn
        return []
    k = min(2, resid.size)
    low_mean = float(np.mean(resid[np.argsort(x)[:k]]))    # residual at the two lowest-T^2 points
    # require BOTH statistical (>2*rms) and physical (>2% of gamma) significance
    if low_mean > 2.0 * rms and low_mean > 0.02 * abs(g):
        return ["possible low-T upturn / nuclear-Schottky contamination; gamma may be inflated"]
    return []


def _cross_field_warnings(groups):
    fit = [(g["field_oe"], next((f for f in g["fits"]
            if f["key"] == "debye_t3" and f.get("ok")), None)) for g in groups
           if g["status"] == "ok"]
    fit = [(h, f) for h, f in fit if f is not None]
    warns = []
    thetas = [f["theta_D"] for _, f in fit if np.isfinite(f.get("theta_D", float("nan")))]
    if len(thetas) >= 2:
        spread = (max(thetas) - min(thetas)) / max(np.mean(thetas), 1e-30)
        if spread > 0.10:                                  # >10% lattice theta_D drift across fields
            warns.append(f"theta_D drift across fields ({spread*100:.0f}%): lattice should be "
                         "field-independent; possible fit contamination")
    if any(f["params"].get("gamma", 0.0) < 0 for _, f in fit):
        warns.append("gamma(H) goes negative at one or more fields (unphysical electronic term)")
    return warns


def _attempt_transition(g, Tg, Cg, hccfg):
    if not hccfg.transitions_enabled:
        return
    tkw = dict(form=hccfg.transition_form, universality=hccfg.transition_universality,
               lattice_t5=hccfg.transition_lattice_t5,
               wing_mask_k=hccfg.transition_wing_mask_k,
               wing_frac=hccfg.transition_wing_frac,
               span_mult=hccfg.transition_span_mult,
               wing_order=hccfg.transition_wing_order,
               prominence_n=hccfg.transition_prominence_n,
               collapse_margin=hccfg.transition_collapse_margin,
               amp_max_frac=hccfg.transition_amp_max_frac,
               aicc_margin=hccfg.transition_aicc_margin,
               rel_sigma=hccfg.identifiability_rel_sigma,
               bound_rail_frac=hccfg.bound_rail_frac)
    try:
        g["transition"] = fit_transition(Tg, Cg, **tkw)
        if hccfg.transition_compare_forms:
            g["transition"]["compare"] = compare_transition_forms(
                Tg, Cg, **{k: v for k, v in tkw.items() if k != "form"},
                indistinguishable_band=hccfg.transition_indistinguishable_band)
    except Exception as exc:                       # decline-not-crash: one bad group must
        g["transition"] = {                        # not take down the whole analysis
            "attempted": False, "form": hccfg.transition_form,
            "universality": hccfg.transition_universality, "Tc": None, "Tc_sigma": None,
            "tc_determined": False, "params": {}, "sigmas": {}, "aicc": None,
            "aicc_bg": None, "delta_aicc": None, "railed": [], "identifiable": False,
            "t_data": [], "cp_data": [], "cp_fit": [], "grid": [], "resid_signal": [],
            "compare": None, "advisories": [f"transition fit crashed: {exc!r}"],
            "prominence": None, "prominence_floor": None, "collapse_delta_aicc": None}


def _group_lowt_model(g):
    """(model_key, params) for a group's chosen low-T fit, or None if no usable fit."""
    key = g.get("chosen_aicc_key")
    if not key:
        return None
    f = next((f for f in g.get("fits", []) if f.get("key") == key and f.get("ok")), None)
    return (key, dict(f["params"])) if f is not None else None


def _build_field_groups(T_all, C_all, F_all, hccfg, n_atoms, lo, hi):
    if F_all is None:
        return [], []
    absF = np.abs(F_all)
    binw = hccfg.field_bin_koe * 1000.0
    bins = np.round(absF / binw)
    uniq = sorted(np.unique(bins).tolist())
    if len(uniq) < 2:
        return [], []                                      # single-field -> no engine output
    groups = []
    for b in uniq:
        sel = bins == b
        Tg, Cg = T_all[sel], C_all[sel]
        order = np.argsort(Tg); Tg, Cg = Tg[order], Cg[order]
        field_oe = float(np.median(absF[sel]))
        low = Tg <= hi
        if lo is not None:
            low = low & (Tg >= lo)
        n_lowt = int(low.sum())
        g = {"field_oe": field_oe, "n_lowt": n_lowt, "is_primary": False,
             "status": "ok", "fits": [], "chosen_aicc_key": None,
             "chosen_bic_key": None, "warnings": [], "t2": [], "cp_over_t": [],
             "full_temperature": Tg.tolist(), "full_cp": Cg.tolist(), "entropy": None}
        # Transition attempt is DECOUPLED from the low-T sufficiency gate: a high-T-only
        # group (e.g. an in-field sweep measured only around the transition) has no low-T
        # points to fit gamma/theta_D from, but its Cp(T) window is exactly where T_c(H)
        # lives — attach the attempt BEFORE the insufficiency continue.
        _attempt_transition(g, Tg, Cg, hccfg)
        if n_lowt < hccfg.min_lowt_per_field:
            g["status"] = "insufficient"
            groups.append(g); continue
        Tl, Cl = Tg[low], Cg[low]
        g["t2"] = (Tl ** 2).tolist()                        # I4: raw per-field points for the overlay
        g["cp_over_t"] = (Cl / Tl).tolist()
        fitset = fit_lowt_models(Tl, Cl, n_atoms, extended=True,
                                 rel_sigma=hccfg.identifiability_rel_sigma,
                                 bound_rail_frac=hccfg.bound_rail_frac,
                                 corr_warn=hccfg.corr_warn)
        g["fits"] = fitset["fits"]
        g["chosen_aicc_key"] = _chosen_aicc(g["fits"])      # C5: identifiability-gated, parsimony-aware
        g["chosen_bic_key"] = _chosen_by(g["fits"], "bic")
        g["warnings"] = _lowt_upturn_warning(Tg[low], Cg[low], g["fits"])
        if hccfg.schottky_enabled:
            dt3 = next((f for f in g["fits"] if f["key"] == "debye_t3" and f.get("ok")), None)
            # Use dt3 seeds only when beta>0 (physical); a contaminated Schottky-in-data fit
            # yields beta<0 and inflated gamma which then corrupt the Schottky seeding.
            _dt3_usable = dt3 and dt3["params"].get("beta", -1) > 0
            g0 = dt3["params"]["gamma"] if _dt3_usable else 0.005
            b0 = dt3["params"]["beta"] if _dt3_usable else 2e-4
            g["schottky"] = fit_schottky(
                Tg, Cg, gamma0=g0, beta0=b0, r=hccfg.schottky_r,
                lattice_t5=hccfg.schottky_lattice_t5,
                include_nuclear=hccfg.schottky_include_nuclear,
                nuclear_max_tmin_k=hccfg.schottky_nuclear_max_tmin_k,
                fit_max_k=hccfg.schottky_fit_max_k, delta_max_k=hccfg.schottky_delta_max_k,
                f_max=hccfg.schottky_f_max, rel_sigma=hccfg.identifiability_rel_sigma,
                bound_rail_frac=hccfg.bound_rail_frac, peak_corr_max=hccfg.schottky_peak_corr_max,
                is_lowest_field=(b == uniq[0]), aicc_margin=hccfg.schottky_aicc_margin)
        groups.append(g)
    groups[0]["is_primary"] = True                         # lowest |field| bin
    return groups, _cross_field_warnings(groups)


def _primary_group_cp(rawtable):
    """(T_ascending, Cp J/(mol*K)) for the lowest-|field| group of a HC .dat, mirroring
    analyze()'s primary-group extraction. Returns None when unreadable."""
    df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
    if "hc_sample" not in cmap.logical or "temperature" not in cmap.logical:
        return None
    temp = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    cp = pd.to_numeric(df[cmap.logical["hc_sample"]], errors="coerce").to_numpy(float) * 1e-3
    field = (pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
             if "field" in cmap.logical else None)
    m = np.isfinite(temp) & np.isfinite(cp) & (temp > 0)
    if field is not None:
        m &= np.isfinite(field)
    if not m.any():
        return None
    Tk, Ck = temp[m], cp[m]
    if field is not None:
        absF = np.abs(field[m])
        bins = np.round(absF / 1000.0)              # same 1 kOe primary-group bin as analyze()
        grp = bins == bins.min()
        Tk, Ck = Tk[grp], Ck[grp]
    order = np.argsort(Tk)
    return Tk[order], Ck[order]


def _reference_lattice_cp(path, full_temperature):
    """Load a reference HC .dat, extract its primary-group Cp(T), interpolate onto
    `full_temperature`. Outside the reference T-overlap the lattice is NaN, so compute_entropy
    genuinely truncates the magnetic curve there (emitting None at those out-of-overlap
    temperatures rather than a flat plateau). Returns a list aligned row-for-row to
    `full_temperature`, or None on any load/extraction failure (caller falls back to fit)."""
    try:
        rt = load_dat(path)
        pg = _primary_group_cp(rt)
    except Exception:
        return None
    if pg is None:
        return None
    Tref, Cref = pg
    Tref, idx = np.unique(Tref, return_index=True)  # unique ascending T -> well-defined interpolant
    Cref = Cref[idx]
    if Tref.size < 2:
        return None
    ft = np.asarray(full_temperature, float)
    lat = np.interp(ft, Tref, Cref)
    lat[(ft < Tref.min()) | (ft > Tref.max())] = np.nan
    return lat.tolist()


class HCAnalyzer:
    probe = "heatcapacity"
    needs = (Need("n_atoms", scope="header", required=False),)

    def analyze(self, rawtable, cfg) -> Result:
        df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
        header = rawtable.header
        prov = Provenance(file=getattr(rawtable, "path", None) or header.title or "",
                          sha256=_sha256(getattr(rawtable, "path", None)),
                          app_version=header.app_version, config=cfg.model_dump(mode="json"))
        if "hc_sample" not in cmap.logical:
            return Result(status="error", errors=["no Samp HC column"], data={"probe": "heatcapacity"}, provenance=prov)
        temp = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
        cp_mJ = pd.to_numeric(df[cmap.logical["hc_sample"]], errors="coerce").to_numpy(float)
        cp = cp_mJ * 1e-3                                # mJ/(mol*K) -> J/(mol*K); already molar (no per-mol norm)
        field = (pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
                 if "field" in cmap.logical else None)
        m = np.isfinite(temp) & np.isfinite(cp) & (temp > 0)
        if field is not None:
            m &= np.isfinite(field)
        if not m.any():
            return Result(status="low_confidence", confidence=0.2,
                          data={"probe": "heatcapacity", "reason": "no usable (T, Cp) points"}, provenance=prov)
        Tk, Ck = temp[m], cp[m]
        T_all, C_all = temp[m].copy(), cp[m].copy()
        F_all = field[m].copy() if field is not None else None
        # Heat capacity is a set of (T, Cp, H) relaxation points, NOT a smooth ramp: selecting fit
        # data via sweep-segmentation fragments multi-field datasets and drops the zero-field low-T
        # points (real bug: a low-T-only Cp file scored 0.13/0.30). theta_D is a zero-field lattice
        # property, so
        # select the lowest-|field| setpoint group directly. Bin |field| to 1 kOe so Tesla-scale
        # setpoints (0 / 1 T=10 kOe / 3 T ...) separate cleanly while a zero-field cluster (~0-50 Oe)
        # stays whole.
        field_sp = None
        if field is not None:
            absF = np.abs(field[m])
            bins = np.round(absF / 1000.0)  # fixed 1 kOe bin for primary group (slice-1 oracle); field_bin_koe scopes multi-field engine only
            grp = bins == bins.min()
            Tk, Ck = Tk[grp], Ck[grp]
            field_sp = float(np.median(field[m][grp]))
        order = np.argsort(Tk)                           # store ascending for clean plotting
        Tk, Ck = Tk[order], Ck[order]
        hccfg = cfg.heatcapacity
        _lo = hccfg.lowt_fit_min_k
        _hi = hccfg.lowt_fit_max_k if hccfg.lowt_fit_max_k is not None else _LOWT_MAX_K
        low = Tk <= _hi
        if _lo is not None:
            low = low & (Tk >= _lo)
        warnings = []
        n_atoms_available = header.n_atoms is not None    # capture BEFORE the n=1 default fires
        n_atoms_header = header.n_atoms                    # None when absent (real n, not the default)
        n_atoms = header.n_atoms
        if n_atoms is None:
            n_atoms = 1.0; warnings.append("no ATOMS in header; theta_D uses n=1 (theta_D scales as n^(1/3))")
        if low.sum() < 5:
            return Result(status="low_confidence", confidence=0.3, warnings=warnings + ["<5 low-T points"],
                          data={"probe": "heatcapacity", "reason": "insufficient low-T data"}, provenance=prov)
        fitset = fit_lowt_models(Tk[low], Ck[low], n_atoms, parsimony_r2=cfg.hc_parsimony_r2)
        fit = fitset["chosen"]
        if fit is None:
            return Result(status="low_confidence", confidence=0.3,
                          warnings=warnings + ["low-T fit failed for all models"],
                          data={"probe": "heatcapacity", "reason": "all low-T fits failed"}, provenance=prov)
        chosen_key = fitset["chosen_key"]
        hd = HCData(probe="heatcapacity", temperature=Tk[low].tolist(), cp=Ck[low].tolist(),
                    cp_over_t=(Ck[low] / Tk[low]).tolist(), t_squared=(Tk[low] ** 2).tolist(),
                    field_setpoint=field_sp, fit=fit, model=chosen_key, lowt_fits=fitset["fits"])
        hd.n_atoms = n_atoms_header                       # real n (None if header had no ATOMS)
        hd.n_atoms_available = n_atoms_available
        fg, cross_warn = _build_field_groups(T_all, C_all, F_all, hccfg, n_atoms, _lo, _hi)
        hd.field_groups = fg
        warnings += cross_warn
        hd.schottky_enabled = hccfg.schottky_enabled
        if hccfg.schottky_enabled and hccfg.schottky_delta_h_model != "none":
            pts = [(g["field_oe"], g["schottky"]["params"]["Delta"]) for g in fg
                   if g.get("status") == "ok" and g.get("schottky", {}).get("delta_determined")]
            if len(pts) >= 3:
                hd.schottky_overlay = fit_delta_h_overlay([p[0] for p in pts], [p[1] for p in pts],
                                                          model=hccfg.schottky_delta_h_model)
        hd.transitions_enabled = hccfg.transitions_enabled
        if hccfg.transitions_enabled:
            # ANY group with a determined Tc contributes — including status="insufficient"
            # (low-T-gated) groups, whose transition attempt is decoupled from the low-T fit
            hd.tc_h = [{"field_oe": g["field_oe"], "Tc": g["transition"]["Tc"],
                        "Tc_sigma": g["transition"]["Tc_sigma"], "form": g["transition"]["form"],
                        "tc_determined": g["transition"]["tc_determined"]}
                       for g in fg if g.get("transition", {}).get("tc_determined")]
        # --- full-range Debye-Einstein fit on the same lowest-field group (additive; never
        #     downgrades the low-T result). n fixed to the SAME n_atoms used for low-T theta_D
        #     so the theta_D comparison is apples-to-apples. ---
        Tg, Cg = Tk, Ck                                  # full group (ascending), not just `low`
        hd.full_temperature = Tg.tolist(); hd.full_cp = Cg.tolist()   # full-group data for cp_vs_t plot
        full = None; avail = False; reason = ""
        if Tg.size < hccfg.full_min_points:
            reason = f"<{hccfg.full_min_points} points in group"
        elif float(Tg.max()) < hccfg.full_max_t_min_k:
            reason = f"T_max {Tg.max():.1f} K < {hccfg.full_max_t_min_k:.0f} K"
        else:
            avail = True
            init = dict(hccfg.full_init); init["n"] = float(n_atoms)
            seed = {"gamma": fit.params.get("gamma"), "theta_D": fit.params.get("theta_D")}
            full = fit_full_range(Tg, Cg, init=init, fixed=dict(hccfg.full_fixed),
                                  fit_min_k=hccfg.full_fit_min_k, fit_max_k=hccfg.full_fit_max_k,
                                  seed=seed)
            if not full.get("ok"):
                warnings.append(f"full-range fit failed: {full.get('reason','')}")
            elif full.get("r2") is not None and full["r2"] < hccfg.full_min_r2:
                full["ok"] = False
                full["reason"] = (f"fit did not converge to a usable solution "
                                  f"(r²={full['r2']:.3g} < {hccfg.full_min_r2})")
                warnings.append(f"full-range fit rejected: r²={full['r2']:.3g} below {hccfg.full_min_r2}")
        lowt_theta = fit.params.get("theta_D")
        lowt_is_lattice = chosen_key in ("debye_t3", "debye_t3_t5")
        full_ok = full if (full and full.get("ok")) else {}
        comparison = {
            "gamma": {"lowt": fit.params.get("gamma"),
                      "full": full_ok.get("params", {}).get("gamma")},
            "theta_D": {"lowt": (lowt_theta if (lowt_is_lattice and lowt_theta is not None
                                                and np.isfinite(lowt_theta)) else "n/a"),
                        "full": full_ok.get("params", {}).get("theta_D")},
            "r2": {"lowt": fit.r2, "full": full_ok.get("r2")},
        }
        hd.full_fit = full; hd.full_fit_available = avail
        hd.full_fit_reason = reason; hd.comparison = comparison
        # --- entropy S(T) (additive; never downgrades the fit result) ---
        from cryosweep_core.fitting.entropy import compute_entropy, suggest_rln
        from cryosweep_core.fitting.heat_capacity import specific_heat_full, _FULL_PARAMS
        lattice_cp = None; lat_src = None; lat_params = None
        if full and full.get("ok"):
            lat_params = {k: full["params"][k] for k in _FULL_PARAMS}
            lattice_cp = specific_heat_full(np.asarray(hd.full_temperature, float),
                                            **lat_params).tolist()
            lat_src = "fit"
        # Reference-file lattice override: when a reference .dat is configured, its Cp(T)
        # replaces the fitted lattice for the magnetic-entropy subtraction (source="reference").
        ref_path = getattr(hccfg, "entropy_lattice_ref_file", None)
        if ref_path:
            ref_lat = _reference_lattice_cp(ref_path, hd.full_temperature)
            if ref_lat is not None:
                lattice_cp = ref_lat; lat_src = "reference"
            else:
                warnings.append(f"entropy lattice reference file '{ref_path}' failed to load; "
                                "using fitted lattice")
        ent = compute_entropy(hd.full_temperature, hd.full_cp,
                              lowt_model=(chosen_key, dict(fit.params)),
                              lattice_cp=lattice_cp,
                              extrapolate=getattr(hccfg, "entropy_extrapolate", True))
        if lat_src and ent.get("s_magnetic") is not None:
            ent["lattice_source"] = lat_src
        hd.entropy_temperature = ent["temperature"]; hd.entropy_total = ent["s_total"]
        hd.entropy_magnetic = ent["s_magnetic"]; hd.entropy_available = bool(ent["s_total"])
        hd.entropy_reason = ent["reason"]; hd.entropy_extrapolated = ent["extrapolated"]
        hd.entropy_lattice_source = ent["lattice_source"]
        hd.entropy_rln_suggestion = suggest_rln(ent["s_magnetic"])
        if getattr(hccfg, "entropy_rln_j", None) is not None and float(hccfg.entropy_rln_j) > 0:
            import math
            from cryosweep_core.fitting.entropy import rln_match_fields
            j = float(hccfg.entropy_rln_j)
            R = 8.314462618
            val = R * math.log(2 * j + 1)
            # owner-forced level still carries the honest O5 verdict against the data
            hd.entropy_rln_suggestion = {"j": j, "value": val,
                                         "label": f"R ln{int(2 * j + 1)}",
                                         **rln_match_fields(val, ent["s_magnetic"])}
        # Closed O5 always-on warning: a nearest-neighbor suggestion that matches nothing
        # within tolerance is not evidence of a doublet — say so out loud.
        _sug = hd.entropy_rln_suggestion
        if _sug and _sug.get("rel_err") is not None and not _sug.get("matched"):
            _fin = [v for v in (ent["s_magnetic"] or []) if v is not None and np.isfinite(v)]
            if _fin:
                warnings.append(
                    f"S_mag saturation ({float(_fin[-1]):.2f} J/mol/K) matches no "
                    f"R ln(2J+1) within {_sug['tol'] * 100:.0f}% — the {_sug['label']} "
                    f"suggestion is nearest-neighbor only, not evidence of a doublet")
        # per-field entropy: reuse the zero-field lattice, evaluated analytically on each
        # group's own T grid (row-aligned by construction) so per-field magnetic S(T) is possible.
        for g in fg:
            g_lat = None
            if lat_params is not None and g.get("full_temperature"):
                g_lat = specific_heat_full(np.asarray(g["full_temperature"], float),
                                           **lat_params).tolist()
            gm = _group_lowt_model(g)
            ent_g = compute_entropy(g.get("full_temperature", []), g.get("full_cp", []),
                                    lowt_model=gm, lattice_cp=g_lat,
                                    extrapolate=getattr(hccfg, "entropy_extrapolate", True))
            if g_lat is not None and ent_g.get("s_magnetic") is not None:
                ent_g["lattice_source"] = lat_src
            g["entropy"] = ent_g if ent_g["s_total"] else None
        conf = min(1.0, fit.r2) if fit.r2 else 0.5
        beta = fit.params.get("beta")
        is_lattice = chosen_key in ("debye_t3", "debye_t3_t5")
        if beta is not None and beta <= 0 and is_lattice:
            # Debye lattice model with non-physical beta<=0 -> inadequate; theta_D is NaN.
            warnings.append("β≤0: Debye lattice model inadequate (low-T upturn); "
                            "a spin-fluctuation model is needed")
            conf = min(conf, 0.4); status = "low_confidence"
        elif beta is not None and beta <= 0:
            # spin-fluctuation model legitimately can have beta<=0: good fit, theta_D unextractable.
            warnings.append("β≤0: lattice Debye θ_D not extractable; "
                            "low-T Cp is spin-fluctuation-dominated")
            status = "ok" if conf >= cfg.confidence_min else "low_confidence"
        else:
            status = "ok" if conf >= cfg.confidence_min else "low_confidence"
        if fit.params.get("gamma", 0.0) < 0:
            warnings.append("γ<0: unphysical electronic (Sommerfeld) coefficient")
        return Result(status=status, confidence=conf,
                      confidence_parts={"detector": 1.0, "segmentation": 1.0, "fit": fit.r2},
                      warnings=warnings, data=hd.model_dump(mode="json"), provenance=prov)
