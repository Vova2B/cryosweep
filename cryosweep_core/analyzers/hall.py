from __future__ import annotations
import hashlib, pathlib
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.sweeps import segment_sweeps
from cryosweep_core.fitting.transport import LinearFitModel
from cryosweep_core.result import Result, Provenance
from cryosweep_core.registry import Need
from cryosweep_core.io.loader import load_dat
from cryosweep_core.grouping import cluster_field_setpoints

E_CHG = 1.602176634e-19     # Coulomb
_OE_PER_T = 10000.0


# ---- typed result models ---------------------------------------------------
class HallTempPoint(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temperature: float
    n_points: int = 0
    # Stage A (raw R_xy vs B linear fit)
    slope_raw_ohm_per_T: float | None = None
    r2_raw: float | None = None
    R_H_raw: float | None = None
    # Stage B (antisymmetrized — trusted)
    antisymmetrized: bool = False
    slope_ohm_per_T: float | None = None
    r2: float | None = None
    R_H: float | None = None            # m^3/C, from Stage B
    # Stage C (derived from Stage B R_H)
    carrier_n: float | None = None      # 1/m^3
    carrier_type: str | None = None     # electrons | holes
    rho_xx: float | None = None         # Ohm*m (longitudinal, matched to this T)
    mobility: float | None = None       # m^2/(V*s)
    # SP-2: per-T field-sweep arrays (additive; for plotting only)
    field_raw_T: list[float] = []       # signed B (Tesla), finite-masked, len == n_points
    R_xy_raw: list[float] = []          # raw R_xy (Ohm), same order
    field_asym_T: list[float] = []      # positive |B| (Tesla) at antisym grid
    R_asym: list[float] = []            # antisymmetrized R (Ohm)
    asym_intercept_ohm: float | None = None   # Stage-B fit intercept (None if Stage B skipped)
    # SP-#2: per-T LONGITUDINAL field sweep (additive; for the two-panel plot only)
    field_rxx_T: list[float] = []       # signed B (Tesla), finite-masked
    R_xx_raw: list[float] = []          # longitudinal resistance (Ohm), same order
    # --- 2026-08-10 uncertainty-honesty additive fields (spec §2.1, declared LAST:
    # append-only JSON key order). Residual (fit-quality) sigma from the linregress stderr
    # that _stage_fit previously discarded. None (never 0.0) at n_points < 3 — zero residual
    # DOF (U4). carrier_n_sigma/mobility_sigma are exact relative propagation from r_h_sigma
    # of the trusted stage; rho_xx sigma is NOT folded into mobility_sigma (deferred §10). ---
    slope_sigma_raw_ohm_per_T: float | None = None   # Stage A residual sigma (Ohm/T)
    r_h_sigma_raw: float | None = None               # Stage A: sigma_slope * thickness (m^3/C)
    slope_sigma_ohm_per_T: float | None = None       # Stage B (antisym) residual sigma (Ohm/T)
    r_h_sigma: float | None = None                   # Stage B: sigma_slope * thickness (m^3/C)
    carrier_n_sigma: float | None = None             # 1/m^3 (relative propagation from r_h_sigma)
    mobility_sigma: float | None = None              # m^2/(V*s) (rho_xx sigma NOT folded, §10)
    sigma_zero_dof: bool = False                     # trusted stage had < 3 points (U4)
    # #20 (2026-09-02, append-only): decline reasons for Stage C. ["antisym_r_h_missing"]
    # = Stage B produced no R_H, so carrier_n/carrier_type/mobility are WITHHELD rather
    # than derived from the untrusted Stage A raw fit. Empty list = nothing withheld.
    derived_flags: list[str] = []

class Capability(BaseModel):
    model_config = ConfigDict(extra="ignore")
    name: str
    applicable: bool
    reason: str = ""

class HallData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    probe: str = "hall"
    hall_channel: int | None = None
    thickness_m: float | None = None
    geometry_sign: int = 1
    longitudinal_source: str | None = None     # "same_file:chN" | "file:<path>:chN" | None
    points: list[HallTempPoint] = []
    capabilities: list[Capability] = []


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""


# ---- pure helpers ----------------------------------------------------------
def _antisymmetrize(H, R):
    """R_asym(H) = [R(+H) - R(-H)]/2 over the positive-H overlap, via interpolation
    (tolerant of non-symmetric / unevenly-spaced sweeps). Returns (H_pos, R_asym).
    NOTE: on a concatenated up+down loop the same |H| appears on both branches; np.interp
    after argsort silently averages the two branches (correct for negligible-hysteresis
    samples; a future maintainer with a hysteretic sample should branch-separate first)."""
    H = np.asarray(H, float); R = np.asarray(R, float)
    m = np.isfinite(H) & np.isfinite(R)
    H, R = H[m], R[m]
    order = np.argsort(H)
    Hs, Rs = H[order], R[order]
    hi = min(Hs.max(), -Hs.min())                 # symmetric overlap
    if hi <= 0:
        return np.empty(0), np.empty(0)
    Hp = np.unique(np.abs(Hs[(Hs > 0) & (Hs <= hi)]))
    if Hp.size < 2:
        return np.empty(0), np.empty(0)
    r_pos = np.interp(Hp, Hs, Rs)
    r_neg = np.interp(-Hp, Hs, Rs)
    return Hp, (r_pos - r_neg) / 2.0

def _stage_fit(H, R, thickness_m, geometry_sign):
    """Linear fit R vs B (B=H/10000); returns slope (Ohm/T), r2, R_H = slope*thickness*sign."""
    H = np.asarray(H, float); R = np.asarray(R, float)
    m = np.isfinite(H) & np.isfinite(R)
    H, R = H[m], R[m]
    if H.size < 2 or np.ptp(H) == 0:
        return None
    B = H / _OE_PER_T
    fit = LinearFitModel().fit(B, R, xunit="T", yunit="Ohm")
    slope = fit.params["slope"]
    R_H = (slope * thickness_m * geometry_sign) if thickness_m else None
    out = {"slope_ohm_per_T": float(slope), "r2": float(fit.r2),
           "intercept": float(fit.params["intercept"]),
           "R_H": (float(R_H) if R_H is not None else None), "n_points": int(H.size)}
    # 2026-08-10 spec §2.1: the residual slope sigma was already computed by linregress and
    # previously discarded here. n < 3 -> zero residual DOF, linregress stderr 0.0 (measured):
    # 0.0 would assert perfect certainty, so it is None + sigma_zero_dof (U4).
    if H.size < 3:
        out["slope_sigma_ohm_per_T"] = None
        out["r_h_sigma"] = None
        out["sigma_zero_dof"] = True
    else:
        ssig = float(fit.sigma["slope"])
        ssig = ssig if np.isfinite(ssig) else None
        out["slope_sigma_ohm_per_T"] = ssig
        out["r_h_sigma"] = (ssig * thickness_m) if (ssig is not None and thickness_m) else None
    return out

def _carrier_n(R_H):
    if not R_H:                                   # None or 0
        return None, None
    n = 1.0 / (E_CHG * abs(R_H))
    return float(n), ("electrons" if R_H < 0 else "holes")

def _mobility(R_H, rho_xx):
    if not R_H or not rho_xx:
        return None
    return float(abs(R_H) / rho_xx)               # |R_H| * sigma = |R_H| / rho_xx

def _long_rho_xx(df, cmap, long_channel, long_df, long_cmap):
    """Return a callable T -> rho_xx (Ohm*m) by interpolating the longitudinal channel's
    instrument resistivity column over temperature, or None if no longitudinal data.
    long_df/long_cmap: a SEPARATE file's frame/columns; if None, use df/cmap (same file)."""
    if long_channel is None:
        return None
    src_df, src_cmap = (long_df, long_cmap) if long_df is not None else (df, cmap)
    rk = f"resistivity_ch{long_channel}"
    if rk not in src_cmap.logical or "temperature" not in src_cmap.logical:
        return None
    T = pd.to_numeric(src_df[src_cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    rho = pd.to_numeric(src_df[src_cmap.logical[rk]], errors="coerce").to_numpy(float)
    m = np.isfinite(T) & np.isfinite(rho) & (rho > 0)
    if m.sum() < 1:
        return None
    Tg, Rg = T[m], rho[m]
    order = np.argsort(Tg)
    Tg, Rg = Tg[order], Rg[order]
    # collapse duplicate temperatures by mean so np.interp has a monotone grid
    uT = np.unique(Tg)
    uR = np.array([Rg[Tg == t].mean() for t in uT])
    def rho_at(temp):
        return float(np.interp(temp, uT, uR))     # np.interp clamps outside the range
    return rho_at


def _capabilities(points, has_thickness, long_source) -> list[Capability]:
    any_anti = any(p.antisymmetrized for p in points)
    any_RH = any(p.R_H is not None for p in points)
    any_mu = any(p.mobility is not None for p in points)
    return [
        Capability(name="hall_coefficient", applicable=any_RH,
                   reason="Stage A/B R_xy(B) line fits with thickness" if any_RH
                   else ("thickness required for R_H (slope-only)" if not has_thickness
                         else "no field loop could be fit")),
        Capability(name="antisymmetrization", applicable=any_anti,
                   reason="field loops contain both +H and -H" if any_anti
                   else "no loop spans both field signs; Stage B skipped"),
        Capability(name="carrier_concentration", applicable=any_RH,
                   reason="n = 1/(e|R_H|) from Stage B" if any_RH else "needs R_H"),
        Capability(name="mobility", applicable=any_mu,
                   reason=f"mu = |R_H|/rho_xx ({long_source})" if any_mu
                   else "no longitudinal channel/file supplied for rho_xx"),
        # Recognized-but-deferred (2026-09-05): decomposing rho_xy = R0*B + R_s*mu0*M
        # requires M(H) of the SAME sample, which no file in this corpus provides. See
        # docs/physics-reference.md, "Anomalous Hall effect".
        Capability(name="anomalous_hall", applicable=False,
                   reason="requires M(H) of the same sample measured in a magnetometer "
                          "(VSM/MPMS); without it R0 and the anomalous term are not "
                          "separable and no partial extraction is defensible"),
    ]


def field_sweep_points(df, cmap, cfg, hc, thickness_m, rho_fn) -> list[HallTempPoint]:
    """Per-held-T field-sweep Hall points (Stage A raw + Stage B antisym + carrier/mobility).
    Pure: no I/O, no cfg mutation. Reused by HallAnalyzer and the temp-dep dual-method block."""
    T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    H = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
    Rxy = pd.to_numeric(df[cmap.logical[f"resistance_ch{hc.hall_channel}"]], errors="coerce").to_numpy(float)
    # SP-#2: same-file longitudinal resistance column (for per-T R_xx(B) field sweep)
    long_ch = hc.longitudinal_channel
    lkey = f"resistance_ch{long_ch}" if long_ch is not None else None
    Rxx_all = (pd.to_numeric(df[cmap.logical[lkey]], errors="coerce").to_numpy(float)
               if lkey is not None and lkey in cmap.logical else None)
    fsegs = [s for s in segment_sweeps(df, cmap, cfg) if s.swept.name == "field"]
    # KNOWN-ISSUES #19 (2026-09-02): a held temperature is decided ACROSS segments, not
    # per segment. round(T, 1) bins to a grid, and every grid has edges: on the real Hall
    # file the 200 K loop arrived as segments at 199.8521 / 199.9904 / 199.9945 K, which
    # straddled the 199.9/200.0 edge and split into a 46-point fragment (R_H None) beside
    # the real 136-point group. cluster_field_setpoints single-link-clusters the setpoints
    # actually present (no edges to straddle), then labels each cluster with
    # setpoint_key(cluster median) — cluster to GROUP, round to LABEL, same as the VSM
    # field-setpoint fix (3d722ff). abs_floor=0.25 K matches StabilityCfg.drift_max
    # temperature: setpoints closer than the allowed hold drift are indistinguishable
    # from drift, while the 0.5 K-spaced low-T holds (4.5 vs 5.0 K) stay distinct.
    withT = [(s, float(t)) for s in fsegs
             if (t := s.setpoint.get("temperature")) is not None]
    labels = cluster_field_setpoints([t for _s, t in withT],
                                     rel_tol=1e-3, abs_floor=0.25)
    by_T: dict = {}
    for (s, _t), lab in zip(withT, labels):
        if np.isfinite(lab):
            by_T.setdefault(float(lab), []).append(s)
    points: list[HallTempPoint] = []
    for Tset in sorted(by_T):
        idx = np.concatenate([s.idx for s in by_T[Tset]])
        Hh, Rr = H[idx], Rxy[idx]
        raw = _stage_fit(Hh, Rr, thickness_m, hc.geometry_sign)
        if raw is None:
            continue
        pt = HallTempPoint(temperature=float(Tset), n_points=raw["n_points"],
                           slope_raw_ohm_per_T=raw["slope_ohm_per_T"], r2_raw=raw["r2"],
                           R_H_raw=raw["R_H"],
                           slope_sigma_raw_ohm_per_T=raw["slope_sigma_ohm_per_T"],
                           r_h_sigma_raw=raw["r_h_sigma"])
        # SP-2: persist the finite-masked raw sweep (same mask _stage_fit fits over,
        # so len(field_raw_T) == n_points).
        mfin = np.isfinite(Hh) & np.isfinite(Rr)
        pt.field_raw_T = (Hh[mfin] / _OE_PER_T).tolist()
        pt.R_xy_raw = Rr[mfin].tolist()
        # SP-#2: persist the per-T longitudinal R_xx(B) sweep over the SAME segment index;
        # empty when no same-file longitudinal channel is supplied (no regression).
        if Rxx_all is not None:
            Rxx_seg = Rxx_all[idx]
            mlong = np.isfinite(Hh) & np.isfinite(Rxx_seg)
            pt.field_rxx_T = (Hh[mlong] / _OE_PER_T).tolist()
            pt.R_xx_raw = Rxx_seg[mlong].tolist()
        Hp, R_asym = _antisymmetrize(Hh, Rr)
        anti = _stage_fit(Hp, R_asym, thickness_m, hc.geometry_sign) if Hp.size >= 2 else None
        if anti is not None:
            pt.antisymmetrized = True
            pt.slope_ohm_per_T = anti["slope_ohm_per_T"]; pt.r2 = anti["r2"]; pt.R_H = anti["R_H"]
            pt.slope_sigma_ohm_per_T = anti["slope_sigma_ohm_per_T"]
            pt.r_h_sigma = anti["r_h_sigma"]
            # SP-2: persist the antisym grid + Stage-B intercept for the fit line.
            pt.field_asym_T = (Hp / _OE_PER_T).tolist()
            pt.R_asym = R_asym.tolist()
            pt.asym_intercept_ohm = anti["intercept"]
        # #20 (2026-09-02): Stage C derives ONLY from the trusted Stage B R_H. The old
        # fallback to R_H_raw published a carrier density and mobility beside an empty
        # R_H cell (Stage A still carries the even-in-B admixture that antisymmetrization
        # exists to remove — on real data ~100x the Hall signal). Decline discipline
        # (cf. the resistivity power-law decline): withhold the derived quantities and
        # carry a machine-readable reason; R_H_raw stays visible for transparency.
        if pt.R_H is None and pt.R_H_raw is not None:
            pt.derived_flags = ["antisym_r_h_missing"]
        pt.carrier_n, pt.carrier_type = _carrier_n(pt.R_H)
        # 2026-08-10 spec §2.1: sigma propagated by relative sigma (n = 1/(e|R_H|) and
        # mu = |R_H|/rho_xx are pure reciprocal/scale). Stage B only, like the values.
        trusted = anti if anti is not None else raw
        pt.sigma_zero_dof = bool(trusted.get("sigma_zero_dof", False))
        if pt.R_H and pt.r_h_sigma is not None:
            rel = pt.r_h_sigma / abs(pt.R_H)
            if pt.carrier_n is not None:
                pt.carrier_n_sigma = float(pt.carrier_n * rel)
        if rho_fn is not None:
            pt.rho_xx = rho_fn(Tset)          # longitudinal measurement, independent of R_H
            pt.mobility = _mobility(pt.R_H, pt.rho_xx)
            if (pt.mobility is not None and pt.R_H and pt.r_h_sigma is not None):
                pt.mobility_sigma = float(pt.mobility * pt.r_h_sigma / abs(pt.R_H))
        points.append(pt)
    return points


_REL_SIGMA_WARN = 0.5   # closed O4: flag (never drop) antisym points with rel sigma > 50 %


def sigma_noise_warnings(points) -> list[str]:
    """Always-on warnings for points whose R_H carries > 50 % relative sigma
    (closed O4 threshold — flag, never drop). Measured trigger: QD example ch2 300 K
    (slope -1.43 +- 2.21 Ohm/T, rel 154 %, r2 0.0021, previously reported clean).

    F6 (final-review): this gated on `r_h_sigma` alone, so a temperature point that has no
    antisymmetrised stage — R_H and its sigma both come from raw Stage A — was EXEMPT from a
    warning §2.1 calls "always-on". Stage A is the noisier stage (it still carries the
    even-in-H R_xx admixture, which §2.2 notes can be ~100x the Hall signal), i.e. exactly
    the branch that most needs the warning. It now tests the TRUSTED stage's sigma, the same
    one `carrier_n_sigma`/`mobility_sigma` were already computed from, and names the stage."""
    out = []
    for p in points:
        trusted_sig = p.r_h_sigma if p.antisymmetrized else p.r_h_sigma_raw
        R_H_trusted = p.R_H if p.R_H is not None else p.R_H_raw
        stage = "Stage B antisym" if p.antisymmetrized else "Stage A raw"
        if R_H_trusted and trusted_sig is not None and R_H_trusted != 0:
            rel = trusted_sig / abs(R_H_trusted)
            r2 = p.r2 if p.antisymmetrized else p.r2_raw
            if rel > _REL_SIGMA_WARN:
                # F16 (final-review): name the sigma FAMILY. In a slice whose thesis is that
                # residual and instrument sigma must never share a name, a bare "relative
                # sigma" was the loose one; the hall_tdep sibling already says "relative
                # instrument sigma". This one is the residual (fit-scatter) sigma.
                r2txt = "n/a" if r2 is None else f"{r2:.3f}"
                out.append(f"R_H at T = {p.temperature:.1f} K carries {rel * 100:.0f}% "
                           f"relative residual sigma ({stage} fit scatter, r² = {r2txt}) — "
                           f"treat as noise, not a carrier density")
    return out


class HallAnalyzer:
    probe = "hall"
    needs = (Need("hall_channel", scope="sample", required=True),
             Need("thickness_mm", scope="sample", required=False),
             Need("longitudinal_channel", scope="sample", required=False),
             Need("longitudinal_file", scope="sample", required=False))

    def analyze(self, rawtable, cfg) -> Result:
        df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
        header = rawtable.header
        hc = cfg.hall
        prov = Provenance(file=getattr(rawtable, "path", None) or header.title or "",
                          sha256=_sha256(getattr(rawtable, "path", None)),
                          app_version=header.app_version, config=cfg.model_dump(mode="json"))
        if hc.hall_channel is None:
            return Result(status="error", errors=["hall_channel required (which bridge is the Hall signal)"],
                          data={"probe": "hall"}, provenance=prov)
        rkey = f"resistance_ch{hc.hall_channel}"
        if rkey not in cmap.logical or "temperature" not in cmap.logical or "field" not in cmap.logical:
            return Result(status="error",
                          errors=[f"hall channel {hc.hall_channel} resistance column / T / H not found"],
                          data={"probe": "hall"}, provenance=prov)
        T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
        H = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
        Rxy = pd.to_numeric(df[cmap.logical[rkey]], errors="coerce").to_numpy(float)
        if np.isfinite(Rxy).sum() == 0:
            return Result(status="error", errors=[f"hall channel {hc.hall_channel} is empty"],
                          data={"probe": "hall"}, provenance=prov)
        thickness_m = (hc.thickness_mm * 1e-3) if hc.thickness_mm else None

        # longitudinal source for mobility
        long_df = long_cmap = None
        long_source = None
        if hc.longitudinal_file:
            lrt = load_dat(hc.longitudinal_file)
            long_df, long_cmap = canonicalize_columns(lrt.df, lrt.header)
            long_source = f"file:{pathlib.Path(hc.longitudinal_file).name}:ch{hc.longitudinal_channel}"
        elif hc.longitudinal_channel is not None:
            long_source = f"same_file:ch{hc.longitudinal_channel}"
        rho_fn = _long_rho_xx(df, cmap, hc.longitudinal_channel, long_df, long_cmap)

        points = field_sweep_points(df, cmap, cfg, hc, thickness_m, rho_fn)

        if not points:
            return Result(status="low_confidence", confidence=0.2,
                          warnings=["no field loops found to fit"],
                          data={"probe": "hall", "reason": "no field loops"}, provenance=prov)
        caps = _capabilities(points, thickness_m is not None, long_source)
        hd = HallData(probe="hall", hall_channel=hc.hall_channel, thickness_m=thickness_m,
                      geometry_sign=hc.geometry_sign, longitudinal_source=long_source,
                      points=points, capabilities=caps)
        r2s = [p.r2 for p in points if p.r2 is not None]
        if thickness_m is None:
            conf, status = 0.4, "low_confidence"
        elif r2s:
            conf = float(np.mean(r2s)); status = "ok" if conf >= cfg.confidence_min else "low_confidence"
        else:
            conf, status = 0.4, "low_confidence"
        return Result(status=status, confidence=conf,
                      confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                        "fit": (float(np.mean(r2s)) if r2s else None)},
                      warnings=sigma_noise_warnings(points),
                      data=hd.model_dump(mode="json"), provenance=prov)
