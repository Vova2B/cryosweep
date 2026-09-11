from __future__ import annotations
import hashlib, pathlib
from types import SimpleNamespace
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.sweeps import segment_sweeps
from cryosweep_core.fitting.transport import LinearFitModel
from cryosweep_core.result import Result, Provenance, Gate
from cryosweep_core.registry import Need
from cryosweep_core.io.loader import load_dat
from cryosweep_core.grouping import cluster_field_setpoints
from cryosweep_core.analyzers.hall_sigma import row_sigma_R, slope_sigma_ols, skip_row_warning

E_CHG = 1.602176634e-19     # Coulomb
from cryosweep_core.units import OE_PER_T as _OE_PER_T   # single-sourced
from cryosweep_core.units import ZERO_FIELD_OE as _ZERO_FIELD_OE   # single-sourced


# ---- typed result models ---------------------------------------------------
class WithheldDerived(BaseModel):
    """What the decline rule withheld, kept so it can be INSPECTED, never published.

    The canonical carrier_n / carrier_type / mobility fields are None whenever this is
    populated. These are not measurements: at sigma >= |R_H| the +-1 sigma interval on
    R_H contains zero, so n is unbounded above and the carrier sign is undetermined
    (spec Sec 4.1).
    """
    model_config = ConfigDict(extra="ignore")
    carrier_n: float | None = None
    carrier_type: str | None = None
    mobility: float | None = None


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
    # 2026-09-07 (spec §4.3): the (interpolated) median |H| (Oe) of the zero-field-masked
    # longitudinal rows AT THIS POINT's temperature -- a per-point figure, not a file-wide
    # one. mu = |R_H|/rho_xx is a zero-field statement; recording the field makes the claim
    # auditable from the output alone. None whenever rho_xx is also None (no zero-field
    # coverage at this setpoint, review round 1 Important #1).
    rho_xx_field_oe: float | None = None
    # 2026-09-07 (spec §4.2): instrument repeat-noise sigma, the same four names
    # HallTDepPoint uses so one parser reads both envelopes. A WEAKER, DIFFERENT claim
    # than the residual sigma above -- instrument noise, not fit quality. The
    # _instrument suffix is load-bearing.
    slope_sigma_instrument_ohm_per_T: float | None = None
    r_h_sigma_instrument: float | None = None
    carrier_n_sigma_instrument: float | None = None
    mobility_sigma_instrument: float | None = None
    # 2026-09-10 (spec Sec 4.1, append-only): what the r_h_unresolved decline withheld, kept
    # for inspection. None whenever nothing was withheld -- see decline_unresolved() below.
    withheld: WithheldDerived | None = None
    # 2026-09-07 (spec §4.6): field-window ladder. Rungs refit R_asym vs B over
    # |B| <= f * B_max. r_h_spread is max-min over RESOLVED rungs and is None -- never
    # 0.0 -- when fewer than two resolved. Judged against each rung's OWN sigma: narrower
    # rungs have fewer points, and comparing to the full fit's sigma makes sample size
    # look like window sensitivity.
    # Fix round 1, Minor: None (never []) when the ladder did not run, matching the
    # sibling convention (power_law_ladder in resistivity.py, cw_ladder in mag.py) so a
    # parser reading all three envelopes never has to special-case Hall.
    r_h_ladder: list[dict] | None = None
    r_h_spread: float | None = None

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
    # 2026-09-10 (task 4b, append-only): how many leading rows HallCfg.skip_rows dropped
    # before this analysis ran. Always present -- a silent skip is the one thing the
    # design must not do, so a reader never has to be told separately what was excluded.
    skipped_rows: int = 0


def _sha256(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""


# ---- pure helpers ----------------------------------------------------------
def _antisymmetrize(H, R, sigma_R=None):
    """R_asym(H) = [R(+H) - R(-H)]/2 over the positive-H overlap, via interpolation
    (tolerant of non-symmetric / unevenly-spaced sweeps). Returns (H_pos, R_asym, sigma_asym).
    NOTE: on a concatenated up+down loop the same |H| appears on both branches; np.interp
    after argsort silently averages the two branches (correct for negligible-hysteresis
    samples; a future maintainer with a hysteretic sample should branch-separate first).

    sigma_R (optional, per-row, Ohm) is interpolated onto the SAME Hp query grid and
    combined as sigma_asym = sqrt(sigma(+H)^2 + sigma(-H)^2)/2 -- the exact propagation
    for R_asym = [R(+H) - R(-H)]/2 with independent per-branch noise. sigma_asym is None
    whenever sigma_R is None.

    Fix round 1 (2026-09): sigma_R's own finite-mask is kept STRICTLY SEPARATE from H,R's.
    A row can carry a perfectly good resistance reading beside a missing/NaN std-dev (the
    two columns' validity does not track each other on real files), and dropping that row
    from R,H because its sigma is bad would silently move R_asym/R_H/r2/field_asym_T for a
    point Task 4 was only supposed to add a sigma family to, not re-fit. Hp and R_asym are
    therefore computed from the H,R mask alone, bit-identical to before sigma_R existed;
    sigma is interpolated from ITS OWN valid rows (mirroring hall_tempdep's fully
    independent _interp_fixed_field_curves / _interp_fixed_field_sigma_curves). If sigma
    cannot be formed at all (fewer than 2 sigma-valid rows), sigma_asym stays None -- a
    sigma problem never reaches back to move R.

    Note what that interpolation means pointwise: a SINGLE missing/NaN std-dev sitting
    between two valid ones is BACKFILLED from its neighbours, not declined at that grid
    point. So a resolved sigma_asym may contain values standing in for a reading the file
    never supplied. This is deliberate and matches hall_tempdep: instrument noise is a
    smooth, slowly varying property (on a real file the p10-p90 band is 1.5e-6 to 2.3e-6
    Ohm about a 1.9e-6 median), so a neighbour's value is a far better estimate than
    nothing -- but it IS an estimate, and a caller reasoning about which points carry a
    genuinely measured sigma should not assume every entry does."""
    H = np.asarray(H, float); R = np.asarray(R, float)
    m = np.isfinite(H) & np.isfinite(R)
    Hm, Rm = H[m], R[m]
    order = np.argsort(Hm)
    Hs, Rs = Hm[order], Rm[order]
    hi = min(Hs.max(), -Hs.min()) if Hs.size else 0.0   # symmetric overlap
    if hi <= 0:
        return np.empty(0), np.empty(0), None
    Hp = np.unique(np.abs(Hs[(Hs > 0) & (Hs <= hi)]))
    if Hp.size < 2:
        return np.empty(0), np.empty(0), None
    r_pos = np.interp(Hp, Hs, Rs)
    r_neg = np.interp(-Hp, Hs, Rs)
    s_asym = None
    if sigma_R is not None:
        S = np.asarray(sigma_R, float)
        ms = np.isfinite(H) & np.isfinite(S)          # sigma's OWN mask, never mixed with R's
        Hms, Sms = H[ms], S[ms]
        if Hms.size >= 2:
            orders = np.argsort(Hms)
            Hss, Ss = Hms[orders], Sms[orders]
            s_pos = np.interp(Hp, Hss, Ss)
            s_neg = np.interp(-Hp, Hss, Ss)
            s_asym = np.sqrt(s_pos ** 2 + s_neg ** 2) / 2.0
    return Hp, (r_pos - r_neg) / 2.0, s_asym

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
        # Spec §4.5(a): the same zero-residual-DOF fact that makes sigma dishonest at 0.0
        # makes r2 dishonest at 1.0 -- a line through two points fits them exactly no
        # matter how noisy the underlying data is, so r2 == 1.0 here is a tautology, not
        # a measurement.
        out["r2"] = None
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


def resolved_sigma(pt):
    """The sigma the decline rule judges by, or None when there is nothing to judge.

    Spec Sec 4.1: instrument sigma where the file supports it (it is the stronger
    constraint and the one the noise warning uses), residual sigma otherwise. None when
    R_H is absent, or when NEITHER family exists -- an unquantified uncertainty is not
    evidence of a small one, so such a point is treated as unresolved (spec Sec 4.5, which
    reuses this same definition for the confidence fraction)."""
    if getattr(pt, "R_H", None) is None:
        return None
    inst = getattr(pt, "r_h_sigma_instrument", None)
    if inst is not None:
        return inst
    return getattr(pt, "r_h_sigma", None)


def is_resolved(pt) -> bool:
    """True iff R_H's own uncertainty is strictly smaller than R_H itself (spec Sec 4.1:
    sigma >= |R_H| means the +-1 sigma interval on R_H contains zero)."""
    s = resolved_sigma(pt)
    return s is not None and pt.R_H is not None and abs(s) < abs(pt.R_H)


def decline_unresolved(pt) -> None:
    """Withhold carrier_n / carrier_type / mobility (and their sigma companions) in place
    when R_H's own sigma is not resolved (spec Sec 4.1). R_H and its sigma stay visible --
    the fit happened, and hiding it would hide the evidence for the decline. A point with
    no R_H at all is skipped here: it already carries its own decline reason
    (antisym_r_h_missing), and spec Sec 4.1 says it keeps that reason rather than gaining
    a second one. Shared by both Hall analyzers -- HallTempPoint and HallTDepPoint carry
    the identical field set this function touches."""
    if pt.R_H is None or is_resolved(pt):
        return
    pt.withheld = WithheldDerived(carrier_n=pt.carrier_n, carrier_type=pt.carrier_type,
                                  mobility=pt.mobility)
    pt.carrier_n = pt.carrier_type = pt.mobility = None
    pt.carrier_n_sigma = pt.mobility_sigma = None
    pt.carrier_n_sigma_instrument = pt.mobility_sigma_instrument = None
    pt.derived_flags = [*pt.derived_flags, "r_h_unresolved"]


_LADDER_FRACTIONS = (1.00, 0.75, 0.50, 0.25)
_LADDER_MIN_POINTS = 5
#: Floor on the spread, RELATIVE to |R_H| of the full-window rung, so float noise on exact
#: data cannot trip the flag. Relative, not absolute: R_H spans many decades across samples
#: and an absolute floor would be a different rule at each scale (the TTO kappa_ph floor is
#: absolute only because its quantity, an exponent, is already dimensionless).
#: Pinned 2026-09-07 by measurement on the real Hall file: 3*sigma, not the floor, is the
#: binding term at every one of the file's 9 temperatures -- spread/(3*sig_max) ratios ran
#: 0.003-0.087, nowhere near either term, so `window_sensitive` is quiet across the whole
#: file (spec §4.6 expected it "quiet except possibly at 200/300 K"; measurement corrects
#: that to quiet everywhere -- see test_real_file_ladder_is_quiet). The floor exists to
#: backstop the degenerate case 3*sigma cannot cover: an exactly-linear fit where BOTH
#: sigma and spread collapse toward float noise together (the straight synthetic fixture,
#: spread ~1e-23 against a 3-sigma of ~1e-16 -- 3-sigma alone already wins there too, but
#: not by construction). The flag's verdict is identical for any floor from 0.005 to 0.5 on
#: both synthetic fixtures and the real file (test_ladder_floor_is_not_finely_tuned).
_LADDER_REL_FLOOR = 0.05


def _r_h_ladder(Hp, R_asym, S_asym, thickness_m, geometry_sign):
    """Refit R_asym vs B over |B| <= f*B_max for f in _LADDER_FRACTIONS. Returns
    (rungs, spread, flags). spread is max-min over RESOLVED rungs, None (never 0.0) when
    fewer than two resolve.

    RULING (controller audit item 3, 2026-09-07; fix round 1, Important #1): a rung's own
    sigma is judged by calling the SAME `resolved_sigma()`/`is_resolved()` this module
    already uses to decide whether a whole point's R_H is resolved -- not a textual
    reimplementation of their precedence. A rung is a bare `_stage_fit` dict, not a
    HallTempPoint, so it has no `r_h_sigma_instrument` attribute to read; a lightweight
    `SimpleNamespace` carrying the three attributes those two functions actually read
    (R_H, r_h_sigma, r_h_sigma_instrument) is passed through them instead. This is not
    cosmetic: fix round 1 found the first version re-derived the identical precedence
    inline, so nothing enforced the two stayed in sync -- if `is_resolved`'s comparison,
    its zero-sigma handling, or its "neither family available" fallback ever changes, a
    hand-rolled copy would not follow, silently. Routing through the real functions closes
    that gap with no behaviour change (same values in, same verdict out).

    MEASURED on the real Hall file before choosing instrument-preferred over residual-only
    (the audit's two options): the rules are NOT equivalent. Residual sigma excluded ZERO
    rungs at every one of the file's 9 temperatures; instrument sigma excluded the f=0.25
    rung at 2-50 K and both f<=0.5 rungs at 100-300 K (e.g. T=100 K, f=0.25: R_H=7.58e-12,
    residual sigma 6.25e-12 [resolved] vs instrument sigma 2.91e-11 [not]). A residual-only
    rule would report every narrow rung "resolved" on a file whose points the point-level
    rule sometimes calls unresolved -- the same word backed by weaker evidence.

    "ladder_thin" (fix round 1, Important #2): at the three temperatures above where
    instrument sigma excludes BOTH narrower rungs, `good` holds exactly the two WIDEST
    windows -- the least-different pair the ladder can compare, sitting right at the
    `ladder_incomplete` floor of two. `window_sensitive`'s absence there is real (verified:
    even folding the excluded rungs back in, 3*sigma still grows faster than the spread
    does), but it is a verdict over half the window range, not the full f=1.00->0.25 span
    the other six points get -- and nothing distinguished the two before this flag. On a
    noisier sample MORE points would fall to two rungs, so the ladder would read quieter
    exactly as the data gets worse; that anti-correlation is worth flagging even though it
    does not change today's verdict on this file.
    """
    rungs, flags = [], []
    if Hp.size == 0:
        return rungs, None, ["ladder_incomplete"]
    bmax = float(np.max(np.abs(Hp)))
    for f in _LADDER_FRACTIONS:
        m = np.abs(Hp) <= f * bmax * (1 + 1e-12)
        if m.sum() < _LADDER_MIN_POINTS or np.unique(np.abs(Hp[m])).size < 2:
            continue
        fit = _stage_fit(Hp[m], R_asym[m], thickness_m, geometry_sign)
        if fit is None or fit["R_H"] is None:
            continue
        sig_inst = None
        if S_asym is not None:
            ssig_i = slope_sigma_ols(Hp[m] / _OE_PER_T, S_asym[m])
            if ssig_i is not None and thickness_m:
                sig_inst = float(ssig_i * thickness_m)
        # The real is_resolved()/resolved_sigma() -- not a look-alike -- via a stand-in
        # carrying only the attributes those two functions read.
        rung_pt = SimpleNamespace(R_H=fit["R_H"], r_h_sigma=fit.get("r_h_sigma"),
                                  r_h_sigma_instrument=sig_inst)
        sig = resolved_sigma(rung_pt)
        unresolved = not is_resolved(rung_pt)
        # Task 8: which family backed this rung's sigma -- exported verbatim in the
        # sibling .hall_ladder.csv so a reader is never left to guess from magnitude.
        # Same precedence resolved_sigma() applies (instrument preferred): sig_inst is
        # None whenever this window's rung had no usable instrument sigma at all.
        # Fix round 1 (Minor): keyed off `sig` itself, not off sig_inst alone -- a window
        # can drop below _stage_fit's own n<3 floor internally (its isfinite mask can
        # discard positions this loop's cruder window-count already accepted), leaving
        # BOTH families None. Calling that "residual" would claim a family was tried and
        # came back empty, which is not what happened; sigma_kind is None whenever
        # neither family produced a number.
        if sig is None:
            sigma_kind = None
        else:
            sigma_kind = "instrument" if sig_inst is not None else "residual"
        rungs.append({"f": f, "R_H": fit["R_H"], "sigma": sig, "sigma_kind": sigma_kind,
                      "r2": fit["r2"], "n_points": fit["n_points"], "unresolved": unresolved})
    good = [r for r in rungs if not r["unresolved"]]
    if len(good) < 2:
        # A rung whose own sigma is unresolved is not a measurement, and several such rungs
        # agree with each other for the wrong reason -- the resistivity precedent, where
        # bound-pinned rungs faked a window-stable exponent. They stay IN the ladder
        # carrying unresolved: True, so nothing is hidden.
        return rungs, None, ["ladder_incomplete"]
    vals = [r["R_H"] for r in good]
    spread = float(max(vals) - min(vals))
    sig_max = max(abs(r["sigma"]) for r in good)
    full = next((r for r in good if r["f"] == 1.00), good[0])
    floor = _LADDER_REL_FLOOR * abs(full["R_H"])
    if spread > max(3.0 * sig_max, floor):
        flags.append("window_sensitive")
    if len(good) == 2:
        # The spread rests on the two WIDEST windows only -- the minimum before
        # ladder_incomplete would fire instead, and the least-different pair available.
        # Distinct from ladder_incomplete (fewer than two): here a spread IS reported, but
        # over half the window range, not the full ladder every other point compares.
        flags.append("ladder_thin")
    return rungs, spread, flags


# Review round 1 (2026-09-07), Important #3: `derived_flags` tokens for the two distinct
# ways a longitudinal source can fail to produce rho_xx. Never collapse them into one
# message -- "channel missing" means the user pointed at data that isn't there; "no zero
# field" means the data exists but never sat near H=0.
_RHO_XX_CHANNEL_MISSING = "rho_xx_channel_missing"
_RHO_XX_NO_ZERO_FIELD = "rho_xx_no_zero_field"

def _long_rho_xx(df, cmap, long_channel, long_df, long_cmap, cfg):
    """Return (T -> (rho_xx, field_oe) | (None, None) callable, decline reason), or
    (None, reason) if no callable could be built at all.

    rho_xx is the ZERO-FIELD longitudinal resistivity, evaluated PER TEMPERATURE (spec
    §4.3): mu = |R_H|/rho_xx is a zero-field statement, and on a magnetoresistive channel
    the field-averaged value is a different quantity. Measured on the real Hall file at
    2 K: the old loop-averaged rho_xx was 1.493x the zero-field value -- the SAME ratio
    read as "49% above" from the zero-field side and "33% low" for mu from the
    loop-averaged side (not two independent measurements); 21% low for mu at 10 K; and
    the reported rho_xx was non-monotonic in T (2 K above 5 K).

    A setpoint with no zero-field row of its own must decline rather than receive a
    different setpoint's value. Review round 1 Important #1 measured the old
    file-wide-only decline handing a 10 K point 50 K's zero-field rho_xx outright
    (np.interp's CLAMP past the zero-field grid's actual range), 3x wrong, with
    rho_xx_field_oe=0.0 stamped as if a real zero-field row existed at 10 K -- a
    fabricated number wearing a provenance stamp, exactly what this project's rules exist
    to prevent. The returned callable therefore refuses (returns (None, None)) whenever
    the NEAREST zero-field temperature node sits farther than HallCfg.temp_interval
    (config.py, default 1.0 K) from the query -- ruling, not my first pass: that first cut
    used StabilityCfg.drift_max["temperature"] (0.25 K), which is the WRONG quantity here.
    drift_max describes how much the INSTRUMENT is allowed to drift while HOLDING one
    setpoint; it says nothing about how close a longitudinal row must sit to count as
    measuring the SAME temperature as a Hall-channel setpoint, and at 0.25 K it would
    decline legitimately-matched rows (a 2 K setpoint whose longitudinal rows sit at
    2.3 K is a normal file, not a defect). temp_interval is this probe's own declared
    temperature resolution -- the same spacing `_interp_fixed_field_curves` already grids
    fixed-field curves at, and user-settable via --temp-interval, so a coarser real
    sequence widens it without a new flag. The nearest-node check subsumes any separate
    "within the grid span" test: a query outside the grid has an endpoint as its nearest
    node, and if that endpoint is within tolerance the clamp is returning a genuinely
    nearby measurement, which is the honest case (a query strictly BETWEEN two zero-field
    points that both sit farther than temp_interval away -- e.g. two held setpoints with
    a wide gap between them -- declines too, not just off-grid queries).

    The |H| < ZERO_FIELD_OE mask is the same convention resistivity's RRR endpoints use,
    with one difference worth stating exactly: resistivity masks a SEGMENT's setpoint
    field, one value per ramp (`resistivity.py:413`, `s.setpoint.get("field")`), while
    this masks raw per-row field readings directly. Coupled in spirit, not byte-for-byte.

    `reason` is None on success (a callable was returned), else one of
    _RHO_XX_CHANNEL_MISSING (the resistivity/temperature/field column isn't present at
    all -- e.g. --long-channel points at a bridge with no data) or _RHO_XX_NO_ZERO_FIELD
    (the columns exist but not one row anywhere satisfies the mask). Important #3: keep
    these apart, so a wrong --long-channel doesn't get diagnosed as a physics finding. A
    per-setpoint decline (file-wide success, but THIS temperature isn't covered) is
    signalled by the callable's own (None, None) return, not by this reason.
    """
    if long_channel is None:
        return None, None
    src_df, src_cmap = (long_df, long_cmap) if long_df is not None else (df, cmap)
    rk = f"resistivity_ch{long_channel}"
    if (rk not in src_cmap.logical or "temperature" not in src_cmap.logical
            or "field" not in src_cmap.logical):
        return None, _RHO_XX_CHANNEL_MISSING
    T = pd.to_numeric(src_df[src_cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    H = pd.to_numeric(src_df[src_cmap.logical["field"]], errors="coerce").to_numpy(float)
    rho = pd.to_numeric(src_df[src_cmap.logical[rk]], errors="coerce").to_numpy(float)
    m = (np.isfinite(T) & np.isfinite(rho) & (rho > 0)
         & np.isfinite(H) & (np.abs(H) < _ZERO_FIELD_OE))
    if m.sum() < 1:
        return None, _RHO_XX_NO_ZERO_FIELD   # DECLINE: no zero-field rho_xx exists in this file
    Tg, Rg, Hg = T[m], rho[m], H[m]
    order = np.argsort(Tg)
    Tg, Rg, Hg = Tg[order], Rg[order], Hg[order]
    uT = np.unique(Tg)
    uR = np.array([Rg[Tg == t].mean() for t in uT])
    uH = np.array([float(np.median(np.abs(Hg[Tg == t]))) for t in uT])
    # HallCfg.temp_interval, NOT StabilityCfg.drift_max["temperature"] -- see the
    # docstring above for why (drift_max is instrument-hold noise, not "same setpoint").
    tol = cfg.hall.temp_interval
    def rho_and_field_at(temp):
        nearest = uT[int(np.argmin(np.abs(uT - temp)))]
        if abs(nearest - temp) > tol:
            return None, None     # DECLINE: no zero-field row within temp_interval of this setpoint
        return float(np.interp(temp, uT, uR)), float(np.interp(temp, uT, uH))
    return rho_and_field_at, None


# Review round 1, Important #2: the mobility capability's decline reason must name the
# ACTUAL cause. `longitudinal_source` is stamped from the requested channel/file before
# `_long_rho_xx` ever runs, so "no longitudinal channel/file supplied" was self-
# contradicting whenever a source WAS supplied but produced nothing. Shared with
# hall_tempdep.py's own _capabilities (imported there) so the two probes never drift.
#
# Review round 2, Important #1 (a regression round 1's per-point decline introduced, not
# a pre-existing gap): `rho_reason` is a FILE-LEVEL signal -- None means _long_rho_xx
# found zero-field rows SOMEWHERE -- and says nothing about whether any of them fell
# within temp_interval of an actual Hall setpoint.
#
# Review round 3: round 2's own fix over-generalised -- it reported the misalignment
# cause whenever ANY declining point carried the flag, an existential claim asserted as a
# universal one, false whenever some points declined for an unrelated reason (their rho_xx
# was fine; their R_H fit failed). It was also unreachable from hall_tempdep.py's call
# site, which passes only two arguments (HallTDepPoint has no derived_flags until Task 5),
# so that probe fell straight through to the very "carries no |H| < ... row" text round 2
# set out to remove -- just from the other probe. Restructured as a ladder, strongest
# evidence first: a file-level fact is reported outright (the two checks right below); a
# per-point cause is reported only when EVERY declining point carries it (universal claim,
# universal evidence); a mix is described as a mix rather than generalised from a subset;
# and with no per-point evidence at all -- today, hall_tempdep.py -- no per-point cause is
# asserted. Weaker and true beats specific and false. hall_tempdep.py stays on the last
# rung until Task 5 gives HallTDepPoint the same per-point flags HallTempPoint already
# has; once it can pass `points`, it climbs the ladder like hall.py already does, with no
# new logic.
def _mobility_gap_reason(long_source, rho_reason, points=None):
    if not long_source:
        return "no longitudinal channel/file supplied for rho_xx"
    if rho_reason == _RHO_XX_CHANNEL_MISSING:
        return f"{long_source}: longitudinal resistivity/temperature/field column not found"
    if rho_reason == _RHO_XX_NO_ZERO_FIELD:
        return f"{long_source} carries no |H| < {_ZERO_FIELD_OE:.0f} Oe row for rho_xx"
    # rho_reason is None: the file-level check succeeded -- rho_fn resolves somewhere in
    # the file -- yet mobility is unavailable everywhere. Climb on per-point evidence, and
    # only as far as that evidence actually reaches.
    if points is not None:
        declining = [p for p in points if p.mobility is None]
        flagged = [p for p in declining
                   if _RHO_XX_NO_ZERO_FIELD in p.derived_flags
                   or _RHO_XX_CHANNEL_MISSING in p.derived_flags]
        if declining and len(flagged) == len(declining):
            # every mobility-less point's own rho_xx missed temp_interval: a universal
            # claim, backed by universal evidence.
            return (f"{long_source} has zero-field rows, but none fall within "
                    f"temp_interval of any Hall setpoint's temperature")
        if flagged:
            # only SOME do -- naming that one cause for all of them would assert something
            # untrue of the rest, which failed for a different, unestablished reason.
            return (f"{long_source}: some setpoints have no temp_interval-aligned "
                    f"zero-field rho_xx row, the rest failed for a different reason")
    # No points supplied (hall_tempdep.py, pending Task 5) or none of the declining points
    # carry an rho_xx-specific flag: there is no per-point evidence to name a cause from.
    return (f"{long_source}: mobility did not resolve at any setpoint "
            f"(cause not established per point)")


def _capabilities(points, has_thickness, long_source, rho_reason=None) -> list[Capability]:
    any_anti = any(p.antisymmetrized for p in points)
    any_RH = any(p.R_H is not None for p in points)
    # 2026-09-10 (fix round 1): any_RH used to be an accurate proxy for "carrier_n is
    # published somewhere" -- decline_unresolved() broke that equivalence on purpose (R_H
    # stays; carrier_n does not), which left carrier_concentration's `applicable` stale: a
    # fully noise-dominated file could report applicable=True while publishing carrier_n
    # on ZERO points. Key it on the live field instead, same as `mobility` already does.
    any_n = any(p.carrier_n is not None for p in points)
    any_mu = any(p.mobility is not None for p in points)
    return [
        Capability(name="hall_coefficient", applicable=any_RH,
                   reason="Stage A/B R_xy(B) line fits with thickness" if any_RH
                   else ("thickness required for R_H (slope-only)" if not has_thickness
                         else "no field loop could be fit")),
        Capability(name="antisymmetrization", applicable=any_anti,
                   reason="field loops contain both +H and -H" if any_anti
                   else "no loop spans both field signs; Stage B skipped"),
        Capability(name="carrier_concentration", applicable=any_n,
                   reason="n = 1/(e|R_H|) from Stage B" if any_n
                   else ("needs R_H" if not any_RH
                         else "R_H resolved, but every point's sigma >= |R_H| "
                              "(r_h_unresolved) -- see point.withheld")),
        Capability(name="mobility", applicable=any_mu,
                   reason=f"mu = |R_H|/rho_xx ({long_source})" if any_mu
                   else _mobility_gap_reason(long_source, rho_reason, points)),
        # Recognized-but-deferred (2026-09-05): decomposing rho_xy = R0*B + R_s*mu0*M
        # requires M(H) of the SAME sample, which no file in this corpus provides. See
        # docs/physics-reference.md, "Anomalous Hall effect".
        Capability(name="anomalous_hall", applicable=False,
                   reason="requires M(H) of the same sample measured in a magnetometer "
                          "(VSM/MPMS); without it R0 and the anomalous term are not "
                          "separable and no partial extraction is defensible"),
    ]


def field_sweep_points(df, cmap, cfg, hc, thickness_m, rho_fn, rho_reason) -> list[HallTempPoint]:
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
    # 2026-09-07 (spec §4.2): per-row instrument sigma of the Hall channel's resistance,
    # same estimator hall_tempdep uses. None (never a shaky number) when the std column,
    # resistance or resistivity column is absent, or the R/rho ratio isn't constant.
    sigma_row = row_sigma_R(df, cmap, hc.hall_channel)
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
        Hp, R_asym, S_asym = _antisymmetrize(Hh, Rr, sigma_row[idx] if sigma_row is not None else None)
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
            # 2026-09-07 (spec §4.2): instrument sigma, same OLS-with-intercept estimator
            # as the residual sigma above, but driven by the file's own repeat-noise
            # columns rather than fit scatter. None (never a shaky number) when the
            # per-row sigma is unavailable (see row_sigma_R).
            if S_asym is not None:
                ssig_i = slope_sigma_ols(Hp / _OE_PER_T, S_asym)
                pt.slope_sigma_instrument_ohm_per_T = ssig_i
                if ssig_i is not None and thickness_m:
                    pt.r_h_sigma_instrument = float(ssig_i * thickness_m)
            # 2026-09-07 (spec §4.6): field-window ladder, RESOLVED-sigma judged rung by
            # rung the same way the point itself is (ruling above _r_h_ladder). RULING
            # (audit item 5): skip the ladder entirely without thickness_m -- every rung's
            # R_H would then be None (thickness gates R_H, not the slope), every rung would
            # be dropped, and every point would wrongly carry ladder_incomplete for a
            # missing USER INPUT rather than for data that could not support rungs. The
            # thickness gate downstream already names that cause with its own remedy; this
            # flag must mean only "the data itself was too thin".
            if thickness_m is not None:
                rungs, spread, lflags = _r_h_ladder(Hp, R_asym, S_asym, thickness_m,
                                                    hc.geometry_sign)
                pt.r_h_ladder = rungs or None    # [] -> None (fix round 1, Minor)
                pt.r_h_spread = spread
                if lflags:
                    pt.derived_flags = [*pt.derived_flags, *lflags]
        # #20 (2026-09-02): Stage C derives ONLY from the trusted Stage B R_H. The old
        # fallback to R_H_raw published a carrier density and mobility beside an empty
        # R_H cell (Stage A still carries the even-in-B admixture that antisymmetrization
        # exists to remove — on real data ~100x the Hall signal). Decline discipline
        # (cf. the resistivity power-law decline): withhold the derived quantities and
        # carry a machine-readable reason; R_H_raw stays visible for transparency.
        if pt.R_H is None and pt.R_H_raw is not None:
            pt.derived_flags = [*pt.derived_flags, "antisym_r_h_missing"]
        pt.carrier_n, pt.carrier_type = _carrier_n(pt.R_H)
        # 2026-08-10 spec §2.1: sigma propagated by relative sigma (n = 1/(e|R_H|) and
        # mu = |R_H|/rho_xx are pure reciprocal/scale). Stage B only, like the values.
        trusted = anti if anti is not None else raw
        pt.sigma_zero_dof = bool(trusted.get("sigma_zero_dof", False))
        if pt.R_H and pt.r_h_sigma is not None:
            rel = pt.r_h_sigma / abs(pt.R_H)
            if pt.carrier_n is not None:
                pt.carrier_n_sigma = float(pt.carrier_n * rel)
        if pt.R_H and pt.r_h_sigma_instrument is not None:
            rel_i = pt.r_h_sigma_instrument / abs(pt.R_H)
            if pt.carrier_n is not None:
                pt.carrier_n_sigma_instrument = float(pt.carrier_n * rel_i)
        if rho_fn is not None:
            # rho_fn(Tset) -> (rho_xx, field_oe) for a covered setpoint, else (None, None)
            # for THIS point only (review round 1 Important #1) -- a covered file can
            # still leave an individual setpoint outside the zero-field grid's range.
            rho_val, field_val = rho_fn(Tset)     # longitudinal measurement, independent of R_H
            if rho_val is not None:
                pt.rho_xx = rho_val
                pt.rho_xx_field_oe = field_val
                pt.mobility = _mobility(pt.R_H, pt.rho_xx)
                if (pt.mobility is not None and pt.R_H and pt.r_h_sigma is not None):
                    pt.mobility_sigma = float(pt.mobility * pt.r_h_sigma / abs(pt.R_H))
                if (pt.mobility is not None and pt.R_H
                        and pt.r_h_sigma_instrument is not None):
                    pt.mobility_sigma_instrument = float(
                        pt.mobility * pt.r_h_sigma_instrument / abs(pt.R_H))
            elif hc.longitudinal_channel is not None:
                # This setpoint isn't covered by any zero-field row: falling back to a
                # different setpoint's value would reintroduce the defect silently, so
                # decline and say why (spec §4.3).
                pt.derived_flags = [*pt.derived_flags, _RHO_XX_NO_ZERO_FIELD]
        elif hc.longitudinal_channel is not None:
            # File-wide decline before any per-point call was even possible. rho_reason
            # (from _long_rho_xx) says whether the channel/columns are simply missing vs.
            # present with no zero-field row anywhere at all -- review round 1 Important
            # #3: never collapse the two into one message.
            pt.derived_flags = [*pt.derived_flags, rho_reason or _RHO_XX_NO_ZERO_FIELD]
        # Spec Sec 4.1: sigma >= |R_H| means the +-1 sigma interval contains zero, so n is
        # unbounded above and the carrier sign is undetermined. Withhold the derived
        # quantities, keep R_H and its sigma visible, and say why. Applied last, once every
        # derived quantity above has been computed, so the withheld copy is complete.
        decline_unresolved(pt)
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
    one `carrier_n_sigma`/`mobility_sigma` were already computed from, and names the stage.

    2026-09-07 (spec §4.2): the field-sweep analyzer now also carries an instrument
    (repeat-noise) sigma alongside the residual (fit-scatter) one, only on the antisym
    (Stage B) stage — Stage A has no instrument family here. Where present, the instrument
    sigma is the one tested (it is a weaker, different claim than fit scatter and the one
    hall_tempdep already prefers for this same warning); the message always names which
    family it tested so the two are never confused."""
    out = []
    for p in points:
        inst = p.r_h_sigma_instrument
        trusted_sig = inst if inst is not None else (
            p.r_h_sigma if p.antisymmetrized else p.r_h_sigma_raw)
        R_H_trusted = p.R_H if p.R_H is not None else p.R_H_raw
        stage = "Stage B antisym" if p.antisymmetrized else "Stage A raw"
        if R_H_trusted and trusted_sig is not None and R_H_trusted != 0:
            rel = trusted_sig / abs(R_H_trusted)
            r2 = p.r2 if p.antisymmetrized else p.r2_raw
            if rel > _REL_SIGMA_WARN:
                # F16 (final-review) + 2026-09-07 (spec §4.2): name the sigma FAMILY. In a
                # slice whose thesis is that residual and instrument sigma must never share
                # a name, a bare "relative sigma" was the loose one; the hall_tdep sibling
                # already says "relative instrument sigma". The message must always name
                # which one it tested — "residual sigma" (fit scatter, r² shown) or
                # "instrument sigma" (repeat noise; r² speaks to fit quality, a different
                # claim, so it is not quoted alongside a noise-family verdict).
                if inst is not None:
                    label, explain = "instrument sigma", "instrument noise, not fit quality"
                    detail = f"({stage}, {explain})"
                else:
                    r2txt = "n/a" if r2 is None else f"{r2:.3f}"
                    label = "residual sigma"
                    detail = f"({stage} fit scatter, r² = {r2txt})"
                out.append(f"R_H at T = {p.temperature:.1f} K carries {rel * 100:.0f}% "
                           f"relative {label} {detail} — "
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
            # A missing hall channel is a missing USER INPUT, not a broken file: gate with
            # a remedy (like molar_mass on VSM), never a hard error with an empty gate[].
            return Result(status="gated", confidence=0.5, data={"probe": "hall"},
                          gate=[Gate(need="hall_channel",
                                     reason="which bridge carries the transverse (Hall) signal is not stated in the file",
                                     remedy={"flag": "--hall-channel", "example": "--hall-channel 1"})],
                          provenance=prov)
        rkey = f"resistance_ch{hc.hall_channel}"
        if rkey not in cmap.logical or "temperature" not in cmap.logical or "field" not in cmap.logical:
            return Result(status="error",
                          errors=[f"hall channel {hc.hall_channel} resistance column / T / H not found"],
                          data={"probe": "hall"}, provenance=prov)
        # Task 4b: a user-controlled leading-row skip, not a detector. Some PPMS runs
        # write a first data row taken before the bridge has settled -- not a noisy
        # reading, not a reading at all (measured: R off by 10-11 orders of magnitude
        # from the file median). skip_rows drops the first N rows of the WHOLE file
        # (every column, not just this channel) before anything else runs; the operator
        # decides N, never a threshold. The count and, when the dropped row looked
        # physical, a reversal warning both go out no matter what the rest of the file
        # yields, so both are computed before df is sliced.
        n_skip = max(0, int(hc.skip_rows))
        skip_warn = skip_row_warning(df, cmap, hc.hall_channel, n_skip) if n_skip else None
        if n_skip:
            df = df.iloc[n_skip:].reset_index(drop=True)
        T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
        H = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
        Rxy = pd.to_numeric(df[cmap.logical[rkey]], errors="coerce").to_numpy(float)
        if np.isfinite(Rxy).sum() == 0:
            return Result(status="error", errors=[f"hall channel {hc.hall_channel} is empty"],
                          warnings=[skip_warn] if skip_warn else [],
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
        rho_fn, rho_reason = _long_rho_xx(df, cmap, hc.longitudinal_channel, long_df, long_cmap, cfg)

        points = field_sweep_points(df, cmap, cfg, hc, thickness_m, rho_fn, rho_reason)

        if not points:
            return Result(status="low_confidence", confidence=0.2,
                          warnings=([skip_warn] if skip_warn else []) + ["no field loops found to fit"],
                          data={"probe": "hall", "reason": "no field loops"}, provenance=prov)
        caps = _capabilities(points, thickness_m is not None, long_source, rho_reason)
        hd = HallData(probe="hall", hall_channel=hc.hall_channel, thickness_m=thickness_m,
                      geometry_sign=hc.geometry_sign, longitudinal_source=long_source,
                      points=points, capabilities=caps, skipped_rows=n_skip)
        r2s = [p.r2 for p in points if p.r2 is not None]
        if thickness_m is None:
            # A missing thickness is a missing USER INPUT, not a broken file (same rule as
            # hall_channel above and molar_mass on VSM): gate with a remedy, and KEEP the
            # slope-only points in data so the work is not discarded. `_stage_fit` computes
            # r2 without needing a thickness (only R_H does), so r2s is genuinely populated
            # here -- report the real mean, not a hardcoded None (Task 2 review, carried).
            return Result(status="gated", confidence=0.4,
                          confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                            "fit": (float(np.mean(r2s)) if r2s else None)},
                          gate=[Gate(need="thickness_mm",
                                     reason="R_H = slope x thickness; without a thickness "
                                            "only the slope is measured",
                                     remedy={"flag": "--thickness",
                                             "example": "--thickness 0.07 --thickness-unit mm"})],
                          warnings=[skip_warn] if skip_warn else [],
                          data=hd.model_dump(mode="json"), provenance=prov)
        # Spec §4.5: two ceilings, and confidence is the lower. `fit` says how well the
        # lines fit; `resolved` says how many R_H are distinguishable from zero. A result
        # cannot be more trustworthy than either. min, not a product: multiplying two
        # ceilings understates a result that is merely noisy OR merely scattered.
        fit_quality = float(np.mean(r2s)) if r2s else 1.0
        resolved_fraction = (sum(1 for p in points if is_resolved(p)) / len(points)
                             if points else 0.0)
        conf = float(min(fit_quality, resolved_fraction))
        status = "ok" if conf >= cfg.confidence_min else "low_confidence"
        return Result(status=status, confidence=conf,
                      confidence_parts={"detector": 1.0, "segmentation": 1.0,
                                        "fit": (float(np.mean(r2s)) if r2s else None),
                                        "resolved": float(resolved_fraction)},
                      warnings=([skip_warn] if skip_warn else []) + sigma_noise_warnings(points),
                      data=hd.model_dump(mode="json"), provenance=prov)
