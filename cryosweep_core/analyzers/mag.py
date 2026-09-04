from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.sweeps import segment_sweeps
from cryosweep_core.detect.vsm_blocks import classify_vsm_blocks, ramps_from_temps
from cryosweep_core.fitting.models import CurieWeissModel, fit_cw_ladder
from cryosweep_core.grouping import cluster_field_setpoints, setpoint_key
from cryosweep_core.result import Result, Gate, Provenance, FitResult
from cryosweep_core.registry import Need
from pydantic import BaseModel, ConfigDict

_N_A = 6.022e23
_MU_B_SI = 9.274e-24


class MHLoop(BaseModel):
    model_config = ConfigDict(extra="ignore")
    temperature: float                 # rounded loop setpoint (K)
    field_oe: list[float] = []         # raw field, row order preserved
    moment: list[float] = []           # same convention as moment_per_fu (mu_B/f.u.)
    n_points: int = 0


class Ramp(BaseModel):
    model_config = ConfigDict(extra="ignore")
    direction: str                     # "warming" | "cooling"
    i0: int                            # inclusive index into the POST-FILTER arrays
    i1: int                            # inclusive index into the POST-FILTER arrays


class TBlock(BaseModel):
    """One monotone temperature ramp of one M(T) sweep block (ZFC/FC = two TBlocks at one
    field). Arrays use the SAME per-point math + physical-point mask as the flat exported
    arrays; lets the M(T)-family renderers split warming/cooling on real files where the flat
    arrays only carry the single widest monotone segment."""
    model_config = ConfigDict(extra="ignore")
    direction: str                     # "warming" | "cooling"
    field_oe: float                    # block's held-field setpoint (rounded like loops' T)
    temperature: list[float] = []
    moment: list[float] = []           # moment_per_fu convention (mu_B/f.u.)
    chi: list[float] = []              # unit-aware molar susceptibility (SI if cfg SI else CGS)
    inv_chi: list[float] = []          # unit-aware inverse susceptibility


class VSMData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    probe: str = "vsm"
    temperature: list[float] = []
    field: list[float] = []
    moment_emu_per_g: list[float] = []
    moment_per_fu: list[float] = []
    chi_molar_cgs: list[float] = []
    chi_molar_si: list[float] = []
    inv_chi: list[float] = []
    inv_chi_unit: str | None = None
    fit: FitResult | None = None
    # --- PQ-3 Task 1 additive fields (declared AFTER `fit`: pydantic v2 serializes in
    # declaration order, so new keys append and the byte-identity oracle holds) ---
    loops: list[MHLoop] = []           # one per contiguous field-sweep branch
    ramps: list[Ramp] = []             # M(T) direction tags over the exported arrays
    # --- PQ-3 Task 3 additive field (declared AFTER `ramps`: append-only key order) ---
    fit_modified: FitResult | None = None  # modified CW (chi = chi0 + C/(T-theta)); None on failure
    # --- PQ-3 t_blocks additive field (declared LAST: append-only JSON key order). One entry
    # per (temperature-sweep block x monotone ramp); powers the M(T)-family warming/cooling
    # split on real ZFC/FC + multi-field files. CW fit / flat arrays / loops / ramps untouched. ---
    t_blocks: list[TBlock] = []
    # --- 2026-08-10 uncertainty-honesty additive fields (declared AFTER t_blocks: append-only
    # JSON key order). CW fit-window ladder (spec §1.2): per-rung refits, spreads = max-min over
    # fitted rungs + the full fit; None (never 0.0) when < 2 rungs fitted (U2). ---
    cw_ladder: list[dict] | None = None
    theta_spread_k: float | None = None
    mu_eff_spread: float | None = None

def _cw_confidence(fit) -> float:
    """Closed O1 (spec §1.3): r2 clamped to [0,1] (fixes falsy-0.0 and negative-r2 defects),
    x0.5 when the CW fit is window-sensitive — a regime-contaminated fit is no longer
    certified ok. theta itself is untouched."""
    r2c = 0.0 if fit.r2 is None else min(1.0, max(0.0, fit.r2))
    penalty = 0.5 if "window_sensitive" in fit.quality_flags else 1.0
    return r2c * penalty


def _sha256(path) -> str:
    import pathlib
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""


def _compute_t_blocks(blocks, temp, field, moment_per_fu, chi_unit, inv_chi):
    """Build one TBlock per (temperature-sweep block x monotone ramp).

    Reuses the already-computed flat per-point arrays (no formula duplication) and applies the
    SAME physical-point mask as the CW flat path (finite T/inv_chi/field AND |field|>1 Oe) per
    block, then splits the masked block into monotone ramps so a single ZFC/FC classifier block
    (field held, T reverses) yields separate warming + cooling TBlocks at one field setpoint.
    A field-plateau constraint (|field - block median| within the classifier's plateau
    tolerance) keeps each TBlock to ONE field setpoint even when the classifier's block boundary
    leaks a few rows of the adjacent field group in (real MPMS: the 40000 Oe ramp's first rows
    can trail the 500 Oe block) — without it those high-field rows spike the M(T) curve.
    """
    # TWO PASSES. The held-field label must be decided ACROSS blocks, not per block: two
    # blocks of one physical 40 kOe ramp had masked medians 40000.8870 and 39999.5860 and
    # setpoint_key's magnitude-blind integer rounding split them into 40001 and 40000, i.e.
    # two M(T) curves for one field. Pass 1 collects each block's masked arrays and median;
    # pass 2 labels them with a cluster shared across the whole file.
    kept: list[tuple] = []
    for blk in blocks:
        if blk.kind != "temperature":
            continue
        sl = slice(blk.start, blk.end)
        t_b, f_b = temp[sl], field[sl]
        m_b, c_b, iv_b = moment_per_fu[sl], chi_unit[sl], inv_chi[sl]
        base = (np.isfinite(t_b) & np.isfinite(iv_b) & np.isfinite(f_b) & (np.abs(f_b) > 1.0))
        if not base.any():
            continue
        med = float(np.median(f_b[base]))                   # block's held-field level
        tol = max(50.0, 0.1 * abs(med))                     # classifier field-plateau tolerance
        mask = base & (np.abs(f_b - med) <= tol)
        if not mask.any():
            continue
        t_m, f_m = t_b[mask], f_b[mask]
        kept.append((t_m, f_m, m_b[mask], c_b[mask], iv_b[mask], float(np.median(f_m))))

    labels = cluster_field_setpoints([k[5] for k in kept])
    out: list[TBlock] = []
    for (t_m, _f_m, m_m, c_m, iv_m), setp in zip([k[:5] for k in kept], labels):
        if not np.isfinite(setp):
            continue
        # min_len=5 (module default 3) merges noise turnarounds: a 3-point "warming" run
        # inside a 116-point cooling ramp at 294.8-300.0 K is instrument noise, not a
        # measurement. The module default is deliberately untouched -- other callers rely
        # on it. Measured radius: vsm_mt 7->5 blocks, vsm 23->11, mpms unchanged.
        for r in ramps_from_temps(t_m.tolist(), min_len=5):
            a, b = r["i0"], r["i1"] + 1                       # ramps indices are INCLUSIVE
            out.append(TBlock(direction=r["direction"], field_oe=float(setp),
                              temperature=t_m[a:b].tolist(), moment=m_m[a:b].tolist(),
                              chi=c_m[a:b].tolist(), inv_chi=iv_m[a:b].tolist()))
    return out

def _moment_notes(moment_source):
    """One warning when the moment came from the DC column instead of `Moment (emu)`.

    Emitted on every return path that carries data, so the provenance of the numbers is
    never lost to an early return."""
    if moment_source != "m_dc":
        return []
    return ["moment read from the DC column 'M-DC (emu)': this file's 'Moment (emu)' "
            "column is present but empty in every row (DC-mode ACMS measurement)"]


class VSMAnalyzer:
    probe = "vsm"
    needs = (Need("molar_mass", scope="header", required=False),
             Need("sample_mass", scope="header", required=False))

    def analyze(self, rawtable, cfg) -> Result:
        df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
        header = rawtable.header
        mol = header.molar_mass
        mass_g = (header.mass_mg / 1000.0) if header.mass_mg else None
        prov = Provenance(file=getattr(rawtable, "path", None) or header.title or "",
                          sha256=_sha256(getattr(rawtable, "path", None)),
                          app_version=header.app_version, config=cfg.model_dump(mode="json"))
        for _k in ("field", "temperature"):
            if _k not in cmap.logical:
                return Result(status="error",
                              errors=[f"vsm needs a '{_k}' column (not found in this file)"],
                              data={"probe": "vsm"}, provenance=prov)
        # Moment source. `moment` normally, `m_dc` for DC-mode ACMS files. The test is
        # "absent OR UNUSABLE", not "absent": those files carry `Moment (emu)` in the
        # header and leave it empty in every row, so an absence-only check never fires
        # and the analyzer reports "no temperature sweep found" on a perfectly good ramp.
        # Pick the column with the MOST finite rows, not the first with >=1. A single stray
        # value decides nothing: on a DC-mode file one finite row in the otherwise-empty
        # `Moment (emu)` column flipped the source away from `m_dc` and destroyed the whole
        # result (ok + CW fit -> "insufficient physical points"). Ties keep declaration
        # order, so a normally-populated `moment` still wins over an equally-populated m_dc.
        moment, moment_source, _best = None, None, 0
        for _cand in ("moment", "m_dc"):
            if _cand not in cmap.logical:
                continue
            _arr = pd.to_numeric(df[cmap.logical[_cand]], errors="coerce").to_numpy(float)
            _n = int(np.isfinite(_arr).sum())
            if _n > _best:
                moment, moment_source, _best = _arr, _cand, _n
        if moment is None:
            _tried = [c for c in ("moment", "m_dc") if c in cmap.logical]
            return Result(
                status="error",
                errors=["vsm needs a 'moment' column (not found in this file)" if not _tried
                        else f"vsm found moment column(s) {_tried} but every row is empty"],
                data={"probe": "vsm"}, provenance=prov)
        field = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
        temp = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
        gates = []
        if mol is None:
            gates.append(Gate(need="molar_mass", reason="no MOLWGHT in header",
                              remedy={"flag": "--molar-mass", "example": "--molar-mass 200.0"}))
        if mass_g is None:
            gates.append(Gate(need="sample_mass", reason="no MASS in header",
                              remedy={"flag": "--mass-mg", "example": "--mass-mg 5.0"}))
        if gates:
            return Result(status="gated", confidence=0.5, data={"probe": "vsm"}, gate=gates, provenance=prov)
        with np.errstate(divide="ignore", invalid="ignore"):
            moment_per_g = moment / mass_g
            moment_per_fu = (moment * 1e-3) * mol / (_MU_B_SI * _N_A * mass_g)
            chi_molar = moment_per_g / field * mol              # emu/(mol*Oe)  (CGS)
            chi_si = chi_molar * (4 * np.pi * 1e-6)             # m^3/mol       (SI)
            inv_chi_cgs = 1.0 / chi_molar
            inv_chi_si = 1.0 / chi_si
        # Bug 1: fit the inverse susceptibility in the REQUESTED unit system so that
        # the Curie constant C and mu_eff = k*sqrt(C) are self-consistent.
        is_si = cfg.unit_system == "SI"
        inv_chi = inv_chi_si if is_si else inv_chi_cgs
        # Dedicated per-block VSM classifier -> M(H) loops (additive). One loop per
        # contiguous field-sweep branch; same-rounded-T blocks stay separate loops;
        # row order preserved; non-finite rows dropped (JSON never carries NaN/inf).
        # Moment uses moment_per_fu (same convention as the existing moment arrays).
        blocks = classify_vsm_blocks(df, cmap, cfg)
        loops = []
        for blk in blocks:
            if blk.kind != "field":
                continue
            sl = slice(blk.start, blk.end)
            f_b = field[sl]; m_b = moment_per_fu[sl]; t_b = temp[sl]
            good = np.isfinite(f_b) & np.isfinite(m_b)
            if not good.any():
                continue
            t_good = t_b[np.isfinite(t_b)]
            setp = setpoint_key(float(np.median(t_good))) if t_good.size else float("nan")
            if not np.isfinite(setp):
                continue
            loops.append(MHLoop(temperature=float(setp),
                                field_oe=f_b[good].tolist(),
                                moment=m_b[good].tolist(),
                                n_points=int(good.sum())))
        # Per-temperature-block M(T) arrays (additive): one TBlock per (block x monotone ramp),
        # unit-aware chi so the M(T)-family renderers can split warming/cooling on real files.
        chi_unit = chi_si if is_si else chi_molar
        t_blocks = _compute_t_blocks(blocks, temp, field, moment_per_fu, chi_unit, inv_chi)
        segs = [s for s in segment_sweeps(df, cmap, cfg) if s.swept.name == "temperature"]
        if not segs:
            # No M(T) sweep. If M(H) loops exist the file is legitimately M(H)-only ->
            # analyze ok (no CW fit). "no temperature sweep found" low_confidence is now
            # reserved for files with NEITHER sweep type.
            if loops:
                vd = VSMData(probe="vsm", inv_chi_unit="mol/m^3" if is_si else "mol*Oe/emu",
                             fit=None, loops=loops, ramps=[], t_blocks=t_blocks)
                _d = vd.model_dump(mode="json")
                if moment_source == "m_dc":
                    _d["moment_source"] = moment_source
                return Result(status="ok", confidence=0.6,
                              confidence_parts={"detector": 1.0, "segmentation": 1.0, "fit": None},
                              warnings=[*_moment_notes(moment_source),
                                        "M(H)-only file: field-sweep loops present, no M(T) "
                                        "temperature sweep -> Curie-Weiss fit skipped"],
                              data=_d, provenance=prov)
            return Result(status="low_confidence", confidence=0.2,
                          warnings=[*_moment_notes(moment_source), "no temperature sweep found"],
                          data={"probe": "vsm", "reason": "no temperature sweep found",
                                **({"moment_source": moment_source} if moment_source == "m_dc" else {})},
                          provenance=prov)
        # Which ramp is THE Curie-Weiss fit. Neither point count nor field alone survives
        # contact with real data, so the choice is two-stage:
        #   1. Keep only ramps covering >=50% of the WIDEST candidate ramp's temperature
        #      span. A CW fit over a truncated window is not a CW fit -- on a real file the
        #      100 Oe ramps span just 2-50 K and fit theta = -216 K at r2 = 0.87, against
        #      -30 K at r2 = 0.99 for the full-range ramps.
        #   2. Among those, prefer the LOWEST |field| (CW wants low-field linear response;
        #      10 T is not), size only breaking ties -- on the real MPMS file the 40 kOe
        #      ramp beat the 500 Oe ramp by ONE point and silently moved theta.
        # Then walk the ranked candidates and take the first that actually SURVIVES the
        # physical-point mask. Selecting on nominal field alone loses the fit outright when
        # the lowest-|field| ramp is a nominal-zero / remanent-field ZFC run: every one of
        # its points fails |field| > 1 Oe and a perfectly good higher-field ramp goes unused.
        # Single-field files have one candidate class, so this reduces to the previous
        # behaviour and their results are unchanged.
        def _seg_span(s):
            t = temp[s.idx]
            t = t[np.isfinite(t)]
            return float(np.ptp(t)) if t.size else 0.0

        def _physical(i):
            # Bug 2: drop non-physical inv_chi: NaN/inf AND field~=0 points (chi=inf ->
            # inv_chi=0, which survives an isfinite mask but is meaningless -> flat fit).
            return i[(np.isfinite(temp[i]) & np.isfinite(inv_chi[i])
                      & np.isfinite(field[i]) & (np.abs(field[i]) > 1.0))]

        def _cw_rank(s):
            h = field[s.idx]
            h = h[np.isfinite(h)]
            return (abs(float(np.median(h))) if h.size else float("inf"), -s.idx.size)

        _widest_span = max((_seg_span(s) for s in segs), default=0.0)
        _full = [s for s in segs if _seg_span(s) >= 0.5 * _widest_span] or list(segs)
        keep = np.asarray([], dtype=int)
        _skipped_unusable = 0
        for _cand in sorted(_full, key=_cw_rank):
            _k = _physical(_cand.idx)
            if _k.size >= 3:
                keep = _k
                break
            _skipped_unusable += 1
        if keep.size < 3:
            return Result(status="low_confidence", confidence=0.2,
                          warnings=["too few physical points for Curie-Weiss fit "
                                    "(field ~ 0 or non-finite susceptibility)"],
                          data={"probe": "vsm", "reason": "insufficient physical points"}, provenance=prov)
        try:
            fit, ladder, th_spread, mu_spread = fit_cw_ladder(
                temp[keep], inv_chi[keep], unit_system=cfg.unit_system)
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError) as e:
            return Result(status="error", confidence=0.0, errors=[f"Curie-Weiss fit failed: {e}"],
                          warnings=["Curie-Weiss fit failed"],
                          data={"probe": "vsm", "reason": "fit failure"}, provenance=prov)
        # PQ-3 Task 3: ALSO run the modified Curie-Weiss fit (chi = chi0 + C/(T-theta)) on the
        # SAME kept rows. Additive/non-blocking: failure -> fit_modified=None + a warning, never
        # errors the result. No new fitting code — the model already implements modified=True.
        warnings: list[str] = []
        try:
            fit_modified = CurieWeissModel().fit(temp[keep], inv_chi[keep],
                                                 unit_system=cfg.unit_system, modified=True)
        except (ValueError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as e:
            fit_modified = None
            warnings.append(f"modified Curie-Weiss fit failed: {e}")
        # Paramagnetic-regime warning (spec §1.4): CW is asymptotic only for T >> |theta|.
        tmin = float(np.min(temp[keep]))
        abs_theta = abs(float(fit.params["theta"]))
        if tmin < abs_theta:
            warnings.append(
                f"CW fit window extends below |theta| (T_min = {tmin:.1f} K < {abs_theta:.1f} K)"
                " — low-T rows likely outside the paramagnetic regime; see cw_ladder")
        # Ramp-direction tags index the POST-FILTER exported arrays (temp[keep]).
        ramps = [Ramp(**r) for r in ramps_from_temps(temp[keep].tolist())]
        vd = VSMData(probe="vsm",
                     temperature=temp[keep].tolist(), field=field[keep].tolist(),
                     moment_emu_per_g=moment_per_g[keep].tolist(),
                     moment_per_fu=moment_per_fu[keep].tolist(),
                     chi_molar_cgs=chi_molar[keep].tolist(),
                     chi_molar_si=chi_si[keep].tolist(),
                     inv_chi=inv_chi[keep].tolist(),
                     inv_chi_unit="mol/m^3" if is_si else "mol*Oe/emu",
                     fit=fit, loops=loops, ramps=ramps, fit_modified=fit_modified,
                     t_blocks=t_blocks,
                     cw_ladder=ladder or None, theta_spread_k=th_spread,
                     mu_eff_spread=mu_spread)
        data = vd.model_dump(mode="json")
        # Emitted ONLY on the fallback path, so a normal VSM file's JSON is unchanged
        # (these results are pinned byte-for-byte by the oracle tests).
        if moment_source == "m_dc":
            data["moment_source"] = moment_source
        warnings.extend(_moment_notes(moment_source))
        conf = _cw_confidence(fit)
        status = "ok" if conf >= cfg.confidence_min else "low_confidence"
        return Result(status=status, confidence=conf,
                      confidence_parts={"detector": 1.0, "segmentation": 1.0, "fit": fit.r2},
                      warnings=warnings, data=data, provenance=prov)
