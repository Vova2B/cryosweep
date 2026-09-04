from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from cryosweep_core.plotting.spec import PlotSpec


@dataclass(frozen=True)
class Series:
    key: str
    label: str
    x: list
    y: list
    group: str | None = None
    default_on: bool = True
    role: str | None = None
    yerr: list | None = None
    open_mask: list | None = None
    linestyle: str | None = None    # per-series line style override (VSM ramp split: ↑ solid, ↓ dashed)
    marker: str | None = None       # per-series marker override (channel -> marker, PQ-4)
    label_suffix: str = ""          # cosmetic legend suffix (direction arrows), spec-gated


@dataclass(frozen=True)
class PlotKind:
    key: str
    label: str
    probe: str
    series: Callable            # (result) -> list[Series]; [] => kind unavailable for this data
    default_xscale: Literal["linear", "log"] = "linear"
    default_yscale: Literal["linear", "log"] = "linear"
    group_colored: bool = False


def series_label(result, series: Series) -> str:
    """Legend label. A (single result): the series label. C overrides to prefix a file tag."""
    # Always call this; do not use series.label directly (Mode C will prefix a file tag here).
    return series.label


def select_series(series_list, spec: "PlotSpec | None") -> list[Series]:
    """curves=None -> default_on set; curves=[] -> none; curves=[keys] -> exactly those."""
    if spec is None or spec.curves is None:
        return [s for s in series_list if s.default_on]
    keyset = set(spec.curves)
    return [s for s in series_list if s.key in keyset]


def _arr(d, key):
    return np.asarray(d.get(key) or [], float)


def fmt_field(value_oe, unit="Oe"):
    """Format a field magnitude for a label. Oe -> '500 Oe'; T -> 3-sig-fig Tesla with
    trailing zeros trimmed ('9999'->'1 T', '10000'->'1 T', '500'->'0.05 T', '40000'->'4 T',
    '137000'->'13.7 T'). NaN/None/non-finite -> '' (caller already guards)."""
    if value_oe is None:
        return ""
    try:
        v = float(value_oe)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    if unit == "T":
        # round to integer Oe first (mirrors the Oe path's :.0f), so a near-zero
        # instrument field (e.g. 0.481 Oe on a nominal zero-field ramp) reads "0 T"
        # instead of scientific-notation clutter ("4.81e-05 T").
        return f"{round(v) / 1e4:.3g} T"
    return f"{v:g} Oe"


def _field_scale(unit):
    """Display scale for a field magnitude stored in Oe. 'T' -> 1e-4 (Oe->Tesla); else 1.0."""
    return 1e-4 if unit == "T" else 1.0

# ---- VSM (flat arrays; no per-loop identity -> single series) ----
# PQ-3 Task 4: warming/cooling ramp split for M(T)-family kinds. When the analyzer
# tags >1 monotone ramp over the flat arrays (ZFC/FC use-case), each M(T) data series
# splits into one sub-series per ramp: key "{base}:r{j}", label gains " ↑" (warming) /
# " ↓" (cooling), all ramps of one quantity share a group (=> one colour) and are
# distinguished by linestyle (warming solid, cooling dashed — meaningful only where
# connect_lines applies; markers otherwise identical). Single-ramp files return EXACTLY
# today's single series (no ":r0" suffix, group/linestyle untouched) so byte-identity holds.
_RAMP_ARROW = {"warming": " ↑", "cooling": " ↓"}
_RAMP_LINESTYLE = {"warming": "-", "cooling": "--"}


def _split_from_tblocks(result, series, tblock_y, multi_label_prefix="", field_unit="Oe"):
    """Split a flat M(T) series into warming/cooling sub-series sourced from `t_blocks`
    (per-temperature-block arrays). `tblock_y(block)` extracts this quantity's y per block.

    Two label regimes (v1 per-field-group behaviour):
      * ONE field setpoint (ZFC/FC at one field): keys "{base}:r{j}", labels "{base} ↑/↓",
        one shared colour group per quantity (=base key) — direction encoded by linestyle.
      * MULTIPLE field setpoints: split per (field, direction), keys "{base}:{field}:{dir}",
        labels "{field} Oe ↑/↓", one colour group per field ("{field}Oe") so ↑/↓ at a field
        share the field's colour and differ by linestyle.
    Contiguous blocks sharing a (field, direction) are concatenated in row order into one
    series so a noisy multi-ramp field collapses to a single ↑ and a single ↓ curve.
    """
    tblocks = (result.data or {}).get("t_blocks") or []
    field_keys = []
    for b in tblocks:
        if b.get("field_oe") not in field_keys:
            field_keys.append(b.get("field_oe"))
    multi = len(field_keys) > 1
    groups: dict = {}                       # (field, direction) -> [x_accum, y_accum]
    order: list = []
    for b in tblocks:
        key = (b.get("field_oe"), b.get("direction"))
        if key not in groups:
            groups[key] = [[], []]
            order.append(key)
        elif groups[key][0]:
            # NaN break between concatenated same-(field,direction) blocks so a connected
            # line does not bridge the T gap between them (markers/labels unaffected; robust
            # view, fits, legend all skip non-finite). See render._connect_sort.
            groups[key][0].append(float("nan"))
            groups[key][1].append(float("nan"))
        groups[key][0].extend(list(b.get("temperature") or []))
        groups[key][1].extend(list(tblock_y(b)))
    out = []
    for j, (field, direction) in enumerate(order):
        xs, ys = groups[(field, direction)]
        arrow = _RAMP_ARROW.get(direction, "")
        ls = _RAMP_LINESTYLE.get(direction, "-")
        if multi:
            flabel = fmt_field(field, field_unit) if field_unit == "T" else f"{field:.0f} Oe"
            out.append(Series(
                key=f"{series.key}:{field:g}:{direction}",
                label=f"{multi_label_prefix}{flabel}{arrow}", x=xs, y=ys, group=f"{field:g}Oe",
                default_on=series.default_on, role=series.role, linestyle=ls))
        else:
            out.append(Series(
                key=f"{series.key}:r{j}", label=f"{series.label}{arrow}", x=xs, y=ys,
                group=(series.group if series.group is not None else series.key),
                default_on=series.default_on, role=series.role, linestyle=ls))
    return out


def _split_by_ramps(result, series, tblock_y=None, multi_label_prefix="", field_unit="Oe"):
    """M(T)-family warming/cooling split. Prefers `t_blocks` (per-block arrays -> works on real
    ZFC/FC + multi-field files); when <2 t_blocks (or no extractor) falls back to the flat
    `ramps` tags, and to a single series when there is only one ramp — byte-identical to today.
    `multi_label_prefix` prefixes only the multi-field labels (e.g. "1/χ " for the twin's inverse
    series so its "500 Oe ↑" entries are disambiguated from the χ side's identical labels)."""
    tblocks = (result.data or {}).get("t_blocks") or []
    if tblock_y is not None and len(tblocks) >= 2:
        return _split_from_tblocks(result, series, tblock_y, multi_label_prefix, field_unit=field_unit)
    ramps = (result.data or {}).get("ramps") or []
    if len(ramps) <= 1:
        return [series]
    grp = series.group if series.group is not None else series.key
    out = []
    for j, rmp in enumerate(ramps):
        d = rmp.get("direction")
        i0 = int(rmp.get("i0", 0)); i1 = int(rmp.get("i1", -1))
        out.append(Series(
            key=f"{series.key}:r{j}",
            label=f"{series.label}{_RAMP_ARROW.get(d, '')}",
            x=list(series.x[i0:i1 + 1]), y=list(series.y[i0:i1 + 1]),
            group=grp, default_on=series.default_on, role=series.role,
            linestyle=_RAMP_LINESTYLE.get(d, "-")))
    return out


def _vsm_series(result, ykey, label, tblock_y, field_unit="Oe", flat_y=None):
    """`flat_y` transforms the FLAT array used when a file has fewer than two t_blocks
    (the per-block path is transformed by `tblock_y` instead)."""
    d = result.data or {}
    T = _arr(d, "temperature"); y = _arr(d, ykey)
    if T.size == 0 or y.size == 0 or T.size != y.size:
        return []
    ys = list(y.tolist())
    if flat_y is not None:
        ys = flat_y(ys)
    return _split_by_ramps(
        result, Series(key="curve", label=label, x=T.tolist(), y=ys, default_on=True),
        tblock_y, field_unit=field_unit)

# 1/chi is the reciprocal of a measured quantity, so wherever chi passes through zero the
# inverse diverges and a handful of noise points set the whole axis. Measured on a real
# multi-field file: one 40 kOe branch has chi crossing zero 18 times above 200 K
# (chi_min = -1.08e-05, i.e. ~1e-6 of the sample's typical chi) and 1/chi swings between
# -1.0e6 and +1.4e6, drawing vertical stripes across the panel and flattening every real
# curve. Those points are not a measurement of anything -- they are 1/noise.
#
# The reference scale must be the FILE, not the block: the offending block is *mostly*
# near-zero chi, so its own median is already tiny and a per-block rule does not fire.
# The threshold is deliberately loose -- measured drop counts are IDENTICAL for rel from
# 1e-4 to 1e-2 (73 of 731 points on the affected file, 0 on both other real VSM files),
# because the noise points sit orders of magnitude below the real ones.
_INV_CHI_NOISE_REL = 1e-3


def _chi_noise_floor(result):
    """|chi| below which 1/chi is reciprocal-of-noise, or None when undeterminable."""
    d = result.data or {}
    vals = []
    for b in (d.get("t_blocks") or []):
        vals.extend(v for v in (b.get("chi") or []) if v is not None)
    if not vals:
        is_si = d.get("inv_chi_unit") == "mol/m^3"
        vals = [v for v in (d.get("chi_molar_si" if is_si else "chi_molar_cgs") or [])
                if v is not None]
    arr = np.asarray(vals, float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    scale = float(np.median(np.abs(arr)))
    return _INV_CHI_NOISE_REL * scale if scale > 0 else None


def _mask_reciprocal_noise(inv_values, chi_values, floor):
    """Blank 1/chi wherever |chi| is below `floor`. Length- and order-preserving."""
    if floor is None or chi_values is None or len(chi_values) != len(inv_values):
        return list(inv_values)
    return [None if (c is not None and abs(float(c)) < floor) else v
            for c, v in zip(chi_values, inv_values)]


def series_inverse_chi(result, field_unit="Oe"):
    floor = _chi_noise_floor(result)
    d = result.data or {}
    is_si = d.get("inv_chi_unit") == "mol/m^3"
    flat_chi = d.get("chi_molar_si" if is_si else "chi_molar_cgs") or []
    return _vsm_series(
        result, "inv_chi", "1/χ",
        lambda b: _mask_reciprocal_noise(b.get("inv_chi") or [], b.get("chi") or [], floor),
        field_unit=field_unit,
        flat_y=lambda y: _mask_reciprocal_noise(y, flat_chi, floor))
def series_vsm_moment_t(result, field_unit="Oe"):
    return _vsm_series(result, "moment_per_fu", "Moment", lambda b: b.get("moment") or [], field_unit=field_unit)


def series_vsm_chi_t(result, field_unit="Oe"):
    """χ (left axis) + χ⁻¹ (right twin axis) — PQ-3 Item 2. The χ series keeps key "curve"
    (checklist back-compat: old presets with explicit curves still select it). The additive
    "inv_chi" series (role "inv_chi") drives the twin right axis; default_on so the twin shows
    by default, but deselectable to fall back to a plain single-axes χ plot. Both are unit-true:
    χ uses chi_molar_si under SI else chi_molar_cgs; inv_chi is already exported per unit_system
    (data['inv_chi_unit'] = 'mol/m^3' under SI)."""
    d = result.data or {}
    T = _arr(d, "temperature")
    is_si = d.get("inv_chi_unit") == "mol/m^3"
    chi = _arr(d, "chi_molar_si" if is_si else "chi_molar_cgs")
    out = []
    if T.size and chi.size == T.size:
        out.extend(_split_by_ramps(
            result, Series(key="curve", label="χ", x=T.tolist(), y=chi.tolist(), default_on=True),
            lambda b: b.get("chi") or [], field_unit=field_unit))
    inv = _arr(d, "inv_chi")
    if T.size and inv.size == T.size:
        floor = _chi_noise_floor(result)
        inv_masked = _mask_reciprocal_noise(inv.tolist(), chi.tolist() if chi.size == T.size else None,
                                            floor)
        out.extend(_split_by_ramps(
            result, Series(key="inv_chi", label="1/χ", x=T.tolist(), y=inv_masked,
                           default_on=True, role="inv_chi"),
            lambda b: _mask_reciprocal_noise(b.get("inv_chi") or [], b.get("chi") or [], floor),
            multi_label_prefix="1/χ ", field_unit=field_unit))
    return out

def series_vsm_chi_t_product(result, field_unit="Oe"):
    d = result.data or {}
    T = _arr(d, "temperature"); chi = _arr(d, "chi_molar_cgs")
    if T.size == 0 or chi.size != T.size:
        return []
    # χT per point; from t_blocks it multiplies the block's (unit-aware) chi by its temperature.
    return _split_by_ramps(
        result, Series(key="curve", label="χT", x=T.tolist(), y=(chi * T).tolist(), default_on=True),
        lambda b: [c * t for c, t in zip(b.get("chi") or [], b.get("temperature") or [])],
        field_unit=field_unit)


# ---- ACMS (AC susceptibility; one Series per (freq, amplitude, field, direction) curve) ----
def _acms_label(c):
    """Group legend label: '<full-precision freq> Hz, <2sf amp> Oe' [, '<field> Oe' when
    |field|>=50]. Frequency renders at FULL precision (real file -> '477 Hz', NOT '480 Hz');
    2-sig-fig formatting applies to amplitude only."""
    def sig2(x):
        if not np.isfinite(x) or x == 0:
            return "0"
        from math import floor, log10
        return f"{round(x, 1 - int(floor(log10(abs(x))))):g}"
    lbl = f"{c['frequency_hz']:g} Hz, {sig2(c['amplitude_oe'])} Oe"
    if abs(c.get("field_oe") or 0.0) >= 50.0:
        lbl += f", {c['field_oe']:.0f} Oe"
    return lbl


_ACMS_MARK = {"up": "o", "down": "^", "mixed": "s"}


def _acms_curve_series(result, which, prefix):
    """One Series per curve for 'chi_prime'/'chi_dprime' (molar when available). Curves of one
    (freq,amp,field) group share group label -> shared colour; marker encodes ramp direction."""
    d = result.data or {}
    out = []
    for i, c in enumerate(d.get("curves") or []):
        molar = c.get(which + "_molar")
        y = molar if molar else c.get(which)
        t = c.get("t") or []
        if not t or not y:
            continue
        lbl = _acms_label(c)
        # role = ramp direction; marker encodes up/down so ramps of one group never mint two
        # identical legend entries. marker override serves the bespoke headline renderer.
        out.append(Series(key=f"{prefix}:{i}", label=lbl, x=list(t), y=list(y),
                          group=lbl, role=c.get("direction"),
                          marker=_ACMS_MARK.get(c.get("direction"), "o")))
    return out


def series_acms_chi_t(result, field_unit="Oe"):
    return _acms_curve_series(result, "chi_prime", "chip") + \
           _acms_curve_series(result, "chi_dprime", "chipp")


def series_acms_chi_prime_t(result, field_unit="Oe"):
    return _acms_curve_series(result, "chi_prime", "chip")


def series_acms_chi_dprime_t(result, field_unit="Oe"):
    return _acms_curve_series(result, "chi_dprime", "chipp")


def series_acms_mdc_t(result, field_unit="Oe"):
    d = result.data or {}
    out = []
    for i, c in enumerate(d.get("curves") or []):
        md = c.get("m_dc")
        if not md:
            continue
        lbl = _acms_label(c)
        out.append(Series(key=f"mdc:{i}", label=lbl, x=list(c.get("t") or []), y=list(md),
                          group=lbl, marker=_ACMS_MARK.get(c.get("direction"), "o")))
    return out


def series_vsm_mh(result, field_unit="Oe"):
    """M(H) hysteresis loops — one Series per field-sweep branch (PQ-3 Task 2).

    Task 1's analyzer emits `data["loops"]`, one per contiguous field-sweep block; same-
    rounded-T blocks stay separate loops (dup-T = both hysteresis branches at one T). Key is
    always `mh:{T}K:{i}` (i = 0-based index among same-rounded-T loops) — a stable, unique
    key even when a T carries multiple branches. Label is `{T:.1f} K` with a " (2)"-style
    suffix on the 2nd+ loop at a repeated T. Group `{T}K` folds a T's branches onto one colour.
    x = field_oe (Oe), y = moment (μ_B/f.u.). [] when loops absent/empty ⇒ kind unavailable."""
    loops = (result.data or {}).get("loops") or []
    if not loops:
        return []
    seen: dict[float, int] = {}
    out = []
    for L in loops:
        T = L.get("temperature")
        field = L.get("field_oe") or []
        moment = L.get("moment") or []
        if T is None or not field or len(field) != len(moment):
            continue
        i = seen.get(T, 0); seen[T] = i + 1
        label = f"{T:.1f} K" if i == 0 else f"{T:.1f} K ({i + 1})"
        sc = _field_scale(field_unit)
        out.append(Series(key=f"mh:{T}K:{i}", label=label,
                          x=[v * sc for v in field], y=list(moment),
                          group=f"{T}K", default_on=True))
    return out

# ---- Resistivity ----
def _held(v, fmt):
    return (fmt % v) if v is not None else "na"

def _multi_channel(bridges, curve_key):
    """True when >1 bridge contributes curves of this kind -> labels need a channel prefix."""
    return sum(1 for b in bridges if b.get(curve_key)) > 1

def _chan_prefix(ch, multi):
    """'Ch{ch} ' only when multiple channels share a plot; single-channel stays unprefixed."""
    return f"Ch{ch} " if multi else ""

_CH_MARKERS = {1: "s", 2: "o", 3: "^", 4: "D"}   # channel -> marker when >1 channel plotted


def _mr_craft(ch, curve, multi):
    """Shared MR-family craft (PQ-4 Task 8): channel-scoped marker (only when >1 channel is
    plotted) and a cosmetic bidirectional-sweep arrow suffix. Returns (marker, label_suffix).
    Both are inert unless the render layer's channel_markers / direction_arrows knobs are on."""
    marker = _CH_MARKERS.get(ch) if multi else None
    dirs = set(curve.get("directions") or [])
    suffix = " ↑↓" if {1, -1} <= dirs else ""
    return marker, suffix

def series_resistivity_rho_t(result, field_unit="Oe"):
    out = []
    bridges = (result.data or {}).get("bridges", [])
    multi = _multi_channel(bridges, "rho_t_curves")
    for b in bridges:
        ch = b.get("channel"); curves = b.get("rho_t_curves", [])
        widest = max(curves, key=lambda c: len(c.get("temperature") or []), default=None)
        for c in curves:
            T = _arr(c, "temperature"); rho = _arr(c, "rho")
            if T.size == 0 or rho.size != T.size:
                continue
            key = f"b{ch}:T:{_held(c.get('held_field_oe'), '%.0f')}:{c.get('direction', 0)}"
            fld = c.get("held_field_oe")
            if fld is None:
                flabel = "ρ(T)"   # no Field column -> don't fake "na Oe"
            else:
                flabel = fmt_field(fld, field_unit) if field_unit == "T" else f"{fld:.0f} Oe"
            out.append(Series(key=key, label=f"{_chan_prefix(ch, multi)}{flabel}",
                              x=T.tolist(), y=rho.tolist(), group=f"Bridge {ch}",
                              default_on=(c is widest)))
    return out

def series_resistivity_rho_t2(result, field_unit="Oe"):
    out = []
    bridges = (result.data or {}).get("bridges", [])
    multi = _multi_channel(bridges, "rho_t_curves")
    for b in bridges:
        ch = b.get("channel"); curves = b.get("rho_t_curves", [])
        widest = max(curves, key=lambda c: len(c.get("temperature") or []), default=None)
        for c in curves:
            T = _arr(c, "temperature"); rho = _arr(c, "rho")
            if T.size == 0 or rho.size != T.size:
                continue
            key = f"b{ch}:T:{_held(c.get('held_field_oe'), '%.0f')}:{c.get('direction', 0)}"
            fld = c.get("held_field_oe")
            if fld is None:
                flabel = "ρ(T²)"   # no Field column -> don't fake "na Oe"
            else:
                flabel = fmt_field(fld, field_unit) if field_unit == "T" else f"{fld:.0f} Oe"
            out.append(Series(key=key, label=f"{_chan_prefix(ch, multi)}{flabel}",
                              x=(T * T).tolist(), y=rho.tolist(), group=f"Bridge {ch}",
                              default_on=(c is widest)))
    return out

def series_resistivity_mr(result, field_unit="Oe"):
    # PQ-4 Task 8: sort by held T ascending across ALL bridges so the render-layer colormap
    # runs dark->light in temperature order; channel markers + arrow suffix via _mr_craft.
    pairs = []
    bridges = (result.data or {}).get("bridges", [])
    multi = _multi_channel(bridges, "rho_h_curves")
    sc = _field_scale(field_unit)
    for b in bridges:
        ch = b.get("channel")
        for c in b.get("rho_h_curves", []):
            H = _arr(c, "field"); rho = _arr(c, "rho")
            if H.size == 0 or rho.size != H.size:
                continue
            key = f"b{ch}:H:{_held(c.get('held_temp_k'), '%.1f')}:{c.get('direction', 0)}"
            marker, sfx = _mr_craft(ch, c, multi)
            t = c.get("held_temp_k")
            pairs.append((t if t is not None else float("inf"),
                          Series(key=key,
                                 label=f"{_chan_prefix(ch, multi)}{_held(t, '%.1f')} K",
                                 x=(H * sc).tolist(), y=rho.tolist(), group=f"Bridge {ch}",
                                 default_on=True, marker=marker, label_suffix=sfx)))
    pairs.sort(key=lambda p: p[0])
    return [s for _, s in pairs]

def series_resistivity_mr_pct(result, field_unit="Oe"):
    # PQ-4 Task 8: same craft as series_resistivity_mr, around the rho0-normalisation.
    pairs = []
    bridges = (result.data or {}).get("bridges", [])
    multi = _multi_channel(bridges, "rho_h_curves")
    sc = _field_scale(field_unit)
    for b in bridges:
        ch = b.get("channel")
        for c in b.get("rho_h_curves", []):
            rho0 = c.get("rho_zero_field")
            if rho0 is None or rho0 <= 0:                 # cannot normalize -> skip loop
                continue
            H = _arr(c, "field"); rho = _arr(c, "rho")
            if H.size == 0 or rho.size != H.size:
                continue
            key = f"b{ch}:H:{_held(c.get('held_temp_k'), '%.1f')}:{c.get('direction', 0)}"
            marker, sfx = _mr_craft(ch, c, multi)
            t = c.get("held_temp_k")
            pairs.append((t if t is not None else float("inf"),
                          Series(key=key,
                                 label=f"{_chan_prefix(ch, multi)}{_held(t, '%.1f')} K",
                                 x=(H * sc).tolist(), y=((rho - rho0) / rho0 * 100.0).tolist(),
                                 group=f"Bridge {ch}", default_on=True,
                                 marker=marker, label_suffix=sfx)))
    pairs.sort(key=lambda p: p[0])
    return [s for _, s in pairs]

def series_resistivity_mr_pct_t(result, field_unit="Oe"):
    """MR% at H_max vs temperature (spec D6): one point per grouped field loop (the widest-ramp
    value per DQ-B, as surfaced by the analyzer), one series per channel, marker-by-channel when
    multi-channel. Backed iff >=1 loop carries mr_percent_at_max_field."""
    out = []
    bridges = (result.data or {}).get("bridges", [])
    multi = _multi_channel(bridges, "rho_h_curves")
    for b in bridges:
        ch = b.get("channel")
        pts = sorted((float(c["held_temp_k"]), float(c["mr_percent_at_max_field"]))
                     for c in b.get("rho_h_curves", [])
                     if c.get("mr_percent_at_max_field") is not None
                     and c.get("held_temp_k") is not None)
        if not pts:
            continue
        out.append(Series(key=f"b{ch}:mrT",
                          label=f"{_chan_prefix(ch, multi)}MR% @ H_max",
                          x=[p[0] for p in pts], y=[p[1] for p in pts],
                          group=f"Bridge {ch}", default_on=True,
                          marker=_CH_MARKERS.get(ch) if multi else None))
    return out

# ---- Hall ----
def _hall_points(result, ykey):
    return [p for p in (result.data or {}).get("points", []) if p.get(ykey) is not None]

def series_hall_rh_t(result, field_unit="Oe"):
    pts = _hall_points(result, "R_H")
    if not pts:
        return []
    pts = sorted(pts, key=lambda p: p["temperature"])
    return [Series(key="R_H", label="R_H",
                   x=[p["temperature"] for p in pts], y=[p["R_H"] for p in pts], default_on=True)]

def series_hall_mobility_t(result, field_unit="Oe"):
    pts = _hall_points(result, "mobility")
    if not pts:
        return []
    pts = sorted(pts, key=lambda p: p["temperature"])
    return [Series(key="mu", label="μ",
                   x=[p["temperature"] for p in pts], y=[p["mobility"] for p in pts], default_on=True)]

def series_hall_n_t(result, field_unit="Oe"):
    pts = _hall_points(result, "carrier_n")
    if not pts:
        return []
    pts = sorted(pts, key=lambda p: p["temperature"])
    return [Series(key="n", label="n",
                   x=[p["temperature"] for p in pts], y=[p["carrier_n"] for p in pts], default_on=True)]

def series_hall_r2_t(result, field_unit="Oe"):
    pts = _hall_points(result, "r2")
    if not pts:
        return []
    pts = sorted(pts, key=lambda p: p["temperature"])
    return [Series(key="r2", label="R²",
                   x=[p["temperature"] for p in pts], y=[p["r2"] for p in pts], default_on=True)]

# ---- Hall field-sweep curves (SP-2) ----
def series_hall_rxy_vs_B(result, field_unit="Oe"):
    """Raw measured R_xy across the full signed field sweep — one Series per held T."""
    pts = [p for p in (result.data or {}).get("points", []) if p.get("field_raw_T")]
    out = []
    for p in sorted(pts, key=lambda p: p["temperature"]):
        T = p["temperature"]
        out.append(Series(key=f"raw:{T}K", label=f"{T} K",
                          x=list(p["field_raw_T"]), y=list(p["R_xy_raw"]),
                          group=f"{T}K", default_on=True))
    return out

def series_hall_asym_vs_B(result, field_unit="Oe"):
    """Antisymmetrized R_asym vs |B| — one Series per held T (fit line added in render)."""
    pts = [p for p in (result.data or {}).get("points", []) if p.get("field_asym_T")]
    out = []
    for p in sorted(pts, key=lambda p: p["temperature"]):
        T = p["temperature"]
        out.append(Series(key=f"asym:{T}K", label=f"{T} K",
                          x=list(p["field_asym_T"]), y=list(p["R_asym"]),
                          group=f"{T}K", default_on=True))
    return out

def series_hall_raw_vs_asym(result, field_unit="Oe"):
    """Per T vs |B|: raw +B branch, raw -B branch (reflected), and R_asym. Branch
    separation = the even-in-H R_xx admixture that antisymmetrization removes."""
    pts = [p for p in (result.data or {}).get("points", [])
           if p.get("field_asym_T") and p.get("field_raw_T")]
    out = []
    for p in sorted(pts, key=lambda p: p["temperature"]):
        T = p["temperature"]
        pos_x = [b for b in p["field_raw_T"] if b > 0]
        pos_y = [r for b, r in zip(p["field_raw_T"], p["R_xy_raw"]) if b > 0]
        neg_x = [-b for b in p["field_raw_T"] if b < 0]
        neg_y = [r for b, r in zip(p["field_raw_T"], p["R_xy_raw"]) if b < 0]
        if pos_x:
            out.append(Series(key=f"rawpos:{T}K", label=f"R_xy(+B) {T} K",
                              x=pos_x, y=pos_y, group=f"{T}K", default_on=True,
                              role="R_xy(+B)"))
        if neg_x:
            out.append(Series(key=f"rawneg:{T}K", label=f"R_xy(−B) {T} K",
                              x=neg_x, y=neg_y, group=f"{T}K", default_on=True,
                              role="R_xy(−B)"))
        out.append(Series(key=f"asym:{T}K", label=f"R_asym {T} K",
                          x=list(p["field_asym_T"]), y=list(p["R_asym"]),
                          group=f"{T}K", default_on=True,
                          role="R_asym"))
    return out

# ---- Heat capacity ----
def series_cp_over_t(result, field_unit="Oe"):
    d = result.data or {}
    x = _arr(d, "t_squared"); y = _arr(d, "cp_over_t")
    if x.size == 0 or y.size != x.size:
        return []
    return [Series(key="curve", label="Cp/T", x=x.tolist(), y=y.tolist(), default_on=True)]

def series_hc_c_over_t_linear(result, field_unit="Oe"):
    d = result.data or {}
    T = _arr(d, "temperature"); y = _arr(d, "cp_over_t")
    if T.size == 0 or y.size != T.size:
        return []
    return [Series(key="curve", label="Cp/T", x=T.tolist(), y=y.tolist(), default_on=True)]

def series_cp_vs_t(result, field_unit="Oe"):
    d = result.data or {}
    # full-group data (NOT the low-T subset in "temperature"/"cp")
    T = _arr(d, "full_temperature"); y = _arr(d, "full_cp")
    if T.size == 0 or y.size != T.size:
        return []
    return [Series(key="curve", label="Cp", x=T.tolist(), y=y.tolist(), default_on=True)]

def series_hc_full_cp_t(result, field_unit="Oe"):
    """PQ-5 Task 5 headline: full Cp(T) vs T. One data series per field group (colour-by-field)
    when >1 group carries full Cp(T); else a single 'curve' series from full_temperature/full_cp.
    A solid-red Debye-Einstein fit series (role='fit') is appended only when full_fit is present
    and ok. Open-square markers + the Dulong–Petit line + low-T inset are drawn by the renderer."""
    d = result.data or {}
    out = []
    groups = [g for g in (d.get("field_groups") or [])
              if g.get("full_temperature") and g.get("full_cp")]
    if len(groups) > 1:
        for g in sorted(groups, key=lambda g: g.get("field_oe") or 0.0):
            f_oe = g.get("field_oe")
            tag = fmt_field(f_oe, "T") if f_oe is not None else "field"
            out.append(Series(key=f"cp@{f_oe:g}", label=tag,
                              x=list(g["full_temperature"]), y=list(g["full_cp"]), group=tag))
    else:
        T = _arr(d, "full_temperature"); y = _arr(d, "full_cp")
        if T.size and y.size == T.size:
            out.append(Series(key="curve", label="Cp", x=T.tolist(), y=y.tolist()))
    ff = d.get("full_fit")
    if ff and ff.get("ok"):
        x = ff.get("t_grid") or []; yf = ff.get("cp_fit") or []
        if x and len(yf) == len(x):
            r2 = ff.get("r2")
            lbl = f"Debye-Einstein (R²={r2:.3f})" if r2 is not None else "Debye-Einstein"
            out.append(Series(key="de_fit", label=lbl, x=list(x), y=list(yf), role="fit"))
    return out


def _drop_none_pairs(x, y):
    """Zip (x, y) keeping only indices where y is not None and finite. entropy_magnetic
    may carry None at out-of-overlap indices (truncated to the lattice T-overlap) — those
    must never reach matplotlib. Returns (xs, ys) as plain float lists."""
    xs, ys = [], []
    for xi, yi in zip(x, y):
        if yi is None:
            continue
        try:
            fv = float(yi)
        except (TypeError, ValueError):
            continue
        if fv != fv:                          # NaN guard
            continue
        xs.append(float(xi)); ys.append(fv)
    return xs, ys

def series_hc_entropy_vs_t(result, field_unit="Oe"):
    """Entropy S(T): total (solid) + magnetic (dashed, when a lattice subtraction ran) vs T,
    plus optional per-field-group overlays (colour-by-field, OFF by default to keep the
    headline uncluttered). x = entropy_temperature. None entries in entropy_magnetic are
    dropped (out-of-overlap truncation)."""
    d = result.data or {}
    if not d.get("entropy_available"):
        return []
    T = d.get("entropy_temperature") or []
    tot = d.get("entropy_total") or []
    out = []
    if T and len(tot) == len(T):
        out.append(Series(key="s_total", label="S total", x=[float(v) for v in T],
                          y=[float(v) for v in tot], role="total",
                          linestyle="-", default_on=True))
    mag = d.get("entropy_magnetic")
    if mag is not None:
        xm, ym = _drop_none_pairs(T, mag)
        if len(ym) >= 2:
            out.append(Series(key="s_magnetic", label="S magnetic", x=xm, y=ym,
                              role="magnetic", linestyle="--", default_on=True))
    # per-field overlays (available for selection; off by default)
    for g in d.get("field_groups") or []:
        eg = g.get("entropy")
        if not (eg and eg.get("s_total")):
            continue
        f_oe = g.get("field_oe")
        glabel = fmt_field(f_oe, "T") if f_oe is not None else "field"
        Tg = eg.get("temperature") or []
        sg = eg.get("s_total") or []
        if Tg and len(sg) == len(Tg):
            out.append(Series(key=f"s_total@{f_oe:g}", label=f"S total {glabel}",
                              x=[float(v) for v in Tg], y=[float(v) for v in sg],
                              group=glabel, role="total_field",
                              linestyle="-", default_on=False))
        mg = eg.get("s_magnetic")
        if mg is not None:
            xg, yg = _drop_none_pairs(Tg, mg)
            if len(yg) >= 2:
                out.append(Series(key=f"s_magnetic@{f_oe:g}", label=f"S magnetic {glabel}",
                                  x=xg, y=yg, group=glabel, role="magnetic_field",
                                  linestyle="--", default_on=False))
    return out

# ---- Hall TempDep (hall_tdep) -----------------------------------------------

def _tdep_points(result, key):
    """Return list of point dicts where ``key`` is not None (mirrors _hall_points)."""
    return [p for p in (result.data or {}).get("points", []) if p.get(key) is not None]


def series_hall_tdep_rh_t(result, field_unit="Oe"):
    """R_H vs T — antisym (trusted) and 2-point (0-field+1 fallback) as DISTINCT series,
    plus optional field-sweep overlay from dual_method."""
    out = []
    pts = _tdep_points(result, "R_H")
    anti = sorted([p for p in pts if p.get("r_h_method") != "2point"],
                  key=lambda p: p["temperature"])
    twop = sorted([p for p in pts if p.get("r_h_method") == "2point"],
                  key=lambda p: p["temperature"])
    if anti:
        out.append(Series(key="R_H_antisym", label="R_H (antisym)",
                          x=[p["temperature"] for p in anti],
                          y=[p["R_H"] for p in anti], default_on=True))
    if twop:
        out.append(Series(key="R_H_2point", label="R_H (0-field+1)",
                          x=[p["temperature"] for p in twop],
                          y=[p["R_H"] for p in twop], default_on=True, role="two_point"))
    dual = [d for d in (result.data or {}).get("dual_method", [])
            if d.get("R_H_fieldsweep") is not None]
    if dual:
        dual = sorted(dual, key=lambda d: d["temperature"])
        out.append(Series(key="R_H_fieldsweep", label="R_H (field sweep)",
                          x=[d["temperature"] for d in dual],
                          y=[d["R_H_fieldsweep"] for d in dual], default_on=True))
    return out


def series_hall_tdep_n_t(result, field_unit="Oe"):
    """Carrier concentration vs T — antisym and 2-point (fallback) as distinct series."""
    pts = _tdep_points(result, "carrier_n")
    anti = sorted([p for p in pts if p.get("r_h_method") != "2point"],
                  key=lambda p: p["temperature"])
    twop = sorted([p for p in pts if p.get("r_h_method") == "2point"],
                  key=lambda p: p["temperature"])
    out = []
    if anti:
        out.append(Series(key="n_antisym", label="n (antisym)",
                          x=[p["temperature"] for p in anti],
                          y=[p["carrier_n"] for p in anti], default_on=True))
    if twop:
        out.append(Series(key="n_2point", label="n (0-field+1)",
                          x=[p["temperature"] for p in twop],
                          y=[p["carrier_n"] for p in twop], default_on=True, role="two_point"))
    return out


def series_hall_tdep_mobility_t(result, field_unit="Oe"):
    """Mobility vs T + conductivity sigma vs T."""
    out = []
    pts_mu = _tdep_points(result, "mobility")
    if pts_mu:
        pts_mu = sorted(pts_mu, key=lambda p: p["temperature"])
        out.append(Series(key="mu", label="μ",
                          x=[p["temperature"] for p in pts_mu],
                          y=[p["mobility"] for p in pts_mu], default_on=True))
    pts_sigma = _tdep_points(result, "sigma")
    if pts_sigma:
        pts_sigma = sorted(pts_sigma, key=lambda p: p["temperature"])
        # sigma is ~1e8 S/m vs mu ~1e-3 m^2/Vs — on a shared linear axis sigma flattens
        # mu to zero, so default sigma OFF; the user can toggle it on (see SP-7 capture).
        out.append(Series(key="sigma", label="σ",
                          x=[p["temperature"] for p in pts_sigma],
                          y=[p["sigma"] for p in pts_sigma], default_on=False))
    return out


def series_hall_tdep_asym_vs_b(result, field_unit="Oe"):
    """R_asym vs |B| — one Series per temperature from stages.

    No PQ-2 ±branch-split upgrade here (unlike series_hall_raw_vs_asym): HallTDepStage.R_raw
    stores only the +field interpolated value per paired ±B setpoint (hall_tempdep.py
    _reconstruct_points sets Rraw from Rmap[pf] only), and R_zero_sub is set identical to
    R_raw ("zero-field subtraction is identity here") — the -field raw value is used
    transiently for antisymmetrization and never retained, so the -H branch is not
    reconstructible from stored data; this kind keeps its current single-curve form."""
    out = []
    for stage in (result.data or {}).get("stages", []):
        T = stage.get("temperature")
        fields = stage.get("fields_T", [])
        rasym = stage.get("R_asym", [])
        if not fields or not rasym or len(fields) != len(rasym):
            continue
        out.append(Series(key=f"asym:{T}K", label=f"{T} K",
                          x=list(fields), y=list(rasym),
                          group=f"{T}K", default_on=True))
    return out


def series_hall_tdep_interp_rt(result, field_unit="Oe"):
    """Interpolated R_xy(T) per fixed field — one Series per interp_curve."""
    out = []
    for c in (result.data or {}).get("interp_curves", []):
        field_oe = c.get("field_oe")
        T = c.get("temperature", [])
        R = c.get("R", [])
        if not T or not R or len(T) != len(R) or field_oe is None:
            continue
        label = fmt_field(field_oe, "T")
        out.append(Series(key=f"interp:{field_oe}Oe", label=label,
                          x=list(T), y=list(R),
                          group=label, default_on=True))
    return out


def series_hall_tdep_stages(result, field_unit="Oe"):
    """Stage diagnostics: R(+|B|) raw, R zero-sub, and R_asym vs |B| per temperature.
    PQ-2 Task 2: the zsub series is emitted whenever R_zero_sub is present and matches the
    fields length -- even though the current analyzer's zero-field subtraction is an identity
    (R_zero_sub == R_raw, see hall_tempdep.py:289 comment). The panel-level identity check
    (skip the Zero-subtracted panel when it would be visually redundant with Raw) lives in the
    renderer, not here, so this factory stays a straightforward data->Series mapping."""
    out = []
    for stage in (result.data or {}).get("stages", []):
        T = stage.get("temperature")
        fields = stage.get("fields_T", [])
        rraw = stage.get("R_raw", [])
        rzsub = stage.get("R_zero_sub", [])
        rasym = stage.get("R_asym", [])
        if not fields:
            continue
        if rraw and len(rraw) == len(fields):
            out.append(Series(key=f"raw:{T}K", label=f"R(+|B|) raw {T} K",
                              x=list(fields), y=list(rraw),
                              group=f"{T}K", default_on=True, role="R(+|B|) raw"))
        if rzsub and len(rzsub) == len(fields):
            out.append(Series(key=f"zsub:{T}K", label=f"R zero-sub {T} K",
                              x=list(fields), y=list(rzsub),
                              group=f"{T}K", default_on=True, role="R zero-sub"))
        if rasym and len(rasym) == len(fields):
            out.append(Series(key=f"asym:{T}K", label=f"R_asym {T} K",
                              x=list(fields), y=list(rasym),
                              group=f"{T}K", default_on=True, role="R_asym"))
    return out


# ---- PQ-2 Task 3: composite kinds -----------------------------------------

def series_hall_two_panel(result, field_unit="Oe"):
    """Left panel: R_xy(B) raw sweeps per T (same data as hall_rxy_vs_B). Right panel:
    literal zero-subtracted R_xx(B) color-by-T when a per-T longitudinal sweep is present,
    else rho_xx(T) scalar (one point per temperature that has a longitudinal channel matched).
    Gate: [] when no point carries rho_xx nor a per-T longitudinal sweep."""
    pts = (result.data or {}).get("points", [])
    has_rho = any(p.get("rho_xx") is not None for p in pts)
    has_rxxb = any(p.get("R_xx_raw") for p in pts)
    if not has_rho and not has_rxxb:
        return []
    out = []
    for p in sorted([p for p in pts if p.get("field_raw_T")], key=lambda p: p["temperature"]):
        T = p["temperature"]
        out.append(Series(key=f"rxy:{T}K", label=f"{T} K",
                          x=list(p["field_raw_T"]), y=list(p["R_xy_raw"]),
                          group=f"{T}K", default_on=True))
    rho_pts = sorted([p for p in pts if p.get("rho_xx") is not None],
                     key=lambda p: p["temperature"])
    if rho_pts:
        out.append(Series(key="rhoxx", label="ρ_xx",
                          x=[p["temperature"] for p in rho_pts],
                          y=[p["rho_xx"] for p in rho_pts], default_on=True))
    for p in sorted([p for p in pts if p.get("R_xx_raw")], key=lambda p: p["temperature"]):
        T = p["temperature"]
        out.append(Series(key=f"rxxb:{T}K", label=f"{T} K",
                          x=list(p["field_rxx_T"]), y=list(p["R_xx_raw"]),
                          group=f"{T}K", default_on=True, role="rxx_b"))
    return out


def series_hall_tdep_summary(result, field_unit="Oe"):
    """R_H / mobility / (optional) J vs T, one Series each. Gate: [] unless >=2 points carry
    R_H AND >=1 point carries mobility -- current_density_J is optional (renderer degrades to
    a 2-axis figure when absent, which is always the case on real analyzer output today)."""
    pts = (result.data or {}).get("points", [])
    rh_pts = [p for p in pts if p.get("R_H") is not None]
    mu_pts = [p for p in pts if p.get("mobility") is not None]
    if len(rh_pts) < 2 or len(mu_pts) < 1:
        return []
    rh_pts = sorted(rh_pts, key=lambda p: p["temperature"])
    mu_pts = sorted(mu_pts, key=lambda p: p["temperature"])
    out = [Series(key="rh", label="R_H", x=[p["temperature"] for p in rh_pts],
                  y=[p["R_H"] for p in rh_pts], default_on=True),
           Series(key="mu", label="μ", x=[p["temperature"] for p in mu_pts],
                  y=[p["mobility"] for p in mu_pts], default_on=True)]
    j_pts = sorted([p for p in pts if p.get("current_density_J") is not None],
                   key=lambda p: p["temperature"])
    if j_pts:
        out.append(Series(key="j", label="J", x=[p["temperature"] for p in j_pts],
                          y=[p["current_density_J"] for p in j_pts], default_on=True))
    return out


def series_hall_rh_n_twin(result, field_unit="Oe"):
    """R_H + carrier n vs T (shared by 'hall' and 'hall_tdep' -- both point schemas carry
    temperature/R_H/carrier_n under the same field names). Gate: [] unless >=2 points carry
    both R_H and carrier_n."""
    pts = [p for p in (result.data or {}).get("points", [])
           if p.get("R_H") is not None and p.get("carrier_n") is not None]
    if len(pts) < 2:
        return []
    pts = sorted(pts, key=lambda p: p["temperature"])
    return [Series(key="rh", label="R_H", x=[p["temperature"] for p in pts],
                   y=[p["R_H"] for p in pts], default_on=True),
            Series(key="n", label="n", x=[p["temperature"] for p in pts],
                   y=[p["carrier_n"] for p in pts], default_on=True)]


def series_hall_tdep_j_t(result, field_unit="Oe"):
    """Current density J vs T — gated; returns [] when J absent (always None currently)."""
    pts = _tdep_points(result, "current_density_J")
    if not pts:
        return []
    pts = sorted(pts, key=lambda p: p["temperature"])
    return [Series(key="J", label="J",
                   x=[p["temperature"] for p in pts],
                   y=[p["current_density_J"] for p in pts], default_on=True)]


# ---- Heat capacity param(H) series factory (Task 7) ----

_MODEL_LABELS = {"debye_t3": "Debye T³", "debye_t3_t5": "Debye T³+T⁵",
                 "spin_fluct_noninteracting": "spin-fl non-int", "spin_fluct_weak": "spin-fl weak"}

def _make_param_vs_field(param, allowed, gate_on_selected):
    """Factory: series fn for one parameter vs field. allowed = set of model keys.
    gate_on_selected: if True, a point is shown only where its model is AICc-selected."""
    def _series(result, field_unit="Oe"):
        fg = (result.data or {}).get("field_groups", [])
        if len(fg) < 2:
            return []
        out = []
        for key in allowed:
            xs, ys, es, om = [], [], [], []
            for g in fg:
                if g["status"] != "ok":
                    continue
                f = next((ff for ff in g["fits"] if ff["key"] == key and ff.get("ok")), None)
                if f is None:
                    continue
                val = f["params"].get(param)
                if val is None or not np.isfinite(val):
                    continue
                if param == "theta_D" and key not in ("debye_t3", "debye_t3_t5"):
                    continue
                if gate_on_selected and g.get("chosen_aicc_key") != key:
                    continue
                flagged = (not f.get("identifiable", True)) or bool(g.get("warnings"))
                if gate_on_selected and not f.get("identifiable", True):
                    continue                                 # A/T0: drop unresolved entirely
                sig = (f.get("sigma") or {}).get(param)      # may be None (sanitized non-finite)
                xs.append(g["field_oe"] * _field_scale(field_unit)); ys.append(float(val))
                es.append(float(sig) if sig is not None and np.isfinite(sig) else 0.0)
                om.append(flagged)
            if xs:
                out.append(Series(key=f"{param}:{key}", label=_MODEL_LABELS[key],
                                  x=xs, y=ys, yerr=es, open_mask=om, group=_MODEL_LABELS[key]))
        return out
    return _series

_ALL = ("debye_t3", "debye_t3_t5", "spin_fluct_noninteracting", "spin_fluct_weak")
_LATTICE = ("debye_t3", "debye_t3_t5")
_SPIN = ("spin_fluct_noninteracting", "spin_fluct_weak")
series_hc_gamma_vs_field  = _make_param_vs_field("gamma",   _ALL,     gate_on_selected=False)
series_hc_thetaD_vs_field = _make_param_vs_field("theta_D", _LATTICE, gate_on_selected=False)
series_hc_A_vs_field      = _make_param_vs_field("A",       _SPIN,    gate_on_selected=True)
series_hc_T0_vs_field     = _make_param_vs_field("T0",      _SPIN,    gate_on_selected=True)


def series_hc_lowt_multifield(result, field_unit="Oe"):
    """One data series per field group: Cp/T vs T² raw points. Returns [] when < 2 field groups."""
    fg = (result.data or {}).get("field_groups", [])
    if len(fg) < 2:
        return []
    out = []
    for g in fg:
        if g["status"] != "ok" or not g.get("t2"):
            continue
        tag = fmt_field(g['field_oe'], field_unit) if field_unit == "T" else f"{g['field_oe']:g} Oe"
        out.append(Series(key=f"mf:{g['field_oe']:g}", label=tag,
                          x=list(g["t2"]), y=list(g["cp_over_t"]),
                          group=tag))
    return out


# ---- Heat capacity Schottky plot-kind series factories (HC slice 3 / Task 7) ----

def _schottky_groups(result, field_unit="Oe"):
    """Return field groups eligible for Schottky plotting: schottky_enabled, >=2 groups, status ok,
    schottky sub-dict attempted."""
    fg = (result.data or {}).get("field_groups", [])
    if not (result.data or {}).get("schottky_enabled") or len(fg) < 2:
        return []
    return [g for g in fg if g.get("status") == "ok" and g.get("schottky", {}).get("attempted")]


def _make_schottky_param_series(param):
    """Factory: series fn for one Schottky parameter vs field.
    Points use hollow markers (open_mask=True) where delta_determined is False."""
    def _series(result, field_unit="Oe"):
        gs = _schottky_groups(result)
        xs, ys, es, om = [], [], [], []
        for g in gs:
            sc = g["schottky"]
            val = sc["params"].get(param)
            if val is None or not np.isfinite(val):
                continue
            xs.append(g["field_oe"] * _field_scale(field_unit))
            ys.append(float(val))
            sig = (sc.get("sigma") or {}).get(param)
            es.append(float(sig) if sig is not None and np.isfinite(sig) else 0.0)
            om.append(not sc.get("delta_determined", False))  # hollow where Δ not determined
        return [Series(key=f"schottky:{param}", label=param, x=xs, y=ys, yerr=es, open_mask=om)] if xs else []
    return _series


series_hc_delta_vs_field  = _make_schottky_param_series("Delta")
series_hc_f_vs_field      = _make_schottky_param_series("f")
series_hc_alphaN_vs_field = _make_schottky_param_series("alphaN")


def series_hc_schottky_multifield(result, field_unit="Oe"):
    """Per field group: raw Cp vs T (data points) + the chosen-model Schottky fit line, group-colored.
    Uses the windowed raw arrays fit_schottky stores (t_data/cp_data) plus the fit curve
    (t_grid/cp_fit) — the group's slice-2 t2/cp_over_t is Cp/T-vs-T², a different basis."""
    out = []
    for g in _schottky_groups(result):
        sc = g["schottky"]
        tag = fmt_field(g['field_oe'], field_unit) if field_unit == "T" else f"{g['field_oe']:g} Oe"
        if sc.get("t_data"):
            out.append(Series(key=f"schdata:{g['field_oe']:g}", label=tag,
                              x=list(sc["t_data"]), y=list(sc["cp_data"]), group=tag))
        if sc.get("t_grid"):
            out.append(Series(key=f"schfit:{g['field_oe']:g}", label=f"{tag} fit",
                              x=list(sc["t_grid"]), y=list(sc["cp_fit"]), group=tag, role="fit"))
    return out


# ---- Heat capacity transition (Tc) plot-kind series factories (HC slice 4 / Task 9) ----

def _transition_groups(result, field_unit="Oe"):
    """Return field groups eligible for transition plotting: transitions_enabled, >=2 groups,
    transition sub-dict attempted. NOT gated on status=="ok": the transition attempt is
    decoupled from the low-T sufficiency gate (slice-4 hardening) — a high-T-only group
    (status="insufficient") carries a valid T_c(H) attempt and must render."""
    fg = (result.data or {}).get("field_groups", [])
    if not (result.data or {}).get("transitions_enabled") or len(fg) < 2:
        return []
    return [g for g in fg if g.get("transition", {}).get("attempted")]


def series_hc_tc_vs_field(result, field_unit="Oe"):
    gs = _transition_groups(result)
    xs, ys, es, om = [], [], [], []
    for g in gs:
        trd = g["transition"]; val = trd.get("Tc")
        if val is None or not np.isfinite(val):
            continue
        xs.append(g["field_oe"] * _field_scale(field_unit)); ys.append(float(val))
        sig = trd.get("Tc_sigma")
        es.append(float(sig) if sig is not None and np.isfinite(sig) else 0.0)
        om.append(not trd.get("tc_determined", False))       # hollow where undetermined
    return [Series(key="tc:field", label="T_c", x=xs, y=ys, yerr=es, open_mask=om)] if xs else []


def series_hc_transition_multifield(result, field_unit="Oe"):
    """Per field group: raw Cp vs T (data points) + the fitted transition curve, group-colored."""
    out = []
    for g in _transition_groups(result):
        trd = g["transition"]; tag = fmt_field(g['field_oe'], field_unit) if field_unit == "T" else f"{g['field_oe']:g} Oe"
        if trd.get("t_data"):
            out.append(Series(key=f"trdata:{g['field_oe']:g}", label=tag,
                              x=list(trd["t_data"]), y=list(trd["cp_data"]), group=tag))
        if trd.get("grid"):
            out.append(Series(key=f"trfit:{g['field_oe']:g}", label=f"{tag} fit",
                              x=list(trd["grid"]), y=list(trd["cp_fit"]), group=tag, role="fit"))
    return out


def series_hc_transition_signal(result, field_unit="Oe"):
    """Per field group: background-subtracted residual signal vs T, group-colored."""
    out = []
    for g in _transition_groups(result):
        trd = g["transition"]; tag = fmt_field(g['field_oe'], field_unit) if field_unit == "T" else f"{g['field_oe']:g} Oe"
        if trd.get("resid_signal"):
            out.append(Series(key=f"trsig:{g['field_oe']:g}", label=tag,
                              x=list(trd["t_data"]), y=list(trd["resid_signal"]), group=tag))
    return out


# ---- TTO (thermal transport; one Series per (field group, ramp direction) curve) ----
_TTO_MARK = {"up": "o", "down": "^", "mixed": "s"}
_TTO_DIRWORD = {"up": "warming", "down": "cooling", "mixed": "mixed"}


def _tto_label(curve, field_unit="Oe"):
    """Legend label for one TTO curve. `field_unit` MUST keep its default — pq_compare's
    _render_v2 calls KINDS[kind].series(result) with no field_unit. The field is shown via the
    existing Oe/T display convention only when |H| >= 50 Oe, so a single-field file (the real
    gate file sits at 0.077 Oe) gets a direction-only label."""
    d = curve.get("direction")
    word = _TTO_DIRWORD.get(d, d or "")
    f = curve.get("field_oe") or 0.0
    if abs(f) >= 50.0:
        return f"{fmt_field(f, field_unit)}, {word}"
    return word


def _tto_curve_series(result, ykey, prefix, field_unit="Oe", group=None, linestyle=None):
    """One Series per curve for a TTO per-point array.

    SERIES KEYS ARE A PERSISTED CONTRACT (select_series matches s.key; presets.py validates
    saved layouts against them; the stacked headline renderer routes panels by key prefix).
    Scheme `<prefix>:<field_oe %g>:<dir>` — do not change it after the slice ships.
    Per-point None holes become NaN here (plot space only; the JSON envelope keeps nulls);
    pinned by test_none_holes_become_nan_in_plot_space.
    `group` overrides the colour group — tto_wf_t colours by COMPONENT, not by field.
    `linestyle` is either a fixed style for every curve or a {field_oe: style} map (tto_wf_t
    encodes the FIELD in the linestyle when more than one field group is present).
    `yerr` is the matching `<ykey>_std` array when the curve carries one (drawn ONLY by
    `_tto_draw`'s band; see I3 — the generic `_plot_data` would turn it into capped
    errorbars)."""
    d = result.data or {}
    out = []
    for c in d.get("curves") or []:
        y = c.get(ykey)
        t = c.get("t") or []
        if not t or not y:
            continue
        lbl = _tto_label(c, field_unit)
        f = c.get("field_oe") or 0.0
        # ±1σ source for the opt-in band (spec §3). Only the four MEASURED quantities feed
        # yerr. Since 2026-08-10 the derived quantities (kappa_e, kappa_ph, lorenz_ratio)
        # DO carry `_std` arrays in the JSON/CSV envelope — but their band WIRING stays
        # deferred (§10 of the uncertainty spec), so they are deliberately excluded here:
        # attaching them would silently light up the opt-in band path unreviewed.
        std = c.get(f"{ykey}_std") if ykey in _TTO_MEASURED_STD_KEYS else None
        yerr = None
        # `is not None`, NOT truthiness: today every `*_std` is a list, but an ndarray from a
        # future analyzer would make `if std` raise "truth value of an array is ambiguous".
        if std is not None and len(std) == len(t):
            yerr = [float("nan") if (v is None or not np.isfinite(float(v))) else float(v)
                    for v in std]
        out.append(Series(
            key=f"{prefix}:{f:g}:{c.get('direction')}",
            label=(lbl if group is None else f"{group} · {lbl}"),
            x=list(t),
            y=[float("nan") if v is None else float(v) for v in y],
            yerr=yerr,
            group=(lbl if group is None else group),
            role=c.get("direction"),
            marker=_TTO_MARK.get(c.get("direction"), "o"),
            linestyle=(linestyle.get(f) if isinstance(linestyle, dict) else linestyle)))
    return out


def series_tto_kappa_t(result, field_unit="Oe"):
    return _tto_curve_series(result, "kappa", "kappa", field_unit)


def series_tto_seebeck_t(result, field_unit="Oe"):
    return _tto_curve_series(result, "seebeck", "seebeck", field_unit)


def series_tto_zt_t(result, field_unit="Oe"):
    return _tto_curve_series(result, "zt", "zt", field_unit)


def series_tto_lorenz_t(result, field_unit="Oe"):
    """L/L0 = kappa*rho/(L0*T) per curve. DERIVED, so no `lorenz_ratio_std` exists and the
    series carry no yerr -- uncertainty propagation through the derived quantities is
    explicitly deferred (spec §7). [] when no curve carries a lorenz_ratio."""
    return _tto_curve_series(result, "lorenz_ratio", "lorenz", field_unit)


_TTO_MEASURED_STD_KEYS = {"kappa", "seebeck", "rho", "zt"}   # yerr sources; derived excluded

_TTO_FIELD_LS = ("-", "--", ":", "-.")


def tto_field_ls_map(curves):
    """{field_oe: linestyle} when a TTO file carries MORE THAN ONE field group, else None.

    tto_wf_t colours by COMPONENT (κ/κ_e/κ_ph — the brief's requirement), so on a multi-field
    file both field groups would otherwise draw in the same three colours AND the same
    direction marker, making the field unrecoverable from the figure. The field therefore
    moves onto the linestyle, and `_tto_handles` adds one grey linestyle proxy per field to
    the folded legend. Single-field files return None -> component linestyles, unchanged."""
    fields = []
    for c in curves or []:
        f = c.get("field_oe") or 0.0
        if f not in fields:
            fields.append(f)
    if len(fields) < 2:
        return None
    return {f: _TTO_FIELD_LS[i % len(_TTO_FIELD_LS)] for i, f in enumerate(fields)}


def tto_field_ls_label(field_oe, field_unit="Oe"):
    """Legend text for a field-linestyle proxy (same display convention as `_tto_label`)."""
    return fmt_field(field_oe, field_unit) if field_unit == "T" else f"{field_oe:g} Oe"


def series_tto_wf_t(result, field_unit="Oe"):
    """kappa / kappa_e / kappa_ph per curve, distinguished by Series.group (colour) with
    linestyle as a secondary cue — the FIELD when several field groups are present (see
    `tto_field_ls_map`), the component otherwise. [] when no curve carries kappa_e -> the
    kind is unavailable for this data (acms_mdc_t pattern). An all-None kappa_e list counts
    as absent: a non-empty list of nulls is truthy, and advertising it would draw three
    all-NaN lines under a 3-entry legend."""
    curves = (result.data or {}).get("curves") or []
    if not any(any(v is not None for v in (c.get("kappa_e") or [])) for c in curves):
        return []
    ls = tto_field_ls_map(curves)
    return (_tto_curve_series(result, "kappa", "kappa", field_unit, group=r"$\kappa$",
                              linestyle=(ls or "-"))
            + _tto_curve_series(result, "kappa_e", "kappa_e", field_unit, group=r"$\kappa_\mathrm{e}$",
                                linestyle=(ls or "--"))
            + _tto_curve_series(result, "kappa_ph", "kappa_ph", field_unit, group=r"$\kappa_\mathrm{ph}$",
                                linestyle=(ls or ":")))


def series_tto_summary_t(result, field_unit="Oe"):
    """Headline series: kappa + Seebeck + rho, one Series per curve per quantity. The stacked
    renderer routes each Series to its panel by the key prefix, so the three prefixes must
    stay distinct."""
    return (_tto_curve_series(result, "kappa", "kappa", field_unit)
            + _tto_curve_series(result, "seebeck", "seebeck", field_unit)
            + _tto_curve_series(result, "rho", "rho", field_unit))


BUILTIN_PLOTKINDS = [
    PlotKind("inverse_chi", "1/χ vs T", "vsm", series_inverse_chi),
    PlotKind("vsm_moment_t", "Moment vs T", "vsm", series_vsm_moment_t),
    PlotKind("vsm_chi_t", "χ vs T", "vsm", series_vsm_chi_t),
    PlotKind("vsm_chi_t_product", "χT vs T", "vsm", series_vsm_chi_t_product),
    PlotKind("vsm_mh", "M vs H", "vsm", series_vsm_mh),
    PlotKind("resistivity_rho_t", "ρ vs T", "resistivity", series_resistivity_rho_t),
    PlotKind("resistivity_mr", "ρ vs H (MR)", "resistivity", series_resistivity_mr),
    PlotKind("resistivity_mr_pct", "MR % vs H", "resistivity", series_resistivity_mr_pct),
    PlotKind("resistivity_mr_pct_t", "MR % vs T", "resistivity", series_resistivity_mr_pct_t),
    PlotKind("resistivity_rho_t2", "ρ vs T²", "resistivity", series_resistivity_rho_t2),
    PlotKind("hall_rh_t", "R_H vs T", "hall", series_hall_rh_t),
    PlotKind("hall_mobility_t", "μ vs T", "hall", series_hall_mobility_t),
    PlotKind("hall_n_t", "carrier n vs T", "hall", series_hall_n_t, default_yscale="log"),
    PlotKind("hall_r2_t", "Hall R² vs T", "hall", series_hall_r2_t),
    PlotKind("hall_rxy_vs_B", "R_xy vs B (raw)", "hall", series_hall_rxy_vs_B),
    PlotKind("hall_asym_vs_B", "R_asym vs |B|", "hall", series_hall_asym_vs_B),
    PlotKind("hall_raw_vs_asym", "Antisymmetrization", "hall", series_hall_raw_vs_asym,
             group_colored=True),
    PlotKind("hall_two_panel", "Hall | Longitudinal", "hall", series_hall_two_panel),
    PlotKind("hall_rh_n_twin", "R_H + carrier n vs T", "hall", series_hall_rh_n_twin),
    PlotKind("cp_over_t", "Cp/T (low-T) vs T²", "heatcapacity", series_cp_over_t),
    PlotKind("hc_c_over_t_linear", "Cp/T vs T (low-T)", "heatcapacity", series_hc_c_over_t_linear),
    PlotKind("cp_vs_t", "Cp vs T (full range)", "heatcapacity", series_cp_vs_t),
    PlotKind("hc_entropy_vs_t", "Entropy S(T)", "heatcapacity", series_hc_entropy_vs_t),
    PlotKind("hc_full_cp_t", "Cp(T) + Dulong-Petit", "heatcapacity", series_hc_full_cp_t),
    PlotKind("hc_gamma_vs_field", "γ vs H", "heatcapacity", series_hc_gamma_vs_field),
    PlotKind("hc_thetaD_vs_field", "θ_D vs H", "heatcapacity", series_hc_thetaD_vs_field),
    PlotKind("hc_A_vs_field", "A vs H (spin-fl)", "heatcapacity", series_hc_A_vs_field),
    PlotKind("hc_T0_vs_field", "T₀ vs H (spin-fl)", "heatcapacity", series_hc_T0_vs_field),
    PlotKind("hc_lowt_multifield", "Cp/T vs T² (multi-field)", "heatcapacity",
             series_hc_lowt_multifield, group_colored=True),
    PlotKind("hc_delta_vs_field", "Δ vs H (Schottky)", "heatcapacity", series_hc_delta_vs_field),
    PlotKind("hc_f_vs_field", "f vs H (Schottky)", "heatcapacity", series_hc_f_vs_field),
    PlotKind("hc_alphaN_vs_field", "αN vs H (nuclear)", "heatcapacity", series_hc_alphaN_vs_field),
    PlotKind("hc_schottky_multifield", "Cp vs T (Schottky)", "heatcapacity",
             series_hc_schottky_multifield, group_colored=True),
    PlotKind("hc_tc_vs_field", "T_c vs H (transition)", "heatcapacity", series_hc_tc_vs_field),
    PlotKind("hc_transition_multifield", "Cp vs T (transition)", "heatcapacity",
             series_hc_transition_multifield, group_colored=True),
    PlotKind("hc_transition_signal", "Transition residual vs T", "heatcapacity",
             series_hc_transition_signal, group_colored=True),
    # ---- Hall TempDep ----
    PlotKind("hall_tdep_RH_T", "R_H vs T (temp-dep)", "hall_tdep", series_hall_tdep_rh_t),
    PlotKind("hall_tdep_n_T", "carrier n vs T", "hall_tdep", series_hall_tdep_n_t, default_yscale="log"),
    PlotKind("hall_tdep_mobility_T", "μ vs T", "hall_tdep", series_hall_tdep_mobility_t),
    PlotKind("hall_tdep_asym_vs_B", "R_asym vs |B|", "hall_tdep", series_hall_tdep_asym_vs_b),
    PlotKind("hall_tdep_interp_RT", "R_xy(T) per field", "hall_tdep", series_hall_tdep_interp_rt),
    PlotKind("hall_tdep_stages", "Stage diagnostics", "hall_tdep", series_hall_tdep_stages,
             group_colored=True),
    PlotKind("hall_tdep_J_T", "J vs T", "hall_tdep", series_hall_tdep_j_t),
    PlotKind("hall_tdep_summary", "R_H / μ / J vs T (summary)", "hall_tdep",
             series_hall_tdep_summary),
    PlotKind("hall_tdep_rh_n_twin", "R_H + carrier n vs T", "hall_tdep", series_hall_rh_n_twin),
    PlotKind("acms_chi_t", "χ′/χ″ vs T", "acms", series_acms_chi_t, group_colored=True),
    PlotKind("acms_chi_prime_t", "χ′ vs T", "acms", series_acms_chi_prime_t, group_colored=True),
    PlotKind("acms_chi_dprime_t", "χ″ vs T", "acms", series_acms_chi_dprime_t, group_colored=True),
    PlotKind("acms_mdc_t", "M-DC vs T", "acms", series_acms_mdc_t, group_colored=True),
    PlotKind("tto_summary_t", "κ / S / ρ vs T", "tto", series_tto_summary_t,
             group_colored=True),
    PlotKind("tto_kappa_t", "κ vs T", "tto", series_tto_kappa_t, group_colored=True),
    PlotKind("tto_seebeck_t", "S vs T", "tto", series_tto_seebeck_t, group_colored=True),
    PlotKind("tto_zt_t", "ZT vs T", "tto", series_tto_zt_t, group_colored=True),
    PlotKind("tto_wf_t", "κ decomposition vs T", "tto", series_tto_wf_t, group_colored=True),
    PlotKind("tto_lorenz_t", "L/L₀ vs T", "tto", series_tto_lorenz_t, group_colored=True),
]

# ---- get_kind lookup (Task 7 C1) ----
_BY_KEY = {k.key: k for k in BUILTIN_PLOTKINDS}
def get_kind(key):
    return _BY_KEY[key]

def build_default_layout(plot_kinds, result):
    """A PlotEntry per backed kind (series() non-empty), in catalog order."""
    from cryosweep_core.plotting.spec import PlotEntry, PlotLayout
    entries = [PlotEntry(kind=k.key) for k in plot_kinds if k.series(result)]
    return PlotLayout(plots=entries)


@dataclass(frozen=True)
class OverlayFile:
    file_id: int
    label: str
    colour: str | None = None


def overlay_series(kind, results, overlay, field_unit="Oe") -> list[Series]:
    """File-qualified union of a kind's series across overlaid results (feeds the file-grouped checklist).
    Effective key = f'{file_id}::{raw_key}'; group = file label.
    NOTE: '::' is a RESERVED delimiter between the integer file_id and the raw key. Raw keys never
    contain '::' (they use single ':' or plain words, e.g. 'b1:T:0:1'); the integer file_id prefix
    keeps effective keys unambiguous even if a future raw key did."""
    out = []
    for r, of in zip(results, overlay):
        for s in kind.series(r, field_unit=field_unit):
            out.append(Series(key=f"{of.file_id}::{s.key}", label=f"{of.label} · {s.label}",
                              x=s.x, y=s.y, group=of.label, default_on=s.default_on))
    return out
