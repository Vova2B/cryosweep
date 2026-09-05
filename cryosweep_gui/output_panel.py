from __future__ import annotations
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QScrollArea,
                             QTableWidget, QTableWidgetItem, QLabel, QFrame,
                             QPushButton, QButtonGroup, QGridLayout, QSizePolicy,
                             QHeaderView)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from cryosweep_core.plotting.render import render_kind, NothingToPlot
from cryosweep_core.plotting.spec import PlotSpec, GlobalStyle
from cryosweep_core.plotting.catalog import build_default_layout, overlay_series, BUILTIN_PLOTKINDS
from cryosweep_core.registry import build_default_registry
from cryosweep_gui.plot_controls import AxisStrip

_SCREEN_DPI = 100
_KIND_MAP = {k.key: k for k in BUILTIN_PLOTKINDS}
_SCALARS = (int, float, str, bool)

# Friendly names for capability keys shown in the hint strip.
_CAP_LABELS = {
    "hall_coefficient": "R_H", "carrier_concentration": "carrier n",
    "mobility": "mobility", "antisymmetrization": "antisymmetrization",
    "dual_method": "dual-method",
    "superconducting_transition": "Superconducting Tc",
    "ac_susceptibility": "AC χ", "superconducting_screening": "SC screening",
    "chi_dprime_peak": "χ″ peak T_f", "molar_normalization": "molar χ",
    "dc_magnetization": "M-DC",
    "thermal_conductivity": "κ(T)", "seebeck": "Seebeck",
    "wiedemann_franz": "Wiedemann-Franz", "power_factor": "power factor",
    "figure_of_merit": "ZT", "rrr": "RRR", "kappa_ph_power_fit": "κ_ph power law",
    # resistivity.py:372 spells the same capability "RRR"; map both so neither probe shows a
    # raw key. (The differing spellings are spec-mandated per probe and are NOT unified here.)
    "RRR": "RRR",
}
# Advisory capabilities that are not about a missing plot -> don't nag in the strip.
# The four TTO entries are permanently inapplicable ("deferred") on a probe that has NO user
# inputs (D4), so the strip's "Set the required inputs in the panel at left." remedy would be
# both permanent and false.
_CAP_ADVISORY = {"rich_field_recommended",
                 "callaway_fit", "boundary_scattering_fit", "diffusive_seebeck",
                 "kappa_field_sweep"}
# Probes whose input panel contributes NOTHING the user can set (TTO reads its geometry from
# the file header, D4). The strip's remedy clause is false there for EVERY capability, not just
# the deferred ones above — "requires finite ρ > 0" is a property of the FILE, and pointing at
# an input panel that holds one explanatory sentence sends the user looking for a knob that does
# not exist. On these probes the diagnosis alone is the whole message.
_NO_INPUT_PROBES = {"tto"}
_HINT_REMEDY = "   Set the required inputs in the panel at left."


def _capability_hint(data: dict | None) -> str:
    """Build a one-line 'why is this empty' hint from a result's inapplicable capabilities,
    e.g. 'Not computed — R_H: thickness required for R_H · mobility: no longitudinal channel.'
    Returns '' when every capability is applicable (or none are reported)."""
    caps = (data or {}).get("capabilities") or []
    unmet = [c for c in caps if isinstance(c, dict) and not c.get("applicable", True)
             and c.get("reason") and c.get("name") not in _CAP_ADVISORY]
    if not unmet:
        return ""
    parts = [f"{_CAP_LABELS.get(c['name'], c['name'])}: {c['reason']}" for c in unmet]
    remedy = "" if (data or {}).get("probe") in _NO_INPUT_PROBES else _HINT_REMEDY
    return "Not computed — " + " · ".join(parts) + "." + remedy

def _kappa_ph_row(kf: dict) -> str:
    """The κ_ph fit row's value (I9).

    Deliberate wording: the ladder number is WINDOW SENSITIVITY, not an error bar. Writing
    "2.03 (spread 1.31-2.03)" reads as "n = 2.03, uncertain by 0.71", which is a DIFFERENT
    over-claim — the <=10 K fit's uncertainty is 0.006; 0.71 is how far n moves when you change
    what you are asking, and 0.019 is how far it moves when you change HOW you fit it.

    Three further rules, from the Task-8 review:
      * NO "±" on n. "±" is the universal glyph for "uncertainty on the preceding number", and a
        parenthetical "(stat)" two tokens later does not undo it — the eye binds "2.03 ± 0.0062"
        as a unit. The statistical scatter is therefore written "σ_stat 0.0062 (fit scatter only,
        not the uncertainty on n)" and moved LAST, behind the sensitivities that dominate it.
      * `window_sensitive` is COMPUTED by the analyzer and exported to the CSV, so it must also
        be a word on screen — silently dropping it made the GUI the least honest surface.
      * When the flag is set, the headline is rounded to one decimal ("n ≈ 2.0"): displayed
        precision must not exceed what the window ladder shows is actually determined.

    C1 (final review): EVERY quality flag is now spelled out in a trailing "flags:" clause,
    verbatim and in the analyzer's order — the same string the CSV's `kappa_ph_flags` column
    carries, so the two surfaces are literally comparable. `window_sensitive` is NOT excluded
    even though the joiner already shouts it: special-casing which flags reach the screen is
    exactly how `n_at_bound` and `kappa_e_dominant` came to be computed, exported and never
    shown. (The two flags that make n meaningless — `n_at_bound`, `degenerate_window` — now
    decline the fit in the analyzer, so this row is never even built for them; the clause
    still carries them because a future flag must not need a GUI edit to become visible.)
    """
    flags = list(kf.get("quality_flags") or [])
    sensitive = "window_sensitive" in flags
    win = kf.get("window_k") or [None, None]
    cut = f" (≤{win[1]:g} K fit)" if win[1] is not None else ""
    head = (f"n ≈ {kf['n']:.1f}{cut}" if sensitive else f"n = {kf['n']:.3g}{cut}")

    parts: list[str] = []
    cf = [e for e in (kf.get("ladder") or []) if e.get("method") == "curve_fit"]
    if kf.get("n_spread") is None or len(cf) < 2:
        parts.append("n(window spread) = not measured")
    else:
        # m7: FIRST -> LAST rung, as spec §5 asks. Note this is not the same quantity as the
        # CSV's `kappa_ph_n_spread`, which is max - min over the whole ladder: on a
        # NON-MONOTONE ladder the two surfaces disagree and this one understates. Both the gate
        # file and tto_real_subset are monotone (2.03 -> 1.80 -> 1.59 -> 1.31), so they agree
        # today; the identity is not guaranteed.
        parts.append(f"n({cf[0]['cutoff_k']:g}→{cf[-1]['cutoff_k']:g} K) = "
                     f"{cf[0]['n']:.3g}→{cf[-1]['n']:.3g}")
    if kf.get("n_loglog") is not None:
        parts.append(f"log-log {kf['n_loglog']:.3g}")
    if flags:
        # BEFORE the sigma clause, not after: "σ_stat ... (fit scatter only, not the
        # uncertainty on n)" stays the LAST thing on the row by the Task-8 rule above.
        parts.append("flags: " + ", ".join(flags))
    parts.append(f"σ_stat {kf['n_sigma']:.2g}"
                 " (fit scatter only, not the uncertainty on n)")
    joiner = " — WINDOW-SENSITIVE: " if sensitive else "; "
    return head + joiner + "; ".join(parts)


def _cw_theta_row(data: dict) -> str:
    """The window-sensitive Curie-Weiss row (2026-08-10 spec §7, U3 idiom).

    Same copy law as _kappa_ph_row: NO ± glyph anywhere (the spread is window sensitivity,
    not an error bar), the ladder drift is the signal, and σ_stat comes LAST, qualified.

    F3 (final-review fix, 2026-08-10): the HEADLINE is the SHIPPED fit's theta — the same
    number the plot annotation, `fit_params.csv` and `result.json` carry. Spec §7's draft copy
    headlined the T>=25 K rung instead (-42 on the MPMS file), which silently re-windowed the
    published value; §1.3 (authoritative, encoding owner decision O1 "flag + penalize
    confidence, do NOT re-window") says "the headline stays -50.3 ... every published number
    is unchanged". The rungs are now clearly-subordinate context, exactly as the TTO
    kappa_ph row treats its ladder."""
    fit = data.get("fit") or {}
    ladder = data.get("cw_ladder") or []
    full_theta = float((fit.get("params") or {}).get("theta"))
    sig = (fit.get("sigma") or {}).get("theta")
    first, last = ladder[0], ladder[-1]
    head = f"θ = {full_theta:.1f} K (full-window fit — REPORTED)"
    parts = [f"θ(full→{last['tmin_k']:g} K) = {full_theta:.1f}→{last['theta_k']:.1f}",
             f"T≥{first['tmin_k']:g} K rung gives {first['theta_k']:.1f}"]
    if data.get("theta_spread_k") is not None:
        parts.append(f"spread {data['theta_spread_k']:.3g} K")
    if sig is not None:
        parts.append(f"σ_stat {sig:.2g} K"
                     " (fit scatter only, not the uncertainty on θ)")
    return head + " — WINDOW-SENSITIVE: " + "; ".join(parts)


from cryosweep_core.fitting.transport import POWER_LAW_DECLINE_FLAGS, ARRHENIUS_DECLINE_FLAGS


def _rho_powerlaw_row(pl: dict, ladder: list | None, n_spread) -> str:
    """The resistivity power-law row — the TTO corrected row idiom verbatim (_kappa_ph_row's
    rules: headline to 1 dp when window-sensitive, no ±, flags: clause, σ_stat last)."""
    flags = list(pl.get("quality_flags") or [])
    declined = [f for f in flags if f in POWER_LAW_DECLINE_FLAGS]
    if declined:
        # Declined: report WHY and the numbers behind the refusal, never the exponent itself.
        nsig = (pl.get("sigma") or {}).get("n")
        n_val = (pl.get("params") or {}).get("n")
        why = ", ".join(declined)
        bits = [f"declined — {why}"]
        if n_val is not None and nsig is not None:
            bits.append(f"n would be {float(n_val):.3g} with σ_stat {float(nsig):.3g}")
        if pl.get("r2") is not None:
            bits.append(f"r² {float(pl['r2']):.3g}")
        fr_d = pl.get("fit_range") or [None, None]
        if fr_d[0] is not None and fr_d[1] is not None:
            bits.append(f"window {float(fr_d[0]):.3g}–{float(fr_d[1]):.3g} K")
        return "; ".join(bits)
    sensitive = "window_sensitive" in flags
    fr = pl.get("fit_range") or [None, None]
    # .3g: fit_range is data-derived (e.g. 29.9144 K) — a display cutoff, not a claim
    cut = f" (≤{fr[1]:.3g} K fit)" if fr[1] is not None else ""
    n = float((pl.get("params") or {}).get("n"))
    head = (f"n ≈ {n:.1f}{cut}" if sensitive else f"n = {n:.3g}{cut}")
    parts: list[str] = []
    lad = list(ladder or [])
    if n_spread is None or len(lad) < 2:
        parts.append("n(window spread) = not measured")
    else:
        parts.append(f"n({lad[0]['cutoff_k']:g}→{lad[-1]['cutoff_k']:g} K) = "
                     f"{lad[0]['n']:.3g}→{lad[-1]['n']:.3g}")
    if flags:
        parts.append("flags: " + ", ".join(flags))
    nsig = (pl.get("sigma") or {}).get("n")
    if nsig is not None:
        parts.append(f"σ_stat {nsig:.3g}"
                     " (fit scatter only, not the uncertainty on n)")
    joiner = " — WINDOW-SENSITIVE: " if sensitive else "; "
    return head + joiner + "; ".join(parts)


def _arrhenius_row(ar: dict, spread) -> str:
    """Activated-transport row — same idiom as _rho_powerlaw_row. E_a is reported AS
    MEASURED; the gap line says ONLY IF intrinsic (the factor-of-two trap), and a declined
    fit reports why with the numbers behind the refusal, never a headline E_a."""
    flags = list(ar.get("quality_flags") or [])
    declined = [f for f in flags if f in ARRHENIUS_DECLINE_FLAGS]
    ea = (ar.get("params") or {}).get("e_a_mev")
    sig = (ar.get("sigma") or {}).get("e_a_mev")
    if declined:
        bits = [f"declined — {', '.join(declined)}"]
        if ea is not None and sig is not None:
            bits.append(f"E_a would be {float(ea):.3g} meV with σ_stat {float(sig):.3g}")
        if ar.get("r2") is not None:
            bits.append(f"r² {float(ar['r2']):.3g}")
        return "; ".join(bits)
    sensitive = "window_sensitive" in flags
    head = f"E_a ≈ {float(ea):.0f} meV" if sensitive else f"E_a = {float(ea):.3g} meV"
    parts = [f"E_g = 2·E_a = {2 * float(ea):.3g} meV ONLY IF intrinsic"]
    if spread is not None:
        parts.append(f"E_a(window spread) = {float(spread):.3g} meV")
    if flags:
        parts.append("flags: " + ", ".join(flags))
    if sig is not None:
        parts.append(f"σ_stat {float(sig):.3g} (fit scatter only)")
    joiner = " — WINDOW-SENSITIVE: " if sensitive else "; "
    return head + joiner + "; ".join(parts)


def flatten_rows(data: dict) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for k, v in data.items():
        if v is None or isinstance(v, _SCALARS):
            rows.append((k, "—" if v is None else str(v)))
    fit = data.get("fit")
    if isinstance(fit, dict):
        for pk, pv in (fit.get("params") or {}).items():
            rows.append((f"fit.{pk}", str(pv)))
        if fit.get("r2") is not None:
            rows.append(("fit.r2", str(fit["r2"])))
    for c in (data.get("capabilities") or []):
        rows.append((f"capability:{c['name']}", f"{c.get('applicable')} — {c.get('reason','')}"))
    if data.get("probe") == "vsm":
        # U3: the window-sensitive CW row (spec §7). Only when the analyzer flagged the fit —
        # a clean fit keeps today's generic fit.* rows and NOTHING else (pinned by test).
        if "window_sensitive" in ((fit or {}).get("quality_flags") or []) and data.get("cw_ladder"):
            rows.append(("Curie-Weiss θ", _cw_theta_row(data)))
    if data.get("probe") == "hall":
        for p in data.get("points") or []:
            if not isinstance(p, dict):
                continue
            t, rh = p.get("temperature"), p.get("R_H")
            if t is None or rh is None:
                continue
            sig = p.get("r_h_sigma")
            if sig is not None:
                # residual sigma = fit scatter; labeled so it can never be read as the
                # instrument sigma (O4 — the two must stay unmistakably distinct).
                val = f"{rh:.4g} ± {sig:.2g} m³/C (σ residual — fit scatter)"
            elif p.get("sigma_zero_dof"):
                val = f"{rh:.4g} m³/C (no σ — 2-point method)"
            else:
                val = f"{rh:.4g} m³/C"
            rows.append((f"R_H@{t:.1f}K", val))
    if data.get("probe") == "hall_tdep":
        # 138 points on real files — aggregate rows, not one per point.
        pts = [p for p in (data.get("points") or [])
               if isinstance(p, dict) and p.get("R_H") is not None]
        if pts:
            n_res = sum(1 for p in pts if p.get("r_h_sigma") is not None)
            n_2pt = sum(1 for p in pts if p.get("sigma_zero_dof"))
            rows.append(("R_H(T) σ (residual)",
                         f"{n_res}/{len(pts)} points carry a residual σ (fit scatter); "
                         f"{n_2pt} are 2-point (no σ)"))
            inst = sorted(p["r_h_sigma_instrument"] for p in pts
                          if p.get("r_h_sigma_instrument") is not None)
            if inst:
                med = inst[len(inst) // 2]
                # F7 (final-review): an absolute m³/C sigma is unreadable until divided by
                # |R_H| — on the real Hall file the median sigma_inst is 139 % of the value it qualifies.
                # The relative number is the entire point, so it rides NEXT TO the number
                # rather than only in the status banner.
                rel = sorted(p["r_h_sigma_instrument"] / abs(p["R_H"]) for p in pts
                             if p.get("r_h_sigma_instrument") is not None and p.get("R_H"))
                rel_txt = f" = {100 * rel[len(rel) // 2]:.0f}% of |R_H|" if rel else ""
                rows.append(("R_H(T) σ_inst",
                             f"median {med:.3g} m³/C{rel_txt} on {len(inst)}/{len(pts)} "
                             "points — σ_inst (instrument noise, not fit quality)"))
    if data.get("probe") == "resistivity":
        for b in data.get("bridges") or []:
            ch = b.get("channel")
            if b.get("rrr") is not None:
                std = b.get("rrr_std")
                # F13 (final-review): rrr_std is INSTRUMENT-derived (propagated from the
                # file's rho-std columns), so U5/O4 require it to be labeled as instrument
                # noise wherever it reaches a surface — a bare ± here was the one unqualified
                # error bar in the slice. It also excludes the dominant systematic (which
                # ramp, which endpoints), which the label now says out loud.
                rows.append((f"ch{ch}.RRR", f"{b['rrr']:.4g}" if std is None
                             else f"{b['rrr']:.4g} ± {std:.2g} "
                                  "(σ_inst — instrument noise; excludes ramp/endpoint choice)"))
            pl = b.get("power_law")
            if pl and (pl.get("params") or {}).get("n") is not None:
                rows.append((f"ch{ch}.ρ = ρ₀ + A·Tⁿ",
                             _rho_powerlaw_row(pl, b.get("power_law_ladder"),
                                               b.get("power_law_n_spread"))))
            ar = b.get("arrhenius")
            if ar:
                rows.append((f"ch{ch}.ρ = ρ₀·exp(E_a/k_BT)",
                             _arrhenius_row(ar, b.get("arrhenius_ea_spread_mev"))))
            for c in b.get("rho_h_curves") or []:
                mr = c.get("mr_percent_at_max_field")
                t = c.get("held_temp_k")
                if mr is None or t is None:
                    continue
                flag = " (low confidence)" if c.get("low_confidence") else ""
                rows.append((f"ch{b.get('channel')}.mr%@{t:.1f}K", f"{mr:.2f}%{flag}"))
    if data.get("probe") == "heatcapacity":
        sug = data.get("entropy_rln_suggestion")
        if sug:
            label, rel = sug.get("label"), sug.get("rel_err")
            if sug.get("matched"):
                val = f"{label} (matched, {rel * 100:.0f}% off)"
            elif rel is not None:
                val = f"{label} (NOT matched — S_mag saturation is {rel * 100:.0f}% away)"
            else:
                # rel_err None = no magnetic saturation to compare against (e.g. S_mag absent)
                val = f"{label} (NOT matched — no S_mag saturation to compare)"
            rows.append(("Rln suggestion", val))
    if data.get("probe") == "acms":
        for c in data.get("curves") or []:
            cp = c.get("chi_prime") or []
            if not cp:
                continue
            tag = f"{c.get('frequency_hz'):.0f}Hz,{c.get('amplitude_oe'):.3g}Oe,{c.get('direction')}"
            rows.append((f"{tag}.n", str(c.get("n_points"))))
            rows.append((f"{tag}.chi'@Tmin/Tmax", f"{cp[0]:.3g} / {cp[-1]:.3g}"))
            if c.get("sc"):
                rows.append((f"{tag}.Tc", f"{c['sc']['tc_mid_k']:.3f} K"))
            if c.get("peak"):
                rows.append((f"{tag}.T_f", f"{c['peak']['t_f_k']:.3f} K"))
    if data.get("probe") == "tto":
        for c in data.get("curves") or []:
            t = c.get("t") or []
            kap = c.get("kappa") or []
            if not t or not kap:
                continue
            # curve arrays are T-ascending, so [0] is T_min and [-1] is T_max
            tag = f"{c.get('field_oe', 0.0):.4g}Oe,{c.get('direction')}"
            rows.append((f"{tag}.n", str(c.get("n_points"))))
            rows.append((f"{tag}.κ@Tmin/Tmax", f"{kap[0]:.3g} / {kap[-1]:.3g}"))
            see = c.get("seebeck")
            if see and see[-1] is not None:
                rows.append((f"{tag}.S@Tmax", f"{see[-1]:.3g} µV/K"))
        rrr = data.get("rrr")
        if rrr:
            std = rrr.get("rrr_std")
            rows.append(("RRR", f"{rrr.get('rrr'):.4g}" if std is None
                         else f"{rrr.get('rrr'):.4g} ± {std:.2g}"))
            rows.append(("classification", str(rrr.get("classification"))))
        summary = data.get("summary") or {}
        if summary.get("pf_at_thigh") is not None:
            rows.append(("PF @ T_high", f"{summary['pf_at_thigh']:.4g} W/(K²·m)"))
        if summary.get("zt_peak") is not None and summary.get("zt_peak_t_k") is not None:
            # honesty: a maximum sitting at an end of the measured range is not an observed
            # peak — the sweep stopped there (analyzer's `zt_peak_at_edge`).
            label = ("ZT peak (at T range edge)" if summary.get("zt_peak_at_edge")
                     else "ZT peak")
            zstd = summary.get("zt_peak_std")
            val = (f"{summary['zt_peak']:.4g}" if zstd is None
                   else f"{summary['zt_peak']:.4g} ± {zstd:.2g}")
            rows.append((label, f"{val} @ {summary['zt_peak_t_k']:.1f} K"))
        kf = data.get("kappa_ph_fit")
        if kf:
            rows.append(("κ_ph ~ T^n", _kappa_ph_row(kf)))
        # `n_error_rows` is already emitted by the generic scalar walker above; a second
        # "error rows" row for the same number just read as two different measurements.
    return rows

class _WrapTable(QTableWidget):
    """The field/value table, with row heights that track the STRETCH column's real width (I2).

    `resizeRowsToContents()` on repopulate and on `sectionResized` is not enough: the stretch
    column's final width is computed by the header AFTER a view resize, and `sectionResized`
    does not fire for that recomputation. Measured on the real TTO file (offscreen, panel
    resized 1100 -> 520 -> 600 -> 650 px), the κ_ph row's height was one resize STALE at every
    step -- 31 px (2 lines) at 520 px where the text needs 3.7 lines, so `log-log 2.01` and the
    whole "fit scatter only, not the uncertainty on n" caveat were elided. The tooltip carried
    them, but the two clauses that stop a reader binding σ_stat to n must not be the part that
    gets cut. Recomputing from the VIEW's own resize fixes the lag at every width.

    Not recursive: `resizeRowsToContents` changes row heights only. It can add or remove the
    vertical scrollbar, which resizes the viewport once more; that second pass computes the
    same heights and stops."""

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.resizeRowsToContents()


class PlotCard(QFrame):
    """One plot kind: a canvas, or a placeholder when render raises/gated."""
    edited = Signal()

    def __init__(self, results, entry, style, overlay=None, parent=None):
        super().__init__(parent)                         # parent from birth: never a top-level
        self.entry = entry                               # widget, so it can't flash as a window on macOS
        self._lay = QVBoxLayout(self)
        self.canvas = None
        self.figure = None
        self.manual_line = None                  # live "model (manual)" Line2D, display-only
        self._badge = None
        self._dup_badge = None
        self._placeholder = None
        self._results = results
        self._style = style
        self._overlay = overlay
        kind = _KIND_MAP[entry.kind]
        self.title = QLabel(kind.label)
        self.title.setStyleSheet("font-weight:bold;")
        self._lay.addWidget(self.title)
        _fu = getattr(style, "field_unit", "Oe")
        series = (overlay_series(kind, results, overlay, field_unit=_fu) if overlay is not None
                  else kind.series(results[0], field_unit=_fu))
        self.strip = AxisStrip(series, entry.spec, kind)
        self.strip.spec_changed.connect(self._on_spec_changed)
        self._lay.addWidget(self.strip)
        self.render(results, style, overlay)
        self._maybe_add_badge(results)

    def _maybe_add_badge(self, results):
        all_diags = [d for r in results for d in (getattr(r, "diagnostics", []) or [])]
        outs = [d for d in all_diags if d.kind == "outliers"]
        dups = [d for d in all_diags if d.kind == "duplicate_setpoints"]
        if outs:
            n = sum(d.data.get("n_outliers", 0) for d in outs)
            badge = QLabel(f"⚠ {n} outliers")
            badge.setStyleSheet("color:#b35900; font-weight:bold;")
            badge.setToolTip("; ".join(f"{d.scope}: {d.message}" for d in outs))
            self._lay.insertWidget(1, badge)
            self._badge = badge
        if dups:
            dbadge = QLabel(f"⚠ {len(dups)} setpoint warning(s)")
            dbadge.setStyleSheet("color:#b35900; font-weight:bold;")
            dbadge.setToolTip("; ".join(f"{d.scope}: {d.message}" for d in dups))
            self._lay.insertWidget(1, dbadge)
            self._dup_badge = dbadge

    def render(self, results, style, overlay=None):
        self._style = style
        if self.canvas is not None:
            self.canvas.hide()                       # hide before orphaning (macOS window flash)
            self._lay.removeWidget(self.canvas); self.canvas.setParent(None)
            self.canvas.deleteLater(); self.canvas = None; self.figure = None
        self.manual_line = None                      # dies with the old figure
        if self._placeholder is not None:
            self._placeholder.hide()
            self._lay.removeWidget(self._placeholder); self._placeholder.setParent(None)
            self._placeholder.deleteLater(); self._placeholder = None
        try:
            display_style = style.model_copy(update={"dpi": _SCREEN_DPI})
            fig = render_kind(results, self.entry.kind, self.entry.spec, display_style, overlay=overlay)
            # overlay mode is a comparison view, not QC of one file -> suppress per-file status warning
            status = results[0].status if overlay is None else "ok"
            if status in ("low_confidence", "gated"):
                fig.suptitle(f"⚠ {status}")
            self.canvas = FigureCanvasQTAgg(fig)
            self.canvas.setMinimumHeight(280)
            # Expanding vertically so the canvas absorbs extra grid-cell height
            # (fixes the 1×2 vertical gap); title/AxisStrip stay stretch 0 at top.
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding,
                                      QSizePolicy.Policy.Expanding)
            self._lay.addWidget(self.canvas, 1)
            self.figure = fig
        except NothingToPlot:                               # benign: the file has no such data
            ph = QLabel(f"{self.entry.kind}: not applicable — no data of this kind in this file")
            ph.setWordWrap(True)                            # a long one-line label would widen the
            ph.setStyleSheet("color:#999; padding:12px;")   # grid cell and squeeze the plot cards
            self._lay.addWidget(ph)
            self._placeholder = ph
        except Exception as e:                              # genuine rendering failure: keep it visible
            msg = str(e)
            ph = QLabel(f"{self.entry.kind}: rendering failed — {type(e).__name__}"
                        + (f": {msg[:200]}" if msg else ""))
            ph.setWordWrap(True)
            ph.setStyleSheet("color:#b00020; padding:12px; font-weight:bold;")
            self._lay.addWidget(ph)
            self._placeholder = ph

    def _on_spec_changed(self):
        self.render(self._results, self._style, self._overlay)
        self.edited.emit()


def _grid_dims(n: int) -> int:
    """Return number of columns for n cards per the spec:
    1→1, 2→2, 3–4→2, 5–6→3, >6→2.
    """
    if n <= 1:
        return 1
    if n == 2:
        return 2
    if n <= 4:
        return 2
    if n <= 6:
        return 3
    return 2


class OutputPanel(QWidget):
    layout_edited = Signal()

    def __init__(self):
        super().__init__()
        self._registry = build_default_registry()
        self.style = GlobalStyle()
        self._cards: list[PlotCard] = []
        self._focus_index: int = 0
        self._mode: str = "grid"

        root = QVBoxLayout(self)

        # ── Grid / Focus toggle bar ──────────────────────────────────────────
        bar = QHBoxLayout()
        self._grid_btn = QPushButton("Grid")
        self._focus_btn = QPushButton("Focus")
        self._grid_btn.setCheckable(True)
        self._focus_btn.setCheckable(True)
        self._grid_btn.setChecked(True)
        self._mode_group = QButtonGroup(self)
        self._mode_group.setExclusive(True)
        self._mode_group.addButton(self._grid_btn)
        self._mode_group.addButton(self._focus_btn)
        bar.addWidget(self._grid_btn)
        bar.addWidget(self._focus_btn)
        # Focus navigation: prev/next stepper (only meaningful in Focus mode)
        self._focus_prev_btn = QPushButton("◀")
        self._focus_next_btn = QPushButton("▶")
        self._focus_prev_btn.setMaximumWidth(36)
        self._focus_next_btn.setMaximumWidth(36)
        self._focus_label = QLabel("")
        bar.addSpacing(12)
        bar.addWidget(self._focus_prev_btn)
        bar.addWidget(self._focus_label)
        bar.addWidget(self._focus_next_btn)
        bar.addStretch(1)
        root.addLayout(bar)

        # ── capability hint strip: explains WHY expected plots are empty ─────
        # (e.g. "R_H: thickness required") so missing-input plots aren't silent.
        self._cap_strip = QLabel("")
        self._cap_strip.setWordWrap(True)
        self._cap_strip.setStyleSheet(
            "color:#8a6d00; background:#fff8e1; border:1px solid #ffe082;"
            " border-radius:4px; padding:6px 8px;")
        self._cap_strip.setVisible(False)
        root.addWidget(self._cap_strip)

        # ── _empty sentinel — sibling of _scroll, NOT inside the grid ───────
        self._empty = QLabel("(no plots selected)")
        self._empty.setStyleSheet("color:#777; padding:20px;")
        root.addWidget(self._empty)

        # ── scroll area with QGridLayout host ────────────────────────────────
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._stack_host = QWidget()
        self._grid = QGridLayout(self._stack_host)
        self._scroll.setWidget(self._stack_host)
        self._scroll.setVisible(False)          # hidden until first plot
        root.addWidget(self._scroll, 1)

        self.table = _WrapTable(0, 2)
        self.table.setHorizontalHeaderLabels(["field", "value"])
        # The default 100 px columns TRUNCATE every long value while ~550 px of the widget sits
        # empty — measured on the real TTO file the κ_ph row needs 371 px and rendered as
        # "n = 2.03 ± ...", i.e. the headline number with every corrective clause cut off. That
        # is the exact over-claim the integrity slice exists to prevent, so the value column
        # STRETCHES to the free width and every value cell also carries itself as a tooltip
        # (see show_result) for anything still too long for the stretched column.
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        # Stretching alone is not enough: the honest κ_ph row is ~160 chars (809 px measured),
        # longer than the stretched column at any ordinary window size. So the value column also
        # WRAPS, and row heights are recomputed on every repopulate and on every column resize —
        # otherwise the wrapped lines 2..n are clipped by a one-line-tall row.
        self.table.setWordWrap(True)
        # ElideNone belongs WITH the wrap: eliding is what silently drops the tail of a value,
        # and with correct row heights (see _WrapTable) there is nothing left to elide.
        self.table.setTextElideMode(Qt.TextElideMode.ElideNone)
        hh.sectionResized.connect(lambda *_: self.table.resizeRowsToContents())
        root.addWidget(self.table)

        # test/back-compat hooks
        self.last_figure = None
        self.placeholder_shown = False
        self.status_stamped = False

        # connect toggle + navigation buttons
        self._grid_btn.clicked.connect(self._on_grid_mode)
        self._focus_btn.clicked.connect(self._on_focus_mode)
        self._focus_prev_btn.clicked.connect(lambda: self._step_focus(-1))
        self._focus_next_btn.clicked.connect(lambda: self._step_focus(+1))
        self._update_nav()

    # ── mode handlers ─────────────────────────────────────────────────────────

    def _on_grid_mode(self):
        # QButtonGroup (exclusive) already set check states before this slot runs.
        self._mode = "grid"
        self._apply_mode()

    def _on_focus_mode(self):
        self._mode = "focus"
        self._apply_mode()

    def _step_focus(self, delta: int):
        """Move the focused card by delta (wraps); only acts in Focus mode."""
        n = len(self._cards)
        if n == 0:
            return
        self._focus_index = (self._focus_index + delta) % n
        if self._mode != "focus":            # navigating implies focusing
            self._mode = "focus"
            self._focus_btn.setChecked(True)
        self._apply_mode()

    def _apply_mode(self):
        """Show/hide cards according to current mode; keep _focus_index valid."""
        n = len(self._cards)
        if n:
            self._focus_index = max(0, min(self._focus_index, n - 1))
        if self._mode == "focus":
            for i, card in enumerate(self._cards):
                card.setVisible(i == self._focus_index)
        else:
            for card in self._cards:
                card.setVisible(True)
        self._update_nav()

    def _update_nav(self):
        """Enable nav buttons + label only when Focus mode has >1 card."""
        n = len(self._cards)
        nav_on = self._mode == "focus" and n > 1
        self._focus_prev_btn.setEnabled(nav_on)
        self._focus_next_btn.setEnabled(nav_on)
        if self._mode == "focus" and n:
            self._focus_label.setText(f"{self._focus_index + 1}/{n}")
        else:
            self._focus_label.setText("")

    # ── layout helpers ────────────────────────────────────────────────────────

    def _relayout_grid(self):
        """Re-place every card into the QGridLayout at its correct (row, col),
        with equal row/column stretch so cells fill the viewport evenly. Stale
        stretches from a previous (larger) arrangement are reset to 0 so empty
        ghost cells don't claim space."""
        n = len(self._cards)
        # clear stretch for every column/row the grid has ever used
        for c in range(self._grid.columnCount()):
            self._grid.setColumnStretch(c, 0)
        for r in range(self._grid.rowCount()):
            self._grid.setRowStretch(r, 0)
        if n == 0:
            self._update_nav()      # clear stale focus label + disable nav on empty view
            return
        ncols = _grid_dims(n)
        nrows = (n + ncols - 1) // ncols
        for i, card in enumerate(self._cards):
            row = i // ncols
            col = i % ncols
            self._grid.addWidget(card, row, col)
        for col in range(ncols):
            self._grid.setColumnStretch(col, 1)
        for r in range(nrows):
            self._grid.setRowStretch(r, 1)
        self._apply_mode()

    def _clear_cards(self):
        for c in self._cards:
            # hide BEFORE orphaning: on macOS, reparenting a visible-flagged widget to None
            # briefly realizes it as its own top-level window (flashes + steals activation)
            c.hide()
            self._grid.removeWidget(c); c.setParent(None); c.deleteLater()
        self._cards = []
        self.last_figure = None

    # ---- live "model (manual)" overlay (ROADMAP 2b) ----
    # Display-only: it never enters the analysis result, so CSV/JSON/report and the
    # "Export plots…" path (which re-render from the result) can never carry it. It lives
    # and dies with the on-screen figure (_clear_cards / PlotCard.render drop it).

    MANUAL_GID = "manual_model"

    def update_manual_curve(self, kinds, x, y, label) -> None:
        """Draw or update the hand-set model curve on the cards for *kinds*. A model
        evaluation, never a refit: dashed, its own gid, labelled distinctly from the fit."""
        from cryosweep_core.plotting.render import refresh_legend
        for card in self._cards:
            if card.entry.kind not in kinds or card.figure is None:
                continue
            ax = card.figure.axes[0]                     # main axes (insets are appended after)
            ln = card.manual_line
            if ln is not None and ln.axes is ax:
                ln.set_data(x, y)
            else:
                ln, = ax.plot(x, y, ls="--", color="#1f77b4", lw=1.4, zorder=6,
                              gid=self.MANUAL_GID, label=label)
                card.manual_line = ln
                refresh_legend(ax, self.style, card.entry.spec)
            card.canvas.draw_idle()

    def clear_manual_curves(self) -> None:
        from cryosweep_core.plotting.render import refresh_legend
        for card in self._cards:
            ln = card.manual_line
            if ln is None:
                continue
            try:
                ln.remove()
            except (ValueError, NotImplementedError):
                pass                                     # already gone with a re-render
            card.manual_line = None
            if card.figure is not None:
                refresh_legend(card.figure.axes[0], self.style, card.entry.spec)
                card.canvas.draw_idle()

    def _default_layout(self, result):
        probe = (result.data or {}).get("probe")
        kinds = self._registry.plot_kinds_for(probe)
        return build_default_layout(kinds, result)

    def show_result(self, results, layout=None, overlay=None) -> None:
        results = results if isinstance(results, list) else [results]
        self._clear_cards()
        self.placeholder_shown = False
        primary = results[0]
        self.status_stamped = (overlay is None) and primary.status in ("low_confidence", "gated")
        layout = layout if layout is not None else self._default_layout(primary)
        hint = "" if overlay is not None else _capability_hint(primary.data)
        self._cap_strip.setText(hint)
        self._cap_strip.setVisible(bool(hint))
        has_plots = len(layout.plots) > 0
        self._empty.setVisible(not has_plots)
        self._scroll.setVisible(has_plots)
        for entry in layout.plots:
            card = PlotCard(results, entry, self.style, overlay, parent=self)  # parented -> no window flash
            self._cards.append(card)
            card.edited.connect(self.layout_edited)
            if card.figure is not None and self.last_figure is None:
                self.last_figure = card.figure
        if not self._cards or all(c.figure is None for c in self._cards):
            self.placeholder_shown = True
        self._relayout_grid()
        rows = flatten_rows(primary.data or {})       # field/value table = the primary/focused file
        self.table.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            self.table.setItem(i, 0, QTableWidgetItem(k))
            item = QTableWidgetItem(v)
            item.setToolTip(v)          # nothing is unreadable even if the column still elides
            self.table.setItem(i, 1, item)
        self.table.resizeRowsToContents()
