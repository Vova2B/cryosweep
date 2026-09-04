from __future__ import annotations

import numpy as np

def _diagnostics_block(result):
    """Return (json_list, markdown_lines) for result.diagnostics (empty -> ([], []))."""
    diags = getattr(result, "diagnostics", []) or []
    if not diags:
        return [], []
    jl = [d.model_dump() for d in diags]
    lines = ["## Data quality", "", "| severity | scope | message |", "|---|---|---|"]
    for d in diags:
        lines.append(f"| {d.severity} | {d.scope} | {d.message} |")
    lines.append("")
    return jl, lines


def build_report(result) -> dict:
    d = result.data
    if d.get("probe") == "resistivity":
        return _resistivity_report(result)
    if d.get("probe") == "hall":
        return _hall_report(result)
    if d.get("probe") == "heatcapacity":
        return _heatcapacity_report(result)
    fit = d.get("fit", {})
    j = {"probe": d.get("probe"), "status": result.status, "confidence": result.confidence,
         "fit": fit, "warnings": result.warnings, "gate": [g.model_dump() for g in result.gate]}
    lines = ["# CryoSweep Analysis Report", "",
             f"- **Probe:** {d.get('probe')}", f"- **Status:** {result.status}",
             f"- **Confidence:** {result.confidence:.3f}", ""]
    if fit:
        model = fit.get("model", "")
        label = model.replace("_", " ").title() if model else ""
        lines += [f"## Fit — {label} (`{model}`)", "", "| param | value | ± σ | unit |", "|---|---|---|---|"]
        for k, v in fit.get("params", {}).items():
            s = fit.get("sigma", {}).get(k); u = fit.get("units", {}).get(k, "")
            lines.append(f"| {k} | {v:.4g} | {('%.2g' % s) if s is not None else ''} | {u} |")
        lines += ["", f"R² = {fit.get('r2'):.5f}, n = {fit.get('n_points')}"]
    if result.gate:
        lines += ["", "## Gated outputs"] + [f"- `{g.need}`: {g.reason}" for g in result.gate]
    dj, dlines = _diagnostics_block(result)
    j["diagnostics"] = dj
    if dlines:
        lines += dlines
    return {"json": j, "markdown": "\n".join(lines)}


def _resistivity_report(result) -> dict:
    d = result.data
    bridges = d.get("bridges", [])
    caps = d.get("capabilities", [])
    j = {"probe": "resistivity", "status": result.status, "confidence": result.confidence,
         "rho_source": d.get("rho_source"), "bridges": bridges, "capabilities": caps,
         "warnings": result.warnings}
    lines = ["# CryoSweep Analysis Report", "",
             "- **Probe:** resistivity", f"- **Status:** {result.status}",
             f"- **Confidence:** {result.confidence:.3f}",
             f"- **rho source:** {d.get('rho_source')}"]
    routing = next((c for c in caps if c.get("name") == "hall_channel_excluded"), None)
    if routing is not None:
        lines.append(f"- **Hall-channel routing:** {routing['reason']}")
    lines.append("")
    if bridges:
        lines += ["## Bridges", "",
                  "| bridge | class | RRR | residual rho0 (Ohm*cm) | power-law n | rho(T) curves | rho(H) curves |",
                  "|---|---|---|---|---|---|---|"]
        for b in bridges:
            pl = b.get("power_law") or {}
            n = pl.get("params", {}).get("n")
            rrr = b.get("rrr"); res0 = b.get("residual_rho")
            lines.append("| {ch} | {cl} | {rrr} | {res0} | {n} | {nt} | {nh} |".format(
                ch=b["channel"], cl=b.get("classification"),
                rrr=("%.3f" % rrr) if rrr is not None else "-",
                res0=("%.3g" % res0) if res0 is not None else "-",
                n=("%.3f" % n) if n is not None else "-",
                nt=len(b.get("rho_t_curves", [])), nh=len(b.get("rho_h_curves", []))))
        lines.append("")
    mr_rows = []
    for b in bridges:
        for c in b.get("rho_h_curves", []):
            mr = c.get("mr_percent_at_max_field")
            if mr is not None:
                mr_rows.append((b["channel"], c.get("held_temp_k"), c.get("direction"),
                                c.get("max_abs_field_oe"), mr, c.get("low_confidence")))
    if mr_rows:
        lines += ["## Magnetoresistance", "",
                  "| bridge | T (K) | dir | |H|max (Oe) | MR% | low-conf |", "|---|---|---|---|---|---|"]
        for ch, Tk, dirn, hmax, mr, low in mr_rows:
            lines.append("| {ch} | {T} | {d} | {h} | {mr:.3f} | {low} |".format(
                ch=ch, T=("%.1f" % Tk) if Tk is not None else "-", d=dirn,
                h=("%.0f" % hmax) if hmax is not None else "-", mr=mr, low=low))
        lines.append("")
    if caps:
        lines += ["## Capabilities", "", "| analysis | applicable | reason |", "|---|---|---|"]
        for c in caps:
            lines.append(f"| {c['name']} | {c['applicable']} | {c['reason']} |")
    dj, dlines = _diagnostics_block(result)
    j["diagnostics"] = dj
    if dlines:
        lines += dlines
    return {"json": j, "markdown": "\n".join(lines)}


def _hall_report(result) -> dict:
    d = result.data
    pts = d.get("points", []); caps = d.get("capabilities", [])
    j = {"probe": "hall", "status": result.status, "confidence": result.confidence,
         "hall_channel": d.get("hall_channel"), "longitudinal_source": d.get("longitudinal_source"),
         "points": pts, "capabilities": caps}
    lines = ["# CryoSweep Analysis Report", "",
             "- **Probe:** hall", f"- **Status:** {result.status}",
             f"- **Confidence:** {result.confidence:.3f}",
             f"- **Hall channel:** {d.get('hall_channel')}",
             f"- **Longitudinal source:** {d.get('longitudinal_source')}", ""]
    if pts:
        lines += ["## Hall vs temperature", "",
                  "| T (K) | R_H (m^3/C) | antisym | carrier | n (1/m^3) | mu (m^2/Vs) | r^2 |",
                  "|---|---|---|---|---|---|---|"]
        for p in pts:
            def g(x, fmt="%.4g"):
                return (fmt % x) if isinstance(x, (int, float)) else "-"
            lines.append("| {T} | {rh} | {a} | {ct} | {n} | {mu} | {r2} |".format(
                T=g(p.get("temperature"), "%.1f"), rh=g(p.get("R_H")), a=p.get("antisymmetrized"),
                ct=p.get("carrier_type") or "-", n=g(p.get("carrier_n")), mu=g(p.get("mobility")),
                r2=g(p.get("r2"), "%.4f")))
        lines.append("")
    if caps:
        lines += ["## Capabilities", "", "| analysis | applicable | reason |", "|---|---|---|"]
        for c in caps:
            lines.append(f"| {c['name']} | {c['applicable']} | {c['reason']} |")
    dj, dlines = _diagnostics_block(result)
    j["diagnostics"] = dj
    if dlines:
        lines += dlines
    return {"json": j, "markdown": "\n".join(lines)}


def _heatcapacity_report(result) -> dict:
    d = result.data
    lines = ["# CryoSweep Heat Capacity Report", "",
             f"- **Status:** {result.status}", f"- **Confidence:** {result.confidence:.3f}", "",
             "## Low-T fits", "", "| model | R² | θ_D (K) |", "|---|---|---|"]
    chosen = d.get("model")
    for mf in d.get("lowt_fits", []):
        star = " ⭐" if mf.get("key") == chosen else ""
        td = mf.get("theta_D")
        td_s = f"{td:.1f}" if isinstance(td, (int, float)) and np.isfinite(td) else "n/a"
        r2 = mf.get("r2")
        lines.append(f"| {mf.get('label', mf.get('key'))}{star} | "
                     f"{r2:.4f} | {td_s} |" if r2 is not None else
                     f"| {mf.get('label')}{star} | n/a | {td_s} |")
    ff = d.get("full_fit") or {}
    lines += ["", "## Full-range Debye-Einstein fit", ""]
    if ff.get("ok"):
        lines += ["| param | value | fixed |", "|---|---|---|"]
        for k, v in (ff.get("params") or {}).items():
            fx = "yes" if (ff.get("fixed") or {}).get(k) else ""
            lines.append(f"| {k} | {v:.4g} | {fx} |")
        lines += ["", f"R² = {ff.get('r2'):.5f}, n = {ff.get('n_points')}"]
    else:
        lines.append(f"_not available: {d.get('full_fit_reason') or (ff.get('reason') if ff else '')}_")
    comp = d.get("comparison") or {}
    lines += ["", "## Comparison (low-T vs full-range)", "", "| param | low-T | full-range |", "|---|---|---|"]
    for k in ("gamma", "theta_D", "r2"):
        row = comp.get(k, {})
        def fmt(x): return f"{x:.4g}" if isinstance(x, (int, float)) and np.isfinite(x) else str(x)
        lines.append(f"| {k} | {fmt(row.get('lowt'))} | {fmt(row.get('full'))} |")
    fg = d.get("field_groups", [])
    if len(fg) >= 2:
        lines += ["", "## Field dependence", "",
                  "| field (Oe) | sel. model | γ | θ_D (K) | flags |", "|---|---|---|---|---|"]
        for g in fg:
            if g["status"] != "ok":
                lines.append(f"| {g['field_oe']:.4g} | _insufficient_ | | | |"); continue
            sel = g.get("chosen_aicc_key")
            f = next((x for x in g["fits"] if x["key"] == sel), None) or {}
            gp = (f.get("params") or {})
            gam = gp.get("gamma"); td = gp.get("theta_D")
            gam_s = f"{gam:.4g}" if isinstance(gam, (int, float)) else "n/a"
            td_s = f"{td:.1f}" if isinstance(td, (int, float)) and np.isfinite(td) else "n/a"
            flags = "; ".join(g.get("warnings", []))
            lines.append(f"| {g['field_oe']:.4g} | {sel} | {gam_s} | {td_s} | {flags} |")
    if d.get("schottky_enabled") and any(g.get("schottky", {}).get("attempted")
                                         for g in d.get("field_groups", [])):
        lines += ["", "## Schottky (opt-in)", "",
                  "| field (Oe) | model | Δ (K) | ±σ | f | determined? | flags |",
                  "|---|---|---|---|---|---|---|"]
        for g in d.get("field_groups", []):
            sc = g.get("schottky")
            if not (g.get("status") == "ok" and sc and sc.get("attempted")):
                continue
            p = sc["params"]; sig = sc.get("sigma", {})
            D = p.get("Delta"); sD = sig.get("Delta")
            D_s = f"{D:.3g}" if isinstance(D, (int, float)) else "n/a"
            sD_s = f"{sD:.2g}" if isinstance(sD, (int, float)) else ""
            f_s = f"{p.get('f'):.3g}" if isinstance(p.get("f"), (int, float)) else "n/a"
            flags = "; ".join(sc.get("warnings", [])) + (f" {sc['reason']}" if sc.get("reason") else "")
            lines.append(f"| {g['field_oe']:.4g} | {sc['chosen_key']} | {D_s} | {sD_s} | {f_s} | "
                         f"{sc.get('delta_determined')} | {flags.strip()} |")
        ov = d.get("schottky_overlay")
        if ov and ov.get("ok"):
            lines += ["", f"**Δ(H) overlay ({ov['model']}):** g = {ov['g_factor']:.3g}"
                      + (f", Δ₀ = {ov['Delta0']:.3g} K" if ov['model'] == 'zfs' else "")
                      + f", r² = {ov['r2']:.3g}"]
    if d.get("transitions_enabled") and any(g.get("transition", {}).get("attempted")
                                            for g in d.get("field_groups", [])):
        lines += ["", "## Transitions (opt-in)", "",
                  "| field (Oe) | form/class | T_c (K) | ±σ | ΔAICc | determined? | notes |",
                  "|---|---|---|---|---|---|---|"]
        for g in d.get("field_groups", []):
            trd = g.get("transition")
            if not (g.get("status") == "ok" and trd and trd.get("attempted")):
                continue
            Tc = trd.get("Tc"); sTc = trd.get("Tc_sigma"); dA = trd.get("delta_aicc")
            Tc_s = f"{Tc:.3g}" if isinstance(Tc, (int, float)) else "n/a"
            sTc_s = f"{sTc:.2g}" if isinstance(sTc, (int, float)) else ""
            dA_s = f"{dA:.2f}" if isinstance(dA, (int, float)) else ""
            label = "Cp-peak" if trd.get("form") == "lambda" else "entropy-mid"
            cls = f"{trd.get('form')}/{trd.get('universality')}"
            notes = "; ".join(trd.get("advisories", []))
            cmp = trd.get("compare")
            if cmp:
                notes = (notes + f"; compare: {cmp.get('verdict')}").strip("; ")
            lines.append(f"| {g['field_oe']:.4g} | {cls} ({label}) | {Tc_s} | {sTc_s} | {dA_s} | "
                         f"{trd.get('tc_determined')} | {notes} |")
    j = {"probe": "heatcapacity", "status": result.status, "confidence": result.confidence,
         "comparison": comp, "warnings": result.warnings}
    dj, dlines = _diagnostics_block(result)
    j["diagnostics"] = dj
    if dlines:
        lines += dlines
    return {"json": j, "markdown": "\n".join(lines)}
