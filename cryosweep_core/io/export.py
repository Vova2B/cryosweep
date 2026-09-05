from __future__ import annotations
import csv, json, math, numbers, pathlib
from cryosweep_core.fitting.transport import POWER_LAW_DECLINE_FLAGS, ARRHENIUS_DECLINE_FLAGS


def _export_hall(result, stem) -> dict:
    d = result.data
    out = {}
    pts = d.get("points", [])
    pp = stem.with_suffix(".points.csv")
    # #20: `derived_flags` appended after the existing columns (name-keyed safe) — the
    # ";".join encoding matches power_law_flags/kappa_ph_flags.
    fields = ["temperature (K)", "R_H (m^3/C)", "R_H_raw (m^3/C)", "slope (Ohm/T)", "r2",
              "antisymmetrized", "carrier_n (1/m^3)", "carrier_type", "rho_xx (Ohm*m)",
              "mobility (m^2/Vs)", "derived_flags"]
    with pp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for p in pts:
            w.writerow({"temperature (K)": p.get("temperature"), "R_H (m^3/C)": p.get("R_H"),
                        "R_H_raw (m^3/C)": p.get("R_H_raw"), "slope (Ohm/T)": p.get("slope_ohm_per_T"),
                        "r2": p.get("r2"), "antisymmetrized": p.get("antisymmetrized"),
                        "carrier_n (1/m^3)": p.get("carrier_n"), "carrier_type": p.get("carrier_type"),
                        "rho_xx (Ohm*m)": p.get("rho_xx"), "mobility (m^2/Vs)": p.get("mobility"),
                        "derived_flags": ";".join(p.get("derived_flags") or [])})
    out["points"] = str(pp)
    cap = stem.with_suffix(".capabilities.csv")
    with cap.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "applicable", "reason"])
        for c in d.get("capabilities", []):
            w.writerow([c["name"], c["applicable"], c["reason"]])
    out["capabilities"] = str(cap)
    meta = {"source": result.provenance.file, "sha256": result.provenance.sha256,
            "hall_channel": d.get("hall_channel"), "thickness_m": d.get("thickness_m"),
            "longitudinal_source": d.get("longitudinal_source"), "config": result.provenance.config}
    mp = stem.with_suffix(".meta.json"); mp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    out["meta"] = str(mp)
    return out


def _export_hall_tdep(result, stem) -> dict:
    """Per-point CSV for the temp-dep Hall reconstruction. Before 2026-09-05 (KNOWN-ISSUES
    21) hall_tdep fell through to the probe-generic exporter, whose point-column scan finds
    no top-level numeric lists on this schema — the CSV was an EMPTY shell. New file, no
    existing readers to break; columns name-keyed like every other probe."""
    d = result.data
    out = {}
    pts = d.get("points", [])
    pp = stem.with_suffix(".points.csv")
    fields = ["temperature (K)", "R_H (m^3/C)", "r_h_method", "r2", "antisym_points",
              "carrier_n (1/m^3)", "carrier_type", "rho_xx (Ohm*m)", "mobility (m^2/Vs)",
              "low_confidence", "excitation (uA)", "current_density_J (A/m^2)"]
    with pp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for p_ in pts:
            w.writerow({"temperature (K)": p_.get("temperature"),
                        "R_H (m^3/C)": p_.get("R_H"),
                        "r_h_method": p_.get("r_h_method"),
                        "r2": p_.get("r2"),
                        "antisym_points": p_.get("antisym_points"),
                        "carrier_n (1/m^3)": p_.get("carrier_n"),
                        "carrier_type": p_.get("carrier_type"),
                        "rho_xx (Ohm*m)": p_.get("rho_xx"),
                        "mobility (m^2/Vs)": p_.get("mobility"),
                        "low_confidence": p_.get("low_confidence"),
                        "excitation (uA)": p_.get("excitation_uA"),
                        "current_density_J (A/m^2)": p_.get("current_density_J")})
    out["points"] = str(pp)
    cap = stem.with_suffix(".capabilities.csv")
    with cap.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "applicable", "reason"])
        for c in d.get("capabilities", []):
            w.writerow([c["name"], c["applicable"], c["reason"]])
    out["capabilities"] = str(cap)
    meta = {"source": result.provenance.file, "sha256": result.provenance.sha256,
            "hall_channel": d.get("hall_channel"), "thickness_m": d.get("thickness_m"),
            "sample_width_m": d.get("sample_width_m"),
            "longitudinal_source": d.get("longitudinal_source"),
            "config": result.provenance.config}
    mp = stem.with_suffix(".meta.json"); mp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    out["meta"] = str(mp)
    return out


def _export_resistivity(result, stem) -> dict:
    d = result.data
    out = {}
    # 1) curves: tidy long format, one row per physical point
    cp = stem.with_suffix(".curves.csv")
    fields = ["bridge", "curve_type", "held_field_oe", "held_temp_k",
              "direction", "x", "x_unit", "rho_ohm_cm"]
    with cp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for b in d.get("bridges", []):
            ch = b["channel"]
            for c in b.get("rho_t_curves", []):
                for x, y in zip(c["temperature"], c["rho"]):
                    w.writerow({"bridge": ch, "curve_type": "rho_T",
                                "held_field_oe": c.get("held_field_oe"), "held_temp_k": "",
                                "direction": c.get("direction"), "x": x, "x_unit": "K",
                                "rho_ohm_cm": y})
            for c in b.get("rho_h_curves", []):
                for x, y in zip(c["field"], c["rho"]):
                    w.writerow({"bridge": ch, "curve_type": "rho_H",
                                "held_field_oe": "", "held_temp_k": c.get("held_temp_k"),
                                "direction": c.get("direction"), "x": x, "x_unit": "Oe",
                                "rho_ohm_cm": y})
    out["curves"] = str(cp)
    # 2) derived: per-bridge scalars
    dp = stem.with_suffix(".derived.csv")
    with dp.open("w", newline="") as f:
        w = csv.writer(f)
        # F1 (final-review): the four honesty columns are APPENDED last, so the original 13
        # keep their names and order (name-keyed readers unaffected; positional readers see
        # 13 -> 17, the same I11 disclosure as the TTO summary CSV's 9 -> 20 growth).
        # Before this, .derived.csv wrote `power_law_n = 0.649` and `rrr = 18.52` BARE while
        # the GUI, for the identical file, said "WINDOW-SENSITIVE: n(15->30 K) = 3.04->0.649".
        # The CSV is the surface the owner publishes from — the number this slice exists to
        # stop being published alone was still leaving through it. `power_law_flags` uses the
        # same ";".join encoding as `kappa_ph_flags` (';' needs no escaping under a ','
        # delimiter — it round-trips through any RFC-4180 reader).
        w.writerow(["channel", "rho_source", "classification", "rrr",
                    "rrr_t_high", "rrr_t_low", "residual_rho_ohm_cm",
                    "power_law_n", "power_law_A", "power_law_r2",
                    "tc_onset_k", "tc_mid_k", "tc_zero_k",
                    "rrr_std", "power_law_n_sigma", "power_law_n_spread",
                    "power_law_flags",
                    # 2026-09-05 activated transport: 17 -> 23, appended last (name-keyed
                    # safe). The gap column carries its assumption in its NAME - E_g = 2*E_a
                    # only if intrinsic; declined fits leave every numeric cell blank.
                    "arrhenius_ea_mev", "arrhenius_ea_sigma_mev",
                    "arrhenius_ea_spread_mev", "arrhenius_r2",
                    "e_g_assuming_intrinsic_mev", "arrhenius_flags"])
        for b in d.get("bridges", []):
            pl = b.get("power_law") or {}
            params = pl.get("params", {})
            # Widest rho(T) ramp carrying a Tc (own copy of the selection logic; export
            # must not depend on the matplotlib-importing render layer — deliberate duplication).
            tc = next((c for c in sorted(b.get("rho_t_curves", []),
                                         key=lambda c: -(c.get("n_points") or 0))
                       if c.get("tc_mid_k") is not None), {})
            pl_flags = ";".join(pl.get("quality_flags") or []) if pl else None
            # Declined fit -> the numeric cells are BLANK and `power_law_flags` carries the
            # reason. The CSV is the surface the owner publishes from, so a search bound or
            # an unresolved exponent must not leave here as a bare number (same rule as the
            # TTO kappa_ph cells).
            if set(pl.get("quality_flags") or []) & POWER_LAW_DECLINE_FLAGS:
                pl, params = {"quality_flags": pl.get("quality_flags")}, {}
            ar = b.get("arrhenius") or {}
            ar_flags = ";".join(ar.get("quality_flags") or []) if ar else None
            ar_params = ar.get("params", {})
            if set(ar.get("quality_flags") or []) & ARRHENIUS_DECLINE_FLAGS:
                ar, ar_params = {"quality_flags": ar.get("quality_flags")}, {}
            w.writerow([b["channel"], b["rho_source"], b["classification"], b.get("rrr"),
                        b.get("rrr_t_high"), b.get("rrr_t_low"), b.get("residual_rho"),
                        params.get("n"), params.get("A"), pl.get("r2"),
                        tc.get("tc_onset_k"), tc.get("tc_mid_k"), tc.get("tc_zero_k"),
                        b.get("rrr_std"), (pl.get("sigma") or {}).get("n"),
                        b.get("power_law_n_spread"), pl_flags,
                        ar_params.get("e_a_mev"), (ar.get("sigma") or {}).get("e_a_mev"),
                        (b.get("arrhenius_ea_spread_mev") if ar_params else None),
                        ar.get("r2"), ar_params.get("e_g_assuming_intrinsic_mev"),
                        ar_flags])
    out["derived"] = str(dp)
    # 3) capabilities
    cap = stem.with_suffix(".capabilities.csv")
    with cap.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "applicable", "reason"])
        for c in d.get("capabilities", []):
            w.writerow([c["name"], c["applicable"], c["reason"]])
    out["capabilities"] = str(cap)
    # 4) sidecar
    meta = {"source": result.provenance.file, "sha256": result.provenance.sha256,
            "app_version": result.provenance.app_version,
            "rho_source": d.get("rho_source", ""), "config": result.provenance.config}
    mp = stem.with_suffix(".meta.json"); mp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    out["meta"] = str(mp)
    # 5) magnetoresistance: one row per grouped field loop
    mrp = stem.with_suffix(".mr_percent.csv")
    with mrp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["channel", "held_temp_k", "max_abs_field_oe", "direction",
                    "mr_percent_at_max_field", "low_confidence"])
        for b in d.get("bridges", []):
            for c in b.get("rho_h_curves", []):
                if c.get("mr_percent_at_max_field") is None:
                    continue
                w.writerow([b["channel"], c.get("held_temp_k"), c.get("max_abs_field_oe"),
                            c.get("direction"), c.get("mr_percent_at_max_field"),
                            c.get("low_confidence")])
    out["mr_percent"] = str(mrp)
    return out


def _export_heatcapacity(result, stem) -> dict:
    d = result.data; out = {}
    # 1) fit params: one row per low-T model + the full-range fit
    fp = stem.with_suffix(".fit_params.csv")
    with fp.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fit", "model", "param", "value", "r2"])
        for mf in d.get("lowt_fits", []):
            for k, v in (mf.get("params") or {}).items():
                w.writerow(["lowt", mf.get("key"), k, v, mf.get("r2")])
        ff = d.get("full_fit") or {}
        if ff.get("ok"):
            for k, v in (ff.get("params") or {}).items():
                fx = "fixed" if (ff.get("fixed") or {}).get(k) else "free"
                w.writerow(["full", "debye_einstein", f"{k} ({fx})", v, ff.get("r2")])
    out["fit_params"] = str(fp)
    # 2) comparison
    cp_ = stem.with_suffix(".comparison.csv")
    comp = d.get("comparison") or {}
    with cp_.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["param", "lowt", "full", "unit"])
        units = {"gamma": "J/(mol*K^2)", "theta_D": "K", "r2": ""}
        for k in ("gamma", "theta_D", "r2"):
            row = comp.get(k, {})
            w.writerow([k, row.get("lowt"), row.get("full"), units[k]])
    out["comparison"] = str(cp_)
    # 3) model curves (tidy long)
    mc = stem.with_suffix(".model_curves.csv")
    with mc.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "x", "y"])
        for mf in d.get("lowt_fits", []):
            xs = mf.get("t2_grid") or []; ys = mf.get("cp_over_t_fit") or []
            for x, y in zip(xs, ys):
                w.writerow([f'lowt:{mf.get("key")}', x, y])
        ff = d.get("full_fit") or {}
        if ff.get("ok"):
            for x, y in zip(ff.get("t_grid") or [], ff.get("cp_fit") or []):
                w.writerow(["full:debye_einstein", x, y])
    out["model_curves"] = str(mc)
    # 4) raw points — the FULL measured group (the complete dataset behind the Cp-vs-T plot),
    #    not the low-T subset. Falls back to the low-T arrays if full_* is absent.
    pts = stem.with_suffix(".points.csv")
    Tcol = d.get("full_temperature") or d.get("temperature", [])
    Ccol = d.get("full_cp") or d.get("cp", [])
    with pts.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["temperature (K)", "cp (J/(mol*K))"])
        for t, c in zip(Tcol, Ccol):
            w.writerow([t, c])
    out["points"] = str(pts)
    # 4b) entropy S(T) — own ragged block aligned to entropy_temperature. S_magnetic
    #     column is present only when a magnetic array exists; None/non-finite entries
    #     (out-of-overlap truncation) become BLANK cells, never "nan"/"None"/"inf".
    if d.get("entropy_available"):
        Tent = d.get("entropy_temperature") or []
        Stot = d.get("entropy_total") or []
        Smag = d.get("entropy_magnetic")
        has_mag = Smag is not None
        ep = stem.with_suffix(".entropy.csv")

        def _cell(v):
            return f"{v:.6g}" if isinstance(v, (int, float)) and math.isfinite(v) else ""

        # Rln match-verdict columns appended AFTER the existing ones (2026-08-10 spec §7):
        # file-level scalars repeated on every row (tidy-long convention), so any row read in
        # isolation carries the verdict. Name-keyed readers safe; exact-width readers break
        # (same I11 disclosure as the TTO summary CSV).
        sug = d.get("entropy_rln_suggestion") or {}
        has_verdict = "matched" in sug
        with ep.open("w", newline="") as f:
            w = csv.writer(f)
            hdr = ["T", "S_total", "S_magnetic"] if has_mag else ["T", "S_total"]
            if has_verdict:
                hdr += ["rln_label", "rln_matched", "rln_rel_err"]
            w.writerow(hdr)
            for i in range(len(Tent)):
                tot = Stot[i] if i < len(Stot) else None
                row = [_cell(Tent[i]), _cell(tot)]
                if has_mag:
                    row.append(_cell(Smag[i] if i < len(Smag) else None))
                if has_verdict:
                    row += [sug.get("label", ""), str(bool(sug.get("matched"))),
                            _cell(sug.get("rel_err"))]
                w.writerow(row)
        out["entropy"] = str(ep)
    # 5) meta
    meta = {"source": result.provenance.file, "sha256": result.provenance.sha256,
            "status": result.status, "confidence": result.confidence,
            "warnings": result.warnings, "comparison": comp,
            "full_fit_available": d.get("full_fit_available"),
            "config": result.provenance.config}
    mp = stem.with_suffix(".meta.json"); mp.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str))
    out["meta"] = str(mp)
    # 6) field-dependence CSV (only when >= 2 field groups)
    fg = d.get("field_groups", [])
    if len(fg) >= 2:
        fdp = stem.with_suffix(".field_dependence.csv")
        with fdp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["field_oe", "model", "param", "value", "sigma", "identifiable",
                        "railed", "aicc", "bic", "r2", "n_lowt", "is_primary"])
            for g in fg:
                if g["status"] != "ok":
                    continue
                for f in g["fits"]:
                    if not f.get("ok"):
                        continue
                    ident = f.get("identifiability", {})
                    for pname, val in f["params"].items():
                        if pname == "theta_D" and f["key"] not in ("debye_t3", "debye_t3_t5"):
                            continue
                        if not isinstance(val, (int, float)) or not math.isfinite(val):
                            continue
                        sig = f.get("sigma", {}).get(pname)
                        pid = ident.get(pname, {})
                        w.writerow([f"{g['field_oe']:.4g}", f["key"], pname, f"{val:.6g}",
                                    ("" if sig is None else f"{sig:.6g}"),
                                    f.get("identifiable", ""), pid.get("railed", ""),
                                    ("" if f.get("aicc") is None else f"{f['aicc']:.6g}"),
                                    f"{f.get('bic', ''):.6g}" if f.get("bic") is not None else "",
                                    f"{f.get('r2', ''):.6g}" if f.get("r2") is not None else "",
                                    g["n_lowt"], g["is_primary"]])
        out["field_dependence"] = str(fdp)
    if d.get("schottky_enabled") and any(g.get("schottky", {}).get("attempted")
                                         for g in fg):
        sp = stem.with_suffix(".schottky.csv")
        with sp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["field_oe", "chosen_model", "param", "value", "sigma",
                        "delta_determined", "peak_covered", "railed", "aicc", "r2",
                        "n_lowt", "is_primary"])
            for g in fg:
                sc = g.get("schottky")
                if not (g.get("status") == "ok" and sc and sc.get("attempted")):
                    continue
                ck = sc["chosen_key"]; aicc = (sc.get("aicc") or {}).get(ck)
                for pname, val in sc["params"].items():
                    if not isinstance(val, (int, float)) or not math.isfinite(val):
                        continue
                    sig = sc.get("sigma", {}).get(pname)
                    railed = sc.get("identifiability", {}).get(pname, {}).get("railed", "")  # I2
                    w.writerow([f"{g['field_oe']:.4g}", ck, pname, f"{val:.6g}",
                                ("" if sig is None else f"{sig:.6g}"),
                                sc.get("delta_determined"), sc.get("peak_covered"), railed,
                                ("" if aicc is None else f"{aicc:.6g}"),
                                ("" if sc.get("r2") is None else f"{sc['r2']:.6g}"),
                                g.get("n_lowt", ""), g.get("is_primary", "")])
            ov = d.get("schottky_overlay")
            if ov and ov.get("ok"):
                w.writerow([])
                w.writerow(["overlay_model", ov["model"], "g_factor", f"{ov['g_factor']:.6g}",
                            ("" if ov.get("Delta0") is None else f"{ov['Delta0']:.6g}"),
                            "", "", "", "", ("" if ov.get("r2") is None else f"{ov['r2']:.6g}"), "", ""])
        out["schottky"] = str(sp)
    if d.get("transitions_enabled") and any(g.get("transition", {}).get("attempted") for g in fg):
        tp = stem.with_suffix(".transitions.csv")
        with tp.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["field_oe", "form", "universality", "Tc", "Tc_sigma", "tc_determined",
                        "delta_aicc", "param", "value", "sigma", "advisories"])
            for g in fg:
                trd = g.get("transition")
                if not (g.get("status") == "ok" and trd and trd.get("attempted")):
                    continue
                adv = "; ".join(trd.get("advisories", []))
                base = [f"{g['field_oe']:.4g}", trd.get("form"), trd.get("universality"),
                        ("" if trd.get("Tc") is None else f"{trd['Tc']:.6g}"),
                        ("" if trd.get("Tc_sigma") is None else f"{trd['Tc_sigma']:.6g}"),
                        trd.get("tc_determined"),
                        ("" if trd.get("delta_aicc") is None else f"{trd['delta_aicc']:.6g}")]
                for pname, val in (trd.get("params") or {}).items():
                    if not isinstance(val, (int, float)) or not math.isfinite(val):
                        continue
                    sig = (trd.get("sigmas") or {}).get(pname)
                    w.writerow(base + [pname, f"{val:.6g}",
                                       ("" if sig is None else f"{sig:.6g}"), adv])
        out["transitions"] = str(tp)
    return out


def _export_acms(result, stem) -> dict:
    """ACMS export: long-format <stem>.chi.csv (one row per (curve, T) point) and a
    pinned-header <stem>.features.csv (SC transitions + chi'' peaks). Molar and M-DC
    columns are appended only when any curve carries them. Non-finite values are written
    as empty cells (core stays csv/stdlib/numpy-only; never emit NaN/inf)."""
    d = result.data
    out = {}
    curves = d.get("curves") or []
    molar = any(c.get("chi_prime_molar") for c in curves)
    has_mdc = any(c.get("m_dc") for c in curves)

    def _cell(v):
        return "" if isinstance(v, float) and not math.isfinite(v) else v

    cp = stem.with_suffix(".chi.csv")
    fields = ["frequency_hz", "amplitude_oe", "field_oe", "direction", "T", "chi_prime", "chi_dprime"]
    if molar:
        fields += ["chi_prime_molar", "chi_dprime_molar"]
    if has_mdc:
        fields += ["m_dc"]
    with cp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for c in curves:
            n = len(c.get("t") or [])
            for i in range(n):
                row = {"frequency_hz": c["frequency_hz"], "amplitude_oe": c["amplitude_oe"],
                       "field_oe": c["field_oe"], "direction": c["direction"],
                       "T": _cell(c["t"][i]), "chi_prime": _cell(c["chi_prime"][i]),
                       "chi_dprime": _cell(c["chi_dprime"][i])}
                if molar and c.get("chi_prime_molar"):
                    row["chi_prime_molar"] = _cell(c["chi_prime_molar"][i])
                    row["chi_dprime_molar"] = _cell(c["chi_dprime_molar"][i])
                if has_mdc and c.get("m_dc"):
                    row["m_dc"] = _cell(c["m_dc"][i])
                w.writerow(row)
    out["chi"] = str(cp)

    fp = stem.with_suffix(".features.csv")
    fh = ["feature_type", "frequency_hz", "amplitude_oe", "field_oe", "direction",
          "tc_onset_k", "tc_mid_k", "drop_emu_per_oe", "chi_dprime_peak_t_k",
          "t_f_k", "prominence", "low_confidence", "reasons"]
    with fp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fh, extrasaction="ignore"); w.writeheader()
        for c in curves:
            base = {"frequency_hz": c["frequency_hz"], "amplitude_oe": c["amplitude_oe"],
                    "field_oe": c["field_oe"], "direction": c["direction"]}
            sc = c.get("sc")
            if sc:
                w.writerow({**base, "feature_type": "sc_transition",
                            "tc_onset_k": _cell(sc["tc_onset_k"]), "tc_mid_k": _cell(sc["tc_mid_k"]),
                            "drop_emu_per_oe": _cell(sc["drop_emu_per_oe"]),
                            "chi_dprime_peak_t_k": _cell(sc["chi_dprime_peak_t_k"]),
                            "low_confidence": sc["low_confidence"], "reasons": "; ".join(sc["reasons"])})
            pk = c.get("peak")
            if pk:
                w.writerow({**base, "feature_type": "chi_dprime_peak", "t_f_k": _cell(pk["t_f_k"]),
                            "prominence": _cell(pk["prominence"]), "low_confidence": pk["low_confidence"],
                            "reasons": "; ".join(pk["reasons"])})
    out["features"] = str(fp)
    return out


def _export_tto(result, stem) -> dict:
    """TTO export: long-format <stem>.tto.csv (one row per (curve, point), pinned 18-column
    header since the 2026-08-10 uncertainty slice, O6) and a one-row <stem>.tto_summary.csv (20 columns since the integrity slice; the
    original 9 keep their names and order). Non-finite/missing values are written as
    empty cells (never NaN/inf).

    FILENAMES use stem.with_name(stem.name + ...), NOT with_suffix: the real gate file's stem
    `sample.a1_export` contains a dot and with_suffix('.tto.csv') truncates it to
    `sample.tto.csv` (measured). Following the acms precedent, no .capabilities.csv and no
    .meta.json sidecar are written. The *_std columns are exported even though error bars are
    not rendered (D12)."""
    d = result.data
    out = {}

    def _cell(v):
        # numbers.Real, NOT float: np.float32("nan") is NOT a float subclass (only np.float64
        # is), so an `isinstance(v, float)` test lets a numpy NaN through as a literal "nan"
        # cell. Not reachable today -- the analyzer wraps every fit scalar in float(...) -- but
        # the never-emit-non-finite constraint is absolute, so the test is made on the numeric
        # TOWER instead. (bool is Real and always finite -> unchanged.)
        if v is None:
            return ""
        return "" if isinstance(v, numbers.Real) and not math.isfinite(v) else v

    def _at(arr, i):
        # Length-tolerant: a short optional array blanks rather than IndexError-ing.
        return None if not arr or i >= len(arr) else arr[i]

    # Closed O6 (2026-08-10): the three derived _std columns are APPENDED (15 -> 18) —
    # name-keyed readers unaffected, positional readers break (same precedent as the
    # summary CSV 9 -> 20).
    fields = ["field_oe", "direction", "T", "kappa", "kappa_std", "seebeck", "seebeck_std",
              "rho_ohm_m", "rho_std", "zt", "zt_std", "kappa_e", "kappa_ph",
              "lorenz_ratio", "power_factor", "kappa_e_std", "kappa_ph_std",
              "lorenz_ratio_std"]
    lp = stem.with_name(stem.name + ".tto.csv")
    with lp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in d.get("curves") or []:
            for i in range(len(c.get("t") or [])):
                w.writerow({
                    "field_oe": _cell(c["field_oe"]), "direction": _cell(c["direction"]),
                    "T": _cell(c["t"][i]), "kappa": _cell(c["kappa"][i]),
                    "kappa_std": _cell(_at(c.get("kappa_std"), i)),
                    "seebeck": _cell(_at(c.get("seebeck"), i)),
                    "seebeck_std": _cell(_at(c.get("seebeck_std"), i)),
                    "rho_ohm_m": _cell(_at(c.get("rho"), i)),
                    "rho_std": _cell(_at(c.get("rho_std"), i)),
                    "zt": _cell(_at(c.get("zt"), i)),
                    "zt_std": _cell(_at(c.get("zt_std"), i)),
                    "kappa_e": _cell(_at(c.get("kappa_e"), i)),
                    "kappa_ph": _cell(_at(c.get("kappa_ph"), i)),
                    "lorenz_ratio": _cell(_at(c.get("lorenz_ratio"), i)),
                    "power_factor": _cell(_at(c.get("power_factor"), i)),
                    "kappa_e_std": _cell(_at(c.get("kappa_e_std"), i)),
                    "kappa_ph_std": _cell(_at(c.get("kappa_ph_std"), i)),
                    "lorenz_ratio_std": _cell(_at(c.get("lorenz_ratio_std"), i))})
    out["tto"] = str(lp)

    # I11: 9 -> 20 columns. The original nine keep their NAMES and ORDER, so name-keyed
    # readers (pandas read_csv, csv.DictReader) are unaffected; POSITIONAL readers (Origin
    # import templates, the owner's own scripts) break. No back-compat shim is provided.
    sh = ["rrr", "rrr_t_high_k", "rrr_t_low_k", "classification", "pf_at_thigh_w_k2m",
          "zt_peak", "zt_peak_t_k", "zt_peak_at_edge", "n_error_rows",
          "rrr_std", "zt_peak_std",
          "kappa_ph_n", "kappa_ph_n_sigma", "kappa_ph_n_spread", "kappa_ph_n_loglog",
          "kappa_ph_n_method_delta", "kappa_ph_b", "kappa_ph_r2", "kappa_ph_window_k_max",
          "kappa_ph_flags"]
    rrr = d.get("rrr") or {}
    summary = d.get("summary") or {}
    kf = d.get("kappa_ph_fit") or {}
    # A ';' inside kappa_ph_flags needs no escaping: DictWriter's delimiter is ',' and its
    # quotechar is '"', so it is written literally and round-trips through any RFC-4180 reader.
    flags = ";".join(kf.get("quality_flags") or []) if kf else None
    window = (kf.get("window_k") or [None, None])[1] if kf else None
    sp = stem.with_name(stem.name + ".tto_summary.csv")
    with sp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sh)
        w.writeheader()
        w.writerow({"rrr": _cell(rrr.get("rrr")),
                    "rrr_t_high_k": _cell(rrr.get("t_high_k")),
                    "rrr_t_low_k": _cell(rrr.get("t_low_k")),
                    "classification": _cell(rrr.get("classification")),
                    "pf_at_thigh_w_k2m": _cell(summary.get("pf_at_thigh")),
                    "zt_peak": _cell(summary.get("zt_peak")),
                    "zt_peak_t_k": _cell(summary.get("zt_peak_t_k")),
                    # honesty flag: the max sits at an end of the measured T range
                    "zt_peak_at_edge": _cell(summary.get("zt_peak_at_edge")),
                    "n_error_rows": d.get("n_error_rows", 0),
                    "rrr_std": _cell(rrr.get("rrr_std")),
                    "zt_peak_std": _cell(summary.get("zt_peak_std")),
                    "kappa_ph_n": _cell(kf.get("n")),
                    "kappa_ph_n_sigma": _cell(kf.get("n_sigma")),
                    # blank -- NEVER 0.0 -- when the window spread was never measured (I1)
                    "kappa_ph_n_spread": _cell(kf.get("n_spread")),
                    "kappa_ph_n_loglog": _cell(kf.get("n_loglog")),
                    "kappa_ph_n_method_delta": _cell(kf.get("n_method_delta")),
                    "kappa_ph_b": _cell(kf.get("b")),
                    "kappa_ph_r2": _cell(kf.get("r2")),
                    "kappa_ph_window_k_max": _cell(window),
                    "kappa_ph_flags": _cell(flags)})
    out["tto_summary"] = str(sp)
    return out


def export_result(result, stem, fmt="csv") -> dict:
    stem = pathlib.Path(stem)
    # --out may name a not-yet-existing directory (`--out results/run1`): the user
    # named that path explicitly, so create it (same convention as export_plots).
    stem.parent.mkdir(parents=True, exist_ok=True)
    d = result.data
    if d.get("probe") == "acms":
        return _export_acms(result, stem)
    if d.get("probe") == "tto":
        return _export_tto(result, stem)
    if d.get("probe") == "resistivity":
        return _export_resistivity(result, stem)
    if d.get("probe") == "hall":
        return _export_hall(result, stem)
    if d.get("probe") == "hall_tdep":
        return _export_hall_tdep(result, stem)
    if d.get("probe") == "heatcapacity":
        return _export_heatcapacity(result, stem)
    # Probe-generic point columns: any data key whose value is a list of numbers
    # is a point column; units come from a merged lookup. Works for VSM and HC alike.
    UNIT_HINTS = {"temperature": "K", "field": "Oe", "moment_emu_per_g": "emu/g",
                  "moment_per_fu": "mu_B/f.u.", "chi_molar_cgs": "emu/(mol*Oe)",
                  "chi_molar_si": "m^3/mol", "inv_chi": "mol*Oe/emu",
                  "cp": "J/(mol*K)", "cp_over_t": "J/(mol*K^2)", "t_squared": "K^2"}
    # A point column is a NON-EMPTY list of numbers. Empty lists and lists of structured
    # values (e.g. VSM `loops`/`ramps`) are not point columns — including an empty list
    # here would desync row counts and IndexError against a longer sibling column.
    def _is_numlist(v): return isinstance(v, list) and bool(v) and isinstance(v[0], (int, float))
    point_cols = sorted(k for k, v in d.items() if _is_numlist(v))
    units = {c: UNIT_HINTS.get(c, "") for c in point_cols}
    if "inv_chi" in units and d.get("inv_chi_unit"):   # preserve the unit-system-aware VSM unit (CGS vs SI)
        units["inv_chi"] = d["inv_chi_unit"]
    n = len(d.get(point_cols[0], [])) if point_cols else 0
    rows = [{f"{c} ({units[c]})": d[c][i] for c in point_cols} for i in range(n)]
    out = {}
    if fmt == "json":
        p = stem.with_suffix(".points.json"); p.write_text(json.dumps(rows, indent=2)); out["points"] = str(p)
    else:
        p = stem.with_suffix(".points.csv")
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
            w.writeheader(); w.writerows(rows)
        out["points"] = str(p)
    # fit-params table. CW-ladder rows/columns are appended AFTER the existing four columns
    # (2026-08-10 spec §7): name-keyed readers (DictReader) are safe; readers assuming exactly
    # four columns per file break — same I11 disclosure as the TTO summary CSV 15->18 growth.
    # A file with NO ladder stays byte-identical to before (4-column header, 4-cell rows).
    # F2 (final-review): two more appended columns. `sigma_kind` says WHAT the sigma cell is
    # — a reader was taking `theta`, a column literally named `sigma`, and publishing
    # theta = -50.27 +- 0.99 K, which is the exact failure this slice exists to prevent
    # (0.99 K is fit scatter; the window moves theta by 12.7 K). `flags` carries
    # fit.quality_flags, which previously reached NO column and NO row of this file even
    # though it holds `window_sensitive`. The spread rows are renamed self-describingly and
    # tagged `window_spread` so they can never be read as another fitted parameter with an
    # unknown error bar (U3: "spread != error bar, in EVERY rendering").
    fit = d.get("fit", {})
    ladder = d.get("cw_ladder") or []
    fp = stem.with_suffix(".fit_params.csv")
    fit_flags = ";".join(fit.get("quality_flags") or [])
    with fp.open("w", newline="") as f:
        w = csv.writer(f)
        header = ["param", "value", "sigma", "unit"]
        if ladder:
            header += ["rung_tmin_k", "r2", "n_points", "sigma_kind", "flags"]
        # F9 (final-review): pad every short row to the header width. The regenerated
        # goldens were ragged ({7: 11, 4: 5} fields) — pandas/DictReader cope, but
        # numpy.genfromtxt and fixed-width importers do not.
        def _row(cells):
            w.writerow(list(cells) + [""] * (len(header) - len(cells)))
        w.writerow(header)
        for k, v in fit.get("params", {}).items():
            _row([k, v, fit.get("sigma", {}).get(k, ""), fit.get("units", {}).get(k, ""),
                  "", "", "", "fit_scatter_stat", fit_flags] if ladder else
                 [k, v, fit.get("sigma", {}).get(k, ""), fit.get("units", {}).get(k, "")])
        for e in ladder:
            tag = f"(T>={e['tmin_k']:g}K)"
            _row([f"theta{tag}", e.get("theta_k"), e.get("sigma_theta_k"), "K",
                  e.get("tmin_k"), e.get("r2"), e.get("n_points"), "fit_scatter_stat", ""])
            _row([f"mu_eff{tag}", e.get("mu_eff"), e.get("sigma_mu_eff"), "mu_B",
                  e.get("tmin_k"), e.get("r2"), e.get("n_points"), "fit_scatter_stat", ""])
        if ladder and d.get("theta_spread_k") is not None:
            _row(["theta_window_spread_not_an_error_bar", d["theta_spread_k"], "", "K",
                  "", "", "", "window_spread", fit_flags])
        if ladder and d.get("mu_eff_spread") is not None:
            _row(["mu_eff_window_spread_not_an_error_bar", d["mu_eff_spread"], "", "mu_B",
                  "", "", "", "window_spread", fit_flags])
    out["fit_params"] = str(fp)
    # derived-quantities table (the scalar fit outputs, same content, model-tagged)
    dq = stem.with_suffix(".derived.csv")
    with dq.open("w", newline="") as f:
        w = csv.writer(f); w.writerow(["quantity", "value", "sigma", "unit", "model"])
        for k, v in fit.get("params", {}).items():
            w.writerow([k, v, fit.get("sigma", {}).get(k, ""), fit.get("units", {}).get(k, ""), fit.get("model", "")])
    out["derived"] = str(dq)
    # sidecar
    meta = {"source": result.provenance.file, "sha256": result.provenance.sha256,
            "app_version": result.provenance.app_version,
            "unit_system": result.provenance.config.get("unit_system", "CGS"),
            "units": units, "config": result.provenance.config}
    mp = stem.with_suffix(".meta.json"); mp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    out["meta"] = str(mp)
    return out
