"""Generate the `examples/` folder: one runnable .dat per probe.

Most are SYNTHETIC, produced by the same generators the test-suite fixtures come from
(`tests/core/fixtures/make_*.py`): nothing in them derives from a real measurement, so they are
leak-proof by construction rather than by anonymization.

TWO are anonymized subsets of real measurements (the multi-field VSM and heat-capacity files),
because no synthetic file reproduces what real multi-field data does to the segmentation and
windowing paths. They are decimated, their sample identity, instrument serial numbers and
acquisition date are replaced, and their formula weight and sample mass are neutral values —
see `write_real_example` in the two generators, whose `assert_no_identity_leak` post-condition
refuses to write the file if any token of the source identity survives. They are REAL
measurements of an undisclosed sample: the shapes are real, the sample metadata is not, and no
scientific conclusion should be drawn from them.

They are also *chosen* rather than merely available: each one places a feature where it
exercises the analyzer for that probe (a Curie-Weiss regime, a resistive superconducting
transition, a phonon-dominated kappa, a clean Hall slope), which real data rarely does on cue.
The numbers are physically consistent but they are not measurements of anything, and no
scientific conclusion should be drawn from them.

Run:  python tools/make_examples.py [--out examples]
"""
from __future__ import annotations
import argparse, pathlib, sys

APP = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP))
sys.path.insert(0, str(APP / "tests" / "core" / "fixtures"))

# Every generator is driven through a write-one-file callable. Example-specific kwargs select
# the SHOWCASE variant of a generator (wider T range, complete instrument header, realistic
# noise) — the no-kwargs defaults are pinned to the committed fixture bytes and are what the
# test suite regenerates, so the two outputs can never drift into each other.
# (example name, description, generator module, callable, kwargs)
SPECS = [
    ("magnetization_vsm.dat",
     "VSM M(T): Curie-Weiss paramagnet, C = 0.5 emu*K/(mol*Oe), theta = -10 K, measured at 1 kOe",
     "make_vsm", "write_vsm", {}),
    ("magnetization_mpms.dat",
     "MPMS bare-CSV M(T) - the second magnetometer format the loader recognises. Curie-Weiss "
     "theta = -30 K, C = 3 emu*K/(mol*Oe), mu_eff = 4.90 mu_B, once molar mass and sample "
     "mass are supplied",
     "make_mpms", "write_mpms", {}),
    ("heat_capacity.dat",
     "Cp(T) 2-300 K: electronic + full Debye lattice - low-T Cp/T vs T^2 AND the 7-parameter "
     "Debye-Einstein full-range fit",
     "make_hc", "write_hc", {"example": True}),
    ("thermal_transport.dat",
     "TTO: kappa, Seebeck, rho and ZT vs T; phonon-dominated below ~10 K (kappa_ph ~ T^3)",
     "make_tto", "write_example", {}),
    ("ac_susceptibility.dat",
     "ACMS chi'/chi'' vs T with a diamagnetic screening step (superconducting transition); "
     "header carries MOLWGHT/MASS so molar chi is on out of the box",
     "make_acms", "write_sc_example", {}),
    ("resistivity_superconductor.dat",
     "rho(T) with a resistive superconducting transition (Tc_mid ~ 8 K, RRR ~ 80) - "
     "exercises the Tc detector on a measurement-like curve",
     "make_rho_sc", "write_rho_sc_example", {}),
    ("resistivity_semiconductor.dat",
     "Semiconductor rho(T), 80-300 K: Arrhenius activated transport with a planted "
     "E_a = 60 meV - the fit reports E_a as measured, and E_g = 120 meV ONLY under "
     "the intrinsic-conduction assumption (the factor-of-two trap)",
     "make_rho_semi", "write_rho_semi_example", {}),
    ("hall_field_sweeps.dat",
     "R_xy(H) loops at 10 / 100 / 300 K - Hall antisymmetrization and R_H",
     "make_hall", "write_hall", {"geometry": (2.0, 2.0)}),
    ("hall_temperature_dependence.dat",
     "Temperature-dependent Hall: R_H(T), carrier density and mobility",
     "make_hall_tdep", "write_tdep", {"geometry": (2.0, 2.0)}),
]

# Anonymized real-derived examples. The source files are NOT in the repo (they are resolved
# through the untracked real_data_map.json), so these are OPTIONAL: on a machine without them
# the committed copies are left untouched and the run still succeeds — the same skip-not-fail
# rule the test suite uses for real data.
REAL_SPECS = [
    ("magnetization_vsm_multifield.dat", "vsm",
     "Anonymized real VSM: multi-field M(T) at 100 Oe / 5 kOe / 40 kOe / 100 kOe plus four "
     "M(H) loops - Curie-Weiss theta = -31 K, mu_eff = 3.70 mu_B at r2 = 0.987",
     "make_vsm", "write_real_example", {}),
    ("heat_capacity_multifield.dat", "hc",
     "Anonymized real Cp(T): four fields (0 / 5 / 10 / 13 T) - low-T Cp/T vs T^2 and the "
     "full-range 7-parameter Debye-Einstein fit (theta_D = 119 K) on a real lattice",
     "make_hc", "write_real_example", {}),
    ("hall_mixed_sweeps.dat", "hall",
     "Anonymized real Hall-wired measurement: nine field loops (2-300 K, +-90 kOe) plus "
     "fixed-field temperature ramps - drifting temperature setpoints and single-pair "
     "field coverage, the messiness the synthetic Hall examples cannot express "
     "(KNOWN-ISSUES 18-20 regression data); geometry deliberately unset",
     "make_hall_real", "write_real_example", {}),
]


# Per-file GUI orientation for examples/README.md - what to open, what to expect, which
# inputs to type. hall_field_sweeps.dat needs the longest note: it is resistivity-FORMAT
# (only the wiring differs), so the app correctly opens the Resistivity tab, where a
# field-independent longitudinal channel (flat by construction) reads as low-confidence
# MR - the file's actual content lives in the Hall tab (green dot).
_GUI_NOTES = {
    "magnetization_vsm.dat":
        "Opens in the Magnetization tab. Curie-Weiss fit gives theta = -10 K, C = 0.5.",
    "magnetization_mpms.dat":
        "Opens in the Magnetization tab, status 'gated': bare MPMS CSVs carry no molar mass "
        "or sample mass, so type them in the left panel (200 g/mol, 5 mg) and re-analyze. "
        "This is a *different* synthetic sample from the VSM file, not the same one in "
        "another format - it fits to theta = -30 K, C = 3, mu_eff = 4.90 mu_B (the VSM file "
        "is theta = -10 K, C = 0.5, mu_eff = 2.00). What the two share is the loader path, "
        "not the physics.",
    "heat_capacity.dat":
        "Opens in the Heat Capacity tab. Both the low-T Cp/T vs T^2 fit (gamma = 0.01, "
        "theta_D ~ 227 K) and the full-range 7-parameter Debye-Einstein fit run. The "
        "'S_mag saturation matches no R ln(2J+1)' warning is the analyzer being honest: "
        "this synthetic sample has no magnetic entropy, so no R ln(2J+1) plateau exists.",
    "thermal_transport.dat":
        "Opens in the Thermal Transport tab: kappa/Seebeck/rho/ZT panels plus the kappa_ph "
        "power-law fit (n ~ 3 below 10 K, with its window-sensitivity honesty flag).",
    "ac_susceptibility.dat":
        "Opens in the AC Susceptibility tab: diamagnetic screening step (Tc mid 5.0 K), "
        "chi'' peak, and - because this header carries MOLWGHT/MASS - molar chi out of the box.",
    "resistivity_superconductor.dat":
        "Opens in the Resistivity tab: resistive transition on Ch1 (onset 8.8 / mid 8.0 / "
        "zero 7.5 K), featureless metal on Ch2, RRR ~ 80. The outlier badge counts the "
        "below-Tc points: the superconducting state sits far outside the normal-state "
        "scatter band, which is physics, not bad data.",
    "resistivity_semiconductor.dat":
        "Opens in the Resistivity tab: Ch1 is an insulating (semiconducting) channel and "
        "gets the Arrhenius fit - E_a = 60 meV as measured. The row also shows "
        "'E_g = 2*E_a = 120 meV ONLY IF intrinsic': for extrinsic conduction the factor "
        "is 1, and transport alone cannot tell the regimes apart, so the assumption "
        "travels with the number everywhere it goes. Ch2 is a featureless metal and gets "
        "no Arrhenius row - the fit is gated on the insulating classification.",
    "magnetization_vsm_multifield.dat":
        "Opens in the Magnetization tab. Four held fields give four M(T) curves plus four "
        "M(H) loops. Status is 'low confidence' on purpose: the Curie-Weiss window reaches "
        "below |theta| (2 K < 31 K), so the analyzer says the low-T rows are likely outside "
        "the paramagnetic regime and points at the window ladder - honest reporting on real "
        "data, not an error. Formula weight and sample mass are neutral values, so mu_eff is "
        "plausible but is not a property of any real material.",
    "heat_capacity_multifield.dat":
        "Opens in the Heat Capacity tab. Four field groups (0 / 5 / 10 / 13 T); both the "
        "low-T Cp/T vs T^2 fit and the full-range 7-parameter Debye-Einstein fit run "
        "(theta_D = 119 K, r2 = 0.9999). The analyzed column is already per-mole, so the "
        "neutral formula weight in the header changes no fitted number.",
    "hall_field_sweeps.dat":
        "This is the HALL example, but Hall measurements use the resistivity FILE FORMAT "
        "(only the wiring differs), so the app opens the Resistivity tab first - where the "
        "longitudinal channel is field-independent by construction and MR reads 0% with a "
        "low-confidence note. That is expected; the content is in the *Hall* tab (it has a "
        "green dot): Hall channel 1 is pre-detected, enter thickness 0.5 mm to get "
        "R_H = -2.5e-7 m^3/C, carrier density and mobility. CLI: see the README quickstart.",
    "hall_temperature_dependence.dat":
        "Resistivity-format file for the Temp-Dep Hall tab (green dot): set the Hall "
        "channel (1) and thickness to get R_H(T), n(T) and mobility(T); the Resistivity "
        "tab shows the ordinary rho(T) of the longitudinal channel.",
    "hall_mixed_sweeps.dat":
        "Real (anonymized) Hall data, and deliberately messy where the synthetic files "
        "are clean. Hall tab: channel 1, thickness 0.07 mm, longitudinal channel 2 - "
        "nine R_xy(H) loops incl. a 200 K loop whose setpoint drifted 199.84-199.99 K "
        "(one loop, not two - that split once fabricated a phantom carrier density). "
        "Temp-Dep Hall tab: most temperatures carry a single +- field pair, fitted as "
        "antisym at full confidence; a few unpaired ones fall back to the labeled "
        "low-confidence 2-point estimate. The Resistivity tab shows the geometry-unset "
        "warning: the header never had sample dimensions, so absolute rho is "
        "scale-arbitrary while RRR and MR% stay valid.",
}


def _real_source(key):
    """Resolve a real source path through the untracked real_data_map.json, or None.

    Same resolver the test suite uses, so one map serves both. Absence is normal — the real
    files are not in the repo — and must not be an error.
    """
    try:
        sys.path.insert(0, str(APP / "tests" / "core"))
        from conftest import real_data
        return real_data(key)
    except Exception:
        return None


def _write_readme(out, rows):
    lines = [
        "# Example data files",
        "",
        "One runnable `.dat` per probe - generated by `tools/make_examples.py`. Most are",
        "synthetic and were never measured on any real material; the two marked *anonymized",
        "real* below are decimated subsets of real measurements with their sample identity,",
        "instrument serial numbers, acquisition date, formula weight and sample mass replaced",
        "(see 'Example data' in the top-level README). Load one in the GUI (`cryosweep-gui`)",
        "or run `cryosweep analyze <file>`.",
        "",
    ]
    for name, why, _sz, is_real in rows:
        tag = " *(anonymized real measurement)*" if is_real else ""
        lines += [f"## {name}{tag}", "", why + ".", ""]
        note = _GUI_NOTES.get(name)
        if note:
            lines += [f"**In the GUI:** {note}", ""]
    (out / "README.md").write_text("\n".join(lines))


def _build_one(spec, scratch, out):
    """Return (size, None) on success or (None, reason) when the generator cannot be driven."""
    import importlib
    name, _why, mod_name, fn_name, kwargs = spec
    mod = importlib.import_module(mod_name)
    dst = out / name
    getattr(mod, fn_name)(dst, **kwargs)
    if not dst.exists():
        return None, "generator wrote nothing"
    return dst.stat().st_size, None


def main() -> int:
    import tempfile
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(APP / "examples"))
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows, skipped, optional = [], [], []
    with tempfile.TemporaryDirectory() as td:
        scratch = pathlib.Path(td)
        for spec in SPECS:
            try:
                size, err = _build_one(spec, scratch, out)
            except Exception as e:
                size, err = None, f"{type(e).__name__}: {e}"
            (rows if err is None else skipped).append(
                (spec[0], spec[1], size, False) if err is None else (spec[0], err))

        # Anonymized real-derived examples: a missing source is a SKIP, never a failure.
        for name, key, why, mod_name, fn_name, kwargs in REAL_SPECS:
            src = _real_source(key)
            if src is None:
                dst = out / name
                if dst.exists():
                    # The committed copy still ships, so the README must keep documenting
                    # it: dropping shipped files from the docs on a machine without the
                    # private source broke the examples-readme tests (measured 2026-09-01).
                    rows.append((name, why, dst.stat().st_size, True))
                optional.append((name, "real source unavailable — committed copy left as is"))
                continue
            import importlib
            dst = out / name
            getattr(importlib.import_module(mod_name), fn_name)(src, dst, **kwargs)
            rows.append((name, why, dst.stat().st_size, True))

    _write_readme(out, rows)
    for n, why, sz, is_real in rows:
        print(f"  {n:<36} {sz:>8,} B   {'[real] ' if is_real else ''}{why}")
    for n, err in skipped:
        print(f"  SKIPPED {n:<28} {err}")
    for n, why in optional:
        print(f"  optional {n:<27} {why}")
    print(f"\n{len(rows)} example files written to {out}")
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
