import numpy as np

HEADER = """[Header]
TITLE,vsm_synth
BYAPP,VSM,1.0,1.0
INFO,5.0,MASS:Sample Mass (mg)
INFO,200.0,MOLWGHT:Formula Weight (g/mole)
INFO,1,ATOMS:Atoms per Formula Unit
[Data]
"""

def write_vsm(path, C=0.5, theta=-10.0, field=1000.0, mass_mg=5.0, molar=200.0, n=300, seed=0):
    rng = np.random.default_rng(seed)
    T = np.linspace(2.0, 300.0, n)
    chi_molar = C / (T - theta)                       # emu/(mol*Oe)
    mass_g = mass_mg / 1000.0
    moment = chi_molar * field * mass_g / molar        # emu
    moment += moment * 1e-4 * rng.standard_normal(n)   # tiny noise
    cols = "Temperature (K),Magnetic Field (Oe),Moment (emu),M. Std. Err. (emu)\n"
    lines = [HEADER, cols]
    for t, m in zip(T, moment):
        lines.append(f"{t:.6f},{field:.4f},{m:.8e},{abs(m)*1e-4:.2e}\n")
    with open(path, "w") as f:
        f.writelines(lines)
    return {"C": C, "theta": theta, "field": field, "mass_mg": mass_mg, "molar": molar, "mu_eff": 2.827 * (C ** 0.5)}


# --------------------------------------------------------------------------------------
# Anonymized real-derived EXAMPLE (examples/ only — never a committed fixture).
# --------------------------------------------------------------------------------------
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # _anonymize is a sibling:
# these generators are both run as scripts AND imported as modules by the test suite,
# and only the script form puts this directory on sys.path.
from _anonymize import (anonymize_header, assert_no_identity_leak, scrub_body,
                        set_info, split_at_data, write_subset)

# QD VSM headers carry far more identity than the TTO/ACMS ones: besides the sample fields,
# five INFO rows hold instrument SERIAL NUMBERS, which identify the physical magnetometer (and
# so the lab) rather than the sample. They are neutralised here, and — because
# assert_no_identity_leak seeds its tokens from the sample fields only — a dedicated test pins
# that none survives. MODULE_NAME / HW_VERSION / SOFTWARE_VERSION / APPNAME / MOMENT_UNITS are
# instrument MODEL and format provenance, identical on every machine of the type, and are kept.
_ID_KEYS = ("SAMPLE_MATERIAL", "SAMPLE_COMMENT", "SAMPLE_SIZE", "SAMPLE_SHAPE",
            "SAMPLE_VOLUME", "SAMPLE_MASS", "SAMPLE_MOLECULAR_WEIGHT",
            "MOTOR_SERIAL_NUMBER", "VSM_SERIAL_NUMBER", "PREAMP_SERIAL_NUMBER",
            "OVEN_SERIAL_NUMBER", "COIL_SERIAL_NUMBER")


def _info_key(ln):
    parts = [p.strip() for p in ln.split(",")]
    if len(parts) >= 3 and parts[0].upper() == "INFO":
        return parts[2].split(":", 1)[0].strip().upper()
    return None


_SAMPLE_RULES = ((lambda ln: _info_key(ln) in _ID_KEYS,
                  lambda ln: f"INFO,anonymized,{_info_key(ln)}"),)
_SAMPLE_LINE = (lambda ln: ln.split(",")[1] if _info_key(ln) in _ID_KEYS else None)


def write_real_example(src, dst, step=24, molar=300.0, mass_mg=1.25, atoms=1,
                       title="magnetization_vsm_multifield.dat"):
    """Anonymized subset of a real multi-field VSM measurement — every `step`-th row.

    The source has no MOLWGHT/MASS (its mass lives under the VSM-specific SAMPLE_MASS /
    SAMPLE_MOLECULAR_WEIGHT keys, which the loader does not read), so the file would analyze
    as `gated`. Neutral values are published instead: they are arbitrary, but chosen so the
    reported mu_eff lands in a physically sensible range (~3.7 mu_B) rather than the ~39 mu_B
    a round 200 g/mol / 5 mg placeholder would imply. theta and r2 do not depend on them.

    Decimation is safe here and was measured: the four held-field labels (100 Oe, 5 kOe,
    40 kOe, 100 kOe) and the four M(H) loops survive every step tested, and mu_eff/theta/r2
    move by ~1% between step 4 and step 48.
    """
    head, body = split_at_data(src)
    head = anonymize_header(head, title, _SAMPLE_RULES)
    assert_no_identity_leak(head, split_at_data(src)[0], _SAMPLE_LINE)
    head = set_info(head, "MOLWGHT", f"{molar}", "Formula Weight (g/mole)")
    head = set_info(head, "MASS", f"{mass_mg}", "Sample Mass (mg)")
    head = set_info(head, "ATOMS", f"{atoms}", "Atoms per Formula Unit")
    # Header anonymisation cannot reach the body's two identity channels: an absolute
    # Time Stamp column decodes to the acquisition instant, and Comment cells carry operator
    # and calibration free text. Neither is read by any analyzer, so scrubbing them is free.
    body, rep = scrub_body(head[-1].split(","), body)
    write_subset(dst, head, body, step=step)
    return {"step": step, "molar": molar, "mass_mg": mass_mg,
            "n_rows": len(body[::step]), **rep}
