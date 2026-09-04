import numpy as np

# Known ground truth (SI units for the fit):
GAMMA = 0.010      # J/(mol*K^2)   electronic
BETA = 5.0e-4      # J/(mol*K^4)   lattice (Debye T^3)
N_ATOMS = 3
R = 8.314462618

_HEADER = """[Header]
TITLE,hc_synth
BYAPP, HeatCapacity, 2.0, 1.0
INFO, 3, ATOMS:Atoms per Formula Unit
[Data]
"""
_COLS = ("Comment,Time Stamp (sec),Status (code),Sample Temp (Kelvin),"
         "Magnetic Field (Oe),Samp HC (mJ/mole-K)\n")

THETA_D = (12 * np.pi**4 * N_ATOMS * R / (5 * BETA)) ** (1.0 / 3.0)   # == theta_D from BETA


def _debye_cp(T, theta_d, n=N_ATOMS):
    """Full Debye lattice heat capacity 9nR(T/theta)^3 * int_0^{theta/T} x^4 e^x/(e^x-1)^2 dx.
    Numeric quad (example path only) so the example's Cp saturates at Dulong-Petit instead of
    following beta*T^3 to absurdity at 300 K."""
    from scipy.integrate import quad
    def one(t):
        xm = theta_d / t
        val, _ = quad(lambda x: x**4 * np.exp(x) / np.expm1(x) ** 2, 0.0, xm, limit=200)
        return 9.0 * n * R * (t / theta_d) ** 3 * val
    return np.array([one(float(t)) for t in np.atleast_1d(T)])


def write_hc(path, example=False):
    """Zero-field temperature ramp 2->15 K. Cp = gamma*T + beta*T^3 so Cp/T = gamma + beta*T^2.
    Column 'Samp HC' is in mJ/(mol*K) (analyzer scales x1e-3 to J).

    example=True (examples/ only; the committed fixture stays byte-identical): the ramp extends
    to 300 K with Cp = gamma*T + n*C_Debye(T, THETA_D) (same theta_D the fixture's beta implies)
    + 0.3% noise, so the 7-parameter full-range Debye-Einstein fit is available (T_max >= 50 K)
    and converges on known ground truth; the <=10 K window still fits gamma/beta as before."""
    if example:
        T = np.concatenate([np.arange(2.0, 15.0 + 1e-9, 0.5),
                            np.arange(16.0, 50.0 + 1e-9, 1.0),
                            np.arange(52.5, 300.0 + 1e-9, 2.5)])
        cp_J = GAMMA * T + _debye_cp(T, THETA_D)
        rng = np.random.default_rng(7)
        cp_J = cp_J * (1.0 + 0.003 * rng.standard_normal(T.size))
    else:
        T = np.arange(2.0, 15.0 + 1e-9, 0.5)     # 27 points; 17 are <= 10 K (the low-T window)
        cp_J = GAMMA * T + BETA * T**3            # J/(mol*K)
    rows = []
    for t, c in zip(T, cp_J):
        cp_mJ = c * 1e3                             # column unit mJ/(mol*K)
        rows.append(f",0,,{t:.4f},0.0,{cp_mJ:.6f}\n")
    with open(path, "w") as f:
        f.write(_HEADER); f.write(_COLS); f.writelines(rows)
    theta_D = (12 * np.pi**4 * N_ATOMS * R / (5 * BETA)) ** (1.0 / 3.0)
    return {"gamma": GAMMA, "beta": BETA, "n_atoms": N_ATOMS, "theta_D": theta_D}

if __name__ == "__main__":
    import pathlib
    print(write_hc(pathlib.Path(__file__).parent / "hc_synth.dat"))


# --------------------------------------------------------------------------------------
# Anonymized real-derived EXAMPLE (examples/ only — never a committed fixture).
# --------------------------------------------------------------------------------------
import pathlib as _pl, sys as _sys
_sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))  # _anonymize is a sibling:
# these generators are both run as scripts AND imported as modules by the test suite,
# and only the script form puts this directory on sys.path.
from _anonymize import (anonymize_header, assert_no_identity_leak, scrub_body,
                        set_info, split_at_data, write_subset)

# The HC header is short: TITLE plus numeric MASS/MASSERR/MOLWGHT/ATOMS and APPNAME. APPNAME is
# format provenance (kept, as in make_tto). There is no free-text sample field to strip — the
# identity that has to go is the FORMULA WEIGHT, a precise enough number to fingerprint a
# compound, so it and the sample mass are replaced with round neutral values.
#
# MEASURED: this changes no result. The analyzed column is `Samp HC (mJ/mole-K)`, which the QD
# software already reduced to a per-mole basis using the mass and formula weight entered at
# measurement time, and nothing in the heat-capacity path reads molar_mass/mass_mg — gamma,
# theta_D and r2 are byte-identical across MOLWGHT 945.68 -> 1900. These two rows are metadata
# only. The file does still carry the absolute `Total/Addenda HC (µJ/K)` columns beside the
# per-mole one, so their ratio recovers the ORIGINAL formula-weight-to-mass ratio; what it
# cannot recover is the formula weight itself, because the true sample mass no longer ships.
_SAMPLE_RULES = ()
_SAMPLE_LINE = (lambda ln: None)


def write_real_example(src, dst, step=1, molar=950.0, mass_mg=5.0,
                       title="heat_capacity_multifield.dat"):
    """Anonymized subset of a real multi-field Cp(T) measurement (fields 0, 5, 10, 13 T).

    step=1 by default: at 337 rows the whole file is already example-sized, and each row is an
    independent measurement point, so decimation costs per-field point density rather than
    file size worth saving.
    """
    head, body = split_at_data(src)
    head = anonymize_header(head, title, _SAMPLE_RULES)
    assert_no_identity_leak(head, split_at_data(src)[0], _SAMPLE_LINE)
    head = set_info(head, "MOLWGHT", f"{molar}", "Formula Weight (g/mole)")
    head = set_info(head, "MASS", f"{mass_mg}", "Sample Mass (mg)")
    # Header anonymisation cannot reach the body's two identity channels: an absolute
    # Time Stamp column decodes to the acquisition instant, and Comment cells carry operator
    # and calibration free text. Neither is read by any analyzer, so scrubbing them is free.
    body, rep = scrub_body(head[-1].split(","), body)
    write_subset(dst, head, body, step=step)
    return {"step": step, "molar": molar, "mass_mg": mass_mg,
            "n_rows": len(body[::step]), **rep}
