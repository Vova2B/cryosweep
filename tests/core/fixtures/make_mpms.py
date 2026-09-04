import numpy as np

# Known ground truth (CGS). A single clean Curie-Weiss ramp at fixed nonzero field.
# theta=-30 keeps (T - theta) > 0 over the whole 2..300 K range (no pole).
C, THETA, FIELD, MASS_MG, MOLAR = 1.5, -30.0, 1000.0, 10.0, 200.0

# Bare CSV: column header on line 0, NO [Header]/[Data] blocks. Includes 'Long Scan Std Dev'
# so the detector's strong fingerprint (needs both 'long moment (emu)' AND 'long scan std dev') fires.
_COLS = "Time,Comment,Field (Oe),Temperature (K),Long Moment (emu),Long Scan Std Dev\n"

def write_mpms(path, n=150):
    T = np.linspace(2.0, 300.0, n)
    chi_molar = C / (T - THETA)                       # emu/(mol*Oe)
    mass_g = MASS_MG / 1000.0
    moment = chi_molar * FIELD * mass_g / MOLAR        # emu  (no noise -> exact oracle)
    with open(path, "w") as f:
        f.write(_COLS)
        for t, m in zip(T, moment):
            f.write(f"0,,{FIELD:.10e},{t:.10e},{m:.10e},1.0000000000e-08\n")
    return {"C": C, "theta": THETA, "field": FIELD, "mass_mg": MASS_MG,
            "molar": MOLAR, "mu_eff": 2.827 * (C ** 0.5)}

if __name__ == "__main__":
    import pathlib
    print(write_mpms(pathlib.Path(__file__).parent / "mpms_synth.dat"))
