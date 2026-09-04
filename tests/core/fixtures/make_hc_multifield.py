import numpy as np

# Multi-field heat-capacity dataset (QD format) where the ZERO-FIELD low-T ramp is interleaved
# row-wise with 1 T and 3 T measurements at the same temperatures. Sweep-segmentation fragments
# the zero-field ramp and loses its low-T points (real bug: the low-T heat-capacity file
# scored 0.13/0.30); the
# analyzer must instead select the lowest-|field| group directly and recover the lattice Debye fit.
#
# Zero-field model is exactly Cp = gamma*T + beta*T^3  -> Cp/T = gamma + beta*T^2 (r^2 == 1).
R = 8.314462618
GAMMA, BETA, N_ATOMS, MOLWGHT = 0.005, 2.0e-4, 2, 150.0      # J/mol/K^2, J/mol/K^4
THETA_D = (12 * np.pi**4 * N_ATOMS * R / (5 * BETA)) ** (1.0/3.0)
_FIELDS_OE = (0.3, 10000.0, 30000.0)                          # ~0 T, 1 T, 3 T

_HEAD = ("[Header]\nTITLE,hc_multifield_synth\nBYAPP,HeatCapacity,1.0,1.0\n"
         f"INFO,{MOLWGHT},MOLWGHT:Formula Weight (g/mole)\n"
         f"INFO,{N_ATOMS},ATOMS:Atoms per Formula Unit\n[Data]\n")
_COLS = "Sample Temp (Kelvin),Field (Oersted),Samp HC (mJ/mole-K)\n"

def write_hc_multifield(path):
    T = np.linspace(2.0, 10.0, 17)
    rows = []
    for t in T:                                              # interleave fields at each T
        for f in _FIELDS_OE:
            if abs(f) < 50:                                  # zero field: pure lattice
                cp = GAMMA * t + BETA * t**3
            else:                                            # in-field: different curve (suppressed)
                cp = GAMMA * t + BETA * t**3 + 0.02 * (f / 10000.0) * t
            rows.append((t, f, cp * 1000.0))                 # J -> mJ for the column
    with open(path, "w") as fh:
        fh.write(_HEAD); fh.write(_COLS)
        for t, f, cp_mJ in rows:
            fh.write(f"{t:.10f},{f:.6f},{cp_mJ:.10f}\n")
    return {"gamma": GAMMA, "beta": BETA, "n_atoms": N_ATOMS, "theta_D": THETA_D}

if __name__ == "__main__":
    import pathlib
    print(write_hc_multifield(pathlib.Path(__file__).parent / "hc_multifield_synth.dat"))
