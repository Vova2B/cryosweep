# tests/core/fixtures/make_hc_schottky.py
import numpy as np
R = 8.314462618
GAMMA, BETA, N_ATOMS, MOLWGHT = 0.005, 2.0e-4, 2, 150.0
G_FACTOR, MU = 2.0, 0.6717                                    # Zeeman: Delta = g*MU*B[T]
# C1: fields chosen so T_peak=0.417*Delta lands INSIDE [1.9,14] K at >=3 fields.
# 6/7.5/9 T -> Delta 8.06/10.08/12.09 K -> T_peak 3.36/4.20/5.04 K; 300 Oe -> Kramers-low/undetermined.
_FIELDS_OE = (300.0, 60000.0, 75000.0, 90000.0)              # ~0.03 T, 6 T, 7.5 T, 9 T
F_FRAC = 0.30                                                 # TLS fraction per formula unit
# C1b: NO nuclear tail here -- default schottky_include_nuclear=False means M2 is never attempted,
# so an injected alphaN/T^2 would be unmodeled contamination. Nuclear is unit-tested in Task 2.

_HEAD = ("[Header]\nTITLE,hc_schottky_synth\nBYAPP,HeatCapacity,1.0,1.0\n"
         f"INFO,{MOLWGHT},MOLWGHT:Formula Weight (g/mole)\n"
         f"INFO,{N_ATOMS},ATOMS:Atoms per Formula Unit\n[Data]\n")
_COLS = "Sample Temp (Kelvin),Field (Oersted),Samp HC (mJ/mole-K)\n"

def _sch(T, f, D):
    z = D / T; ez = np.exp(z)
    return f * R * z ** 2 * ez / (1 + ez) ** 2

def write_hc_schottky(path):
    T = np.linspace(1.9, 14.0, 40)
    rows = []
    for f_oe in _FIELDS_OE:
        B = f_oe / 1e4
        Delta = max(G_FACTOR * MU * B, 0.05)                 # ~0 at lowest field (Kramers), rises with H
        cp = GAMMA * T + BETA * T ** 3 + _sch(T, F_FRAC, Delta)
        for t, c in zip(T, cp):
            rows.append((t, f_oe, c * 1000.0))               # J -> mJ
    with open(path, "w") as fh:
        fh.write(_HEAD); fh.write(_COLS)
        for t, f_oe, cp_mJ in rows:
            fh.write(f"{t:.10f},{f_oe:.6f},{cp_mJ:.10f}\n")
    return {"gamma": GAMMA, "beta": BETA, "n_atoms": N_ATOMS, "g_factor": G_FACTOR,
            "f": F_FRAC, "fields_oe": _FIELDS_OE}

if __name__ == "__main__":
    import pathlib
    print(write_hc_schottky(pathlib.Path(__file__).parent / "hc_schottky_synth.dat"))
