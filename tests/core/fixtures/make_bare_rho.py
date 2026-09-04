import numpy as np

# Bare, TAB-separated resistivity export (Origin "dc rho" style): column header on line 0,
# NO [Header]/[Data] blocks, NO Field column (single zero-field cooling ramp), resistivity
# already in micro-ohm-cm. Mirrors the real bare tab-separated dc-resistivity file.
# Metallic ramp rho = RHO0 + SLOPE*T (uOhm-cm) -> RRR = rho(T_hi)/rho(T_lo).
RHO0, SLOPE = 50.0, 0.5          # uOhm-cm ; at 2 K ~51, at 300 K ~200
_HEADER = "T (K)\tResistivity He4+He3 (mikroOhm-cm)_H=0T_COOL\n"

def write_bare_rho(path, n=120):
    T = np.linspace(2.0, 300.0, n)[::-1]              # cooling: high -> low T, like the real file
    rho_uohm_cm = RHO0 + SLOPE * T                    # uOhm-cm (no noise -> exact oracle)
    with open(path, "w") as f:
        f.write(_HEADER)
        for t, r in zip(T, rho_uohm_cm):
            f.write(f"{t:.10f}\t{r:.10f}\n")
    rho_ohm_cm_hi = (RHO0 + SLOPE * 300.0) * 1e-6
    rho_ohm_cm_lo = (RHO0 + SLOPE * 2.0) * 1e-6
    return {"rho0_uohm_cm": RHO0, "slope": SLOPE,
            "rho_ohm_cm_at_300": rho_ohm_cm_hi, "rho_ohm_cm_at_2": rho_ohm_cm_lo,
            "rrr": rho_ohm_cm_hi / rho_ohm_cm_lo}

if __name__ == "__main__":
    import pathlib
    print(write_bare_rho(pathlib.Path(__file__).parent / "bare_rho_synth.dat"))
