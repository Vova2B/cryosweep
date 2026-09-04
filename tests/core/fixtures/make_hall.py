import numpy as np

# Known ground truth (SI): R_H, thickness, longitudinal resistivity.
R_H = -5.0e-8       # m^3/C  (n-type)
THICK_M = 1.0e-4    # m  (0.1 mm)
RHO_XX = 1.0e-6     # Ohm*m  (longitudinal, used in hall_synth.dat Bridge 2)
RHO_XX_LONG = 2.0e-6  # Ohm*m  (DISTINCT longitudinal for hall_long_synth.dat — proves separate-file routing)
E_CHG = 1.602176634e-19

_HEADER = """[Header]
TITLE,hall_synth
BYAPP, Resistivity, 2.0, 1.0
INFO, , Sample1 Name
INFO, 1, Sample1 Cross Section
INFO, 1, Sample1 Length
[Data]
"""

_COLS = ("Comment,Time Stamp (sec),Status (code),Temperature (K),Magnetic Field (Oe),"
         "Sample Position (degrees),Bridge 1 Resistivity (Ohm-m),Bridge 2 Resistivity (Ohm-m),"
         "Number of Readings,Bridge 1 Resistance (Ohms),Bridge 2 Resistance (Ohms)\n")

def _loop_rows(T, n_field=81, even_admix=2.0e-5, offset=1.0e-6, seed=0):
    """One up+down field loop at held T. Bridge1 = R_xy (odd Hall + even admixture + offset);
    Bridge2 = longitudinal R_xx (field-independent, rho_xx)."""
    rng = np.random.default_rng(seed)
    H = np.concatenate([np.linspace(-90000, 90000, n_field),
                        np.linspace(90000, -90000, n_field)])
    B = H / 10000.0
    slope = R_H / THICK_M                          # Ohm/T
    r_xy = slope * B + even_admix * B**2 + offset  # odd Hall + even admixture + constant offset
    r_xy += abs(slope * 90000/10000.0) * 1e-4 * rng.standard_normal(r_xy.size)
    # Bridge2 longitudinal resistance such that instrument resistivity col == RHO_XX:
    # instrument resistivity = resistance * (Cross Section / Length) with header 1/1 -> equal.
    r_xx = np.full(H.size, RHO_XX)                 # report as both resistivity and resistance
    rows = []
    for h, rxy, rxx in zip(H, r_xy, r_xx):
        rows.append(f",0,,{T:.4f},{h:.4f},90.0,{rxy:.8e},{rxx:.8e},25,{rxy:.8e},{rxx:.8e}\n")
    return rows

def _geom_header(cross_mm2, length_mm):
    """Header with real geometry for BOTH channels (examples/ only; 1/1 = QD unset sentinel)."""
    return ("[Header]\nTITLE,hall_synth\nBYAPP, Resistivity, 2.0, 1.0\n"
            "INFO, , Sample1 Name\n"
            f"INFO, {cross_mm2}, Sample1 Cross Section\nINFO, {length_mm}, Sample1 Length\n"
            f"INFO, {cross_mm2}, Sample2 Cross Section\nINFO, {length_mm}, Sample2 Length\n"
            "[Data]\n")


def write_hall(path, temps=(10.0, 100.0, 300.0), geometry=None):
    """geometry=None reproduces the committed fixture byte-for-byte. geometry=(cross_mm2,
    length_mm) (examples/ variant) writes that geometry into the header and makes the
    resistivity columns rho = R * A/L; the RESISTANCE columns — what the Hall analyzer and
    the geometry-recompute path read — are untouched, so R_H/n/mobility are unchanged."""
    a_over_l = (geometry[0] * 1e-3 / geometry[1]) if geometry else 1.0
    lines = [_geom_header(*geometry) if geometry else _HEADER, _COLS]
    for i, T in enumerate(temps):
        if geometry is None:
            lines += _loop_rows(T, seed=i)
        else:
            for row in _loop_rows(T, seed=i):
                c = row.rstrip("\n").split(",")
                # Bridge1 (Hall): RESISTANCE (idx 9) is the R_xy signal -> keep; make its
                # resistivity column (idx 6) consistent: rho = R * A/L.
                # Bridge2 (longitudinal): the RESISTIVITY column (idx 7) is what the mobility
                # path reads -> keep at RHO_XX; make its resistance (idx 10) R = rho / (A/L).
                c[6] = f"{float(c[9]) * a_over_l:.8e}"
                c[10] = f"{float(c[7]) / a_over_l:.8e}"
                lines.append(",".join(c) + "\n")
    with open(path, "w") as f:
        f.writelines(lines)
    return {"R_H": R_H, "thickness_m": THICK_M, "rho_xx": RHO_XX,
            "slope_ohm_per_T": R_H / THICK_M,
            "n": 1.0 / (E_CHG * abs(R_H)), "sigma": 1.0 / RHO_XX,
            "mu": abs(R_H) / RHO_XX}

def write_long_only(path, temps=(10.0, 100.0, 300.0)):
    """Separate longitudinal-only file: Bridge 2 = RHO_XX_LONG (DISTINCT from hall_synth.dat's
    RHO_XX), no Hall channel (Bridge 1 empty).  Using a different value proves the two-file path
    actually reads this file rather than falling back to the same-file Bridge 2."""
    lines = [_HEADER, _COLS]
    for T in temps:
        for h in (-90000.0, 0.0, 90000.0):
            lines.append(f",0,,{T:.4f},{h:.4f},90.0,,{RHO_XX_LONG:.8e},25,,{RHO_XX_LONG:.8e}\n")
    with open(path, "w") as f:
        f.writelines(lines)
    return {"rho_xx": RHO_XX_LONG}

if __name__ == "__main__":
    import pathlib
    here = pathlib.Path(__file__).parent
    truth = write_hall(here / "hall_synth.dat")
    write_long_only(here / "hall_long_synth.dat")
    print(truth)
