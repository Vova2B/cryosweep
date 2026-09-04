# tests/core/fixtures/make_hall_tdep_std.py
"""Generate hall_tdep_std_synth.dat: make_hall_tdep.py's geometry PLUS the instrument
per-row std columns for bridge 1, for the O4 instrument-sigma propagation oracles
(2026-08-10 spec §2.2).

New columns vs hall_tdep_synth.dat:
  Bridge 1 Resistivity (Ohm-m) = Rxy / RATIO   (exact constant Resistance/Resistivity
                                                ratio 1000 — the firmware geometry factor,
                                                passes the <1e-6 constancy gate)
  Bridge 1 Std. Dev. (Ohm-m)   = STD_OHM_M     (CONSTANT absolute 1e-6 Ohm-m ->
                                                sigma_R = std * ratio = 1e-3 Ohm per row)

Hand-checkable closed forms (thickness 0.05 mm, sign +1):
  * Antisym region (3 pairs at B = 2/4/6 T), sigma_asym = sigma_R/sqrt(2):
      sigma_slope_inst = sqrt(sum(w_i^2 sig_i^2)), w_i=(B_i-4)/8 -> exactly 2.5e-4 Ohm/T
      r_h_sigma_instrument = 2.5e-4 * 5e-5 = 1.25e-8 m^3/C
  * 2-point tail (0 + 20000 Oe, B = 2 T), sigma_y^2 = 2 sigma_R^2:
      sigma_slope_inst = sqrt(2)*1e-3/2 = 7.0710678e-4 Ohm/T
      r_h_sigma_instrument = 3.5355339e-8 m^3/C
"""
import pathlib

from make_hall_tdep import (FIELDS_OE, TEMPS, TEMPS_EXT, _EXTENDED, HEADER,
                            rxy, rxx, rho2)

RATIO = 1000.0            # Resistance/Resistivity — the instrument geometry factor
STD_OHM_M = 1e-6          # constant per-row std (resistivity units) -> sigma_R = 1e-3 Ohm


def main():
    rows = ["Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
            "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m),"
            "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)"]
    for B_oe in FIELDS_OE:
        temps = TEMPS_EXT if B_oe in _EXTENDED else TEMPS
        for T in temps:
            r1 = rxy(T, B_oe)
            rows.append(f"{T:.4f},{B_oe:.1f},{r1:.10e},{r1 / RATIO:.10e},"
                        f"{STD_OHM_M:.10e},{rxx(T):.10e},{rho2(T):.10e}")
    out = pathlib.Path(__file__).with_name("hall_tdep_std_synth.dat")
    out.write_text(HEADER + "\n".join(rows) + "\n")
    print(f"wrote {out}  ({len(rows) - 1} data rows)")


if __name__ == "__main__":
    main()
