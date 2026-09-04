# tests/core/fixtures/make_hall_tdep.py
"""Generate hall_tdep_synth.dat: rho(T) at fixed fields with an EXACT known R_H(T).
Rxy(T,B) = slope_true*B + even(T,B); antisymmetrizing over +/-B leaves slope_true*B exactly.
R_H_true = slope_true * thickness * sign = -3.0e-8 m^3/C (electrons).

Each fixed field gets a temperature ramp (1 K step) written as a contiguous block, so the
real `segment_sweeps` detector resolves a temperature-swept segment per held field — the
same production path that runs on the real Hall / resistivity measurement files.

SEGMENTER-AWARE geometry (Sub-feature B, 2-point tail):
The production segmenter (`find_blocks`, stability.window = 16) attributes ~16 rows on each
side of every field TRANSITION to the field axis (the field jump dominates the local
peak-to-peak activity there) and discards them — so each held-field ramp survives only its
interior. To engineer a CLEAN high-T two-point tail through that real path:

  * Field ORDER is [-20000, 40000, -40000, 60000, -60000, 20000, 0]: the two fields carrying
    the tail (0 and +20000 Oe) are the last two blocks, whose tops survive the fewest
    transition boundaries.
  * PAIRED fields ramp 2..55 K (TEMPS): after boundary erosion they survive ~18..40 K, so
    every +/-B pair overlaps over 18..40 -> the trusted antisym R_H = -3.0e-8 there.
  * EXTENDED fields 0 and +20000 Oe ramp 2..70 K (TEMPS_EXT): they survive up to ~55 K, so
    at 41..55 K ONLY the zero field and +20000 Oe remain (all pairs gone) -> the
    zero-field-subtracted 2-point fallback fires there. Its slope is
    (R(+20000) - R(0)) / (2 T) = SLOPE_TRUE + 1e-4 = -5e-4 -> R_H = -5e-4 * thickness = -2.5e-8,
    low-confidence, T > 40 K. (Regenerate + re-verify the oracle if stability.window changes.)

Columns (5):
  Bridge 1 Resistance (Ohms)      — transverse Hall channel (ch1)
  Bridge 2 Resistance (Ohms)      — longitudinal resistance (ch2)
  Bridge 2 Resistivity (Ohm-m)    — genuine longitudinal resistivity (ch2)
                                    rho2(T) = 1e-6 + 1e-8 * T  (Ohm*m)
The resistivity column is what the production _long_rho_xx helper reads; having it
in the fixture lets the test verify dimensionally-correct sigma/mobility."""
import pathlib

THICKNESS_M = 5e-5          # 0.05 mm
SIGN = 1
R_H_TRUE = -3.0e-8          # m^3/C
SLOPE_TRUE = R_H_TRUE / (THICKNESS_M * SIGN)   # Ohm/T = -6.0e-4
# Block order: extended tail fields (0, +20000) LAST so their tops survive segmentation.
FIELDS_OE = [-20000, 40000, -40000, 60000, -60000, 20000, 0]
TEMPS = list(range(2, 56))                     # paired fields: 2..55 K (survive ~18..40 after erosion)
TEMPS_EXT = list(range(2, 71))                 # 0 & +20000 Oe: 2..70 K (survive to ~55 -> 41..55 K tail)
_EXTENDED = {0, 20000}                          # fields carrying the sparse 2-point tail

def rxy(T, B_oe):
    B = B_oe / 10000.0
    return SLOPE_TRUE * B + 1e-3 + 5e-5 * B**2 + 2e-6 * T   # even terms cancel under antisym

def rxx(T):
    return 1e-3 + 1e-5 * T

def rho2(T):
    """Genuine longitudinal resistivity for ch2 (Ohm*m)."""
    return 1e-6 + 1e-8 * T

HEADER = """[Header]
BYAPP, Resistivity
INFO, hall_tdep_synth, SAMPLE
[Data]
"""

def write_tdep(path, geometry=None):
    """Write the file to `path`. No-kwargs output is byte-identical to the committed
    hall_tdep_synth.dat (main() below regenerates the fixture through this).
    geometry=(cross_mm2, length_mm) (examples/ variant) only ADDS header geometry INFO lines:
    with (2.0, 2.0) -> A/L = 1e-3 m the existing columns are already exactly self-consistent
    (rho2 = 1e-6 + 1e-8*T == rxx * 1e-3), so no data cell changes."""
    header = HEADER
    if geometry:
        header = HEADER.replace("[Data]\n",
            f"INFO, {geometry[0]}, Sample1 Cross Section\nINFO, {geometry[1]}, Sample1 Length\n"
            f"INFO, {geometry[0]}, Sample2 Cross Section\nINFO, {geometry[1]}, Sample2 Length\n"
            "[Data]\n")
    rows = ["Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)"]
    # one fixed-field temperature ramp per field (rows grouped so the segmenter sees T-sweeps at held H)
    for B_oe in FIELDS_OE:
        temps = TEMPS_EXT if B_oe in _EXTENDED else TEMPS
        for T in temps:
            rows.append(f"{T:.4f},{B_oe:.1f},{rxy(T,B_oe):.10e},{rxx(T):.10e},{rho2(T):.10e}")
    out = pathlib.Path(path)
    out.write_text(header + "\n".join(rows) + "\n")
    print(f"wrote {out}  ({len(rows)-1} data rows, R_H_true={R_H_TRUE})")


def main():
    write_tdep(pathlib.Path(__file__).with_name("hall_tdep_synth.dat"))

if __name__ == "__main__":
    main()
