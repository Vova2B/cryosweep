"""Synth QD Resistivity-option file with a superconducting drop.

Bridge 1: rho_N = 1e-4 Ohm*cm plateau; piecewise-linear drop with rho = rho_N*(0.1+0.4*(T-7))
on 7-9 K so the 90%/50%/10% crossings are exactly 9/8/7 K; floor 1e-8 Ohm*cm below 6.5 K
(strictly >0 so points survive the analyzer's physical mask). Bridge 2: featureless metal.
Field column is 0 everywhere (single zero-field cooling ramp). Instrument column is Ohm-m
(= Ohm*cm / 100); INFO cross-section/length = 1 so geometry recompute matches."""
import numpy as np

RHO_N = 1e-4            # Ohm*cm  (100 uOhm*cm)
FLOOR = 1e-8            # Ohm*cm  below the transition (>0 so _clean_mask keeps the points)

_HEADER = (
    "[Header]\n"
    "TITLE,rho_sc_synth\n"
    "BYAPP, Resistivity, 2.0, 1.0\n"
    "INFO, , Sample1 Name\n"
    "INFO, 1, Sample1 Cross Section\n"
    "INFO, 1, Sample1 Length\n"
    "[Data]\n"
)

def _geom_header(cross_mm2, length_mm):
    """Header with real geometry set for BOTH channels (examples/ only). Cross Section = Length
    = 1 is the QD 'user never set geometry' sentinel the analyzer warns about."""
    return (
        "[Header]\n"
        "TITLE,rho_sc_synth\n"
        "BYAPP, Resistivity, 2.0, 1.0\n"
        "INFO, , Sample1 Name\n"
        f"INFO, {cross_mm2}, Sample1 Cross Section\n"
        f"INFO, {length_mm}, Sample1 Length\n"
        f"INFO, {cross_mm2}, Sample2 Cross Section\n"
        f"INFO, {length_mm}, Sample2 Length\n"
        "[Data]\n"
    )
_COLS = ("Comment,Time Stamp (sec),Status (code),Temperature (K),Magnetic Field (Oe),"
         "Sample Position (degrees),Bridge 1 Resistivity (Ohm-m),Bridge 2 Resistivity (Ohm-m),"
         "Number of Readings,Bridge 1 Resistance (Ohms),Bridge 2 Resistance (Ohms)\n")


def _rho_sc(T):
    """Bridge 1 rho(T) in Ohm*cm. Knots at 6.5 / 7 / 9 / 9.75 K; linear between."""
    if T >= 9.75:
        return RHO_N
    if T >= 9.0:
        return RHO_N * (0.9 + 0.1 * (T - 9.0) / 0.75)
    if T >= 7.0:
        return RHO_N * (0.1 + 0.4 * (T - 7.0))
    if T >= 6.5:
        return max(RHO_N * 0.1 * (T - 6.5) / 0.5, FLOOR)
    return FLOOR


def write_rho_sc(path, geometry=None):
    """geometry=None reproduces the committed fixture byte-for-byte. geometry=(cross_mm2,
    length_mm) is the examples/ variant: the header carries that real sample geometry and the
    resistance columns become R = rho / (A/L) so the file is self-consistent (the resistivity
    columns — everything the no-user-geometry analysis path reads — are unchanged)."""
    # cooling ramp, grid includes every knot so linear interp between samples is exact
    T = np.concatenate([np.linspace(2.0, 6.5, 10), np.linspace(6.5, 9.75, 66)[1:],
                        np.linspace(9.75, 300.0, 60)[1:]])[::-1]
    # A/L in meters: mm^2 * 1e-6 / (mm * 1e-3); None keeps the legacy resistance == resistivity
    a_over_l = (geometry[0] * 1e-3 / geometry[1]) if geometry else 1.0
    with open(path, "w") as f:
        f.write(_geom_header(*geometry) if geometry else _HEADER)
        f.write(_COLS)
        for t in T:
            r1 = _rho_sc(float(t))                       # Ohm*cm
            r2 = 5e-5 + 1e-7 * float(t)                  # featureless metal, Ohm*cm
            f.write(f",0,,{t:.6f},0.0000,90.0,{r1 / 100:.10e},{r2 / 100:.10e},25,"
                    f"{r1 / 100 / a_over_l:.10e},{r2 / 100 / a_over_l:.10e}\n")
    return {"tc_onset_k": 9.0, "tc_mid_k": 8.0, "tc_zero_k": 7.0,
            "rho_n_ohm_cm": RHO_N, "sc_channel": 1, "normal_channel": 2}


def write_rho_sc_example(path, geometry=(2.0, 2.0), seed=11):
    """examples/ file ONLY — never a committed fixture. Same physics story as the fixture
    (resistive superconducting transition, featureless-metal second channel) but shaped like a
    measurement instead of a unit test:

      * uniform-density cooling grid (2.5 K steps, 0.25 K only through the 2-12 K transition
        window) — the fixture's 56%-of-points-below-10-K grid collapses the robust MAD scale
        and falsely flags 51 points of the LINEAR bridge-2 metal as outliers;
      * near-flat normal state rho_N(T) = 1e-4*(0.92 + 0.08*T/300) Ohm*cm (the Tc detector's
        90%-crossing + narrowness gate require a plateau-like normal state — verified: a
        sloped metal moves the onset crossing to ~190 K and the gate then declines);
      * smooth tanh transition, mid 8 K;
      * noise floor ~1.2e-6 Ohm*cm below Tc (a real voltmeter floor, not the fixture's exact
        1e-8) -> RRR ~ 80 instead of the unphysical 1e4, while still < 2% rho_N so the Tc
        detector fires;
      * 0.3% multiplicative instrument noise everywhere.
    """
    rng = np.random.default_rng(seed)
    T = np.concatenate([np.arange(300.0, 12.0, -2.5), np.arange(12.0, 2.0 - 1e-9, -0.25)])
    a_over_l = geometry[0] * 1e-3 / geometry[1]
    with open(path, "w") as f:
        f.write(_geom_header(*geometry))
        f.write(_COLS)
        for t in T:
            rho_n = 1e-4 * (0.92 + 0.08 * t / 300.0)           # Ohm*cm, near-flat normal state
            frac = 0.5 * (1.0 + np.tanh((t - 8.0) / 0.45))     # SC order-parameter-ish step
            floor = 1.2e-6 * (1.0 + 0.2 * rng.standard_normal())
            r1 = max(rho_n * frac, 1e-8) + max(floor, 2e-7)
            r1 *= 1.0 + 0.003 * rng.standard_normal()
            r2 = (5e-5 + 1e-7 * t) * (1.0 + 0.003 * rng.standard_normal())
            f.write(f",0,,{t:.6f},0.0000,90.0,{r1 / 100:.10e},{r2 / 100:.10e},25,"
                    f"{r1 / 100 / a_over_l:.10e},{r2 / 100 / a_over_l:.10e}\n")
    return {"tc_mid_k": 8.0, "rho_n_ohm_cm": 1e-4, "sc_channel": 1, "normal_channel": 2}


if __name__ == "__main__":
    import pathlib
    print(write_rho_sc(pathlib.Path(__file__).parent / "rho_sc_synth.dat"))
