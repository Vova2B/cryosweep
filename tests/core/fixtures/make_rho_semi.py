"""Synth QD Resistivity-option files for activated transport (Arrhenius) — the positive
control, the VRH impostor, and the weak-insulator decline case.

No shipped example was insulating before this (measured 2026-09-05: the whole corpus holds
exactly ONE insulating channel, a real local-only file whose rho changes just 1.3x over
3-340 K — a bad metal that the honest Arrhenius fit DECLINES on). A fit with no ground
truth and no negative control is not verified, so:

* `write_rho_semi`     — ch1 intrinsic-like semiconductor with a PLANTED E_a = 60 meV
                         (rho = rho0*exp(E_a/k_B T), exact), ch2 featureless metal. The
                         fit must recover 60 meV.
* `write_rho_vrh`      — ch1 Mott VRH (rho = rho0*exp[(T0/T)^(1/4)], T0 = 1e6 K): a
                         genuine insulator that is NOT Arrhenius; the Arrhenius fit must
                         flag itself `window_sensitive` (E_a drifts with the window).
* `write_rho_weak`     — ch1 rho falling linearly by ~25% (the real corpus channel's
                         shape): classified insulating, but < 1 e-fold of change — the
                         fit must DECLINE with `insufficient_rho_span`.
* `write_rho_semi_example` — the examples/ variant of the semiconductor: measurement-like
                         grid, 0.3% noise, real geometry header.
"""
import numpy as np

KB_EV = 8.617333262e-5          # Boltzmann, eV/K
EA_EV = 0.060                   # planted activation energy: 60 meV
RHO0 = 1e-3                     # Ohm*cm prefactor (semiconductor)
VRH_T0 = 1.0e6                  # K (Mott T0)
VRH_RHO0 = 1e-4                 # Ohm*cm

_COLS = ("Comment,Time Stamp (sec),Status (code),Temperature (K),Magnetic Field (Oe),"
         "Sample Position (degrees),Bridge 1 Resistivity (Ohm-m),Bridge 2 Resistivity (Ohm-m),"
         "Number of Readings,Bridge 1 Resistance (Ohms),Bridge 2 Resistance (Ohms)\n")


def _header(title, cross_mm2=None, length_mm=None):
    geom = (f"INFO, {cross_mm2}, Sample1 Cross Section\n"
            f"INFO, {length_mm}, Sample1 Length\n"
            f"INFO, {cross_mm2}, Sample2 Cross Section\n"
            f"INFO, {length_mm}, Sample2 Length\n") if cross_mm2 else (
            "INFO, 1, Sample1 Cross Section\n"
            "INFO, 1, Sample1 Length\n")
    return ("[Header]\n"
            f"TITLE,{title}\n"
            "BYAPP, Resistivity, 2.0, 1.0\n"
            "INFO, , Sample1 Name\n" + geom + "[Data]\n")


def _write(path, title, rho1_fn, T, geometry=None, noise=0.0, seed=7):
    rng = np.random.default_rng(seed)
    a_over_l = (geometry[0] * 1e-3 / geometry[1]) if geometry else 1.0
    with open(path, "w") as f:
        f.write(_header(title, *(geometry or (None, None))))
        f.write(_COLS)
        for t in T:
            r1 = rho1_fn(float(t)) * (1.0 + noise * rng.standard_normal())
            r2 = (5e-5 + 1e-7 * float(t)) * (1.0 + noise * rng.standard_normal())
            f.write(f",0,,{t:.6f},0.0000,90.0,{r1 / 100:.10e},{r2 / 100:.10e},25,"
                    f"{r1 / 100 / a_over_l:.10e},{r2 / 100 / a_over_l:.10e}\n")


def _rho_arrhenius(t):
    return RHO0 * np.exp(EA_EV / (KB_EV * t))


def write_rho_semi(path):
    T = np.linspace(300.0, 80.0, 120)               # cooling ramp
    _write(path, "rho_semi_synth", _rho_arrhenius, T)
    return {"ea_mev": EA_EV * 1000, "rho0_ohm_cm": RHO0}


def write_rho_vrh(path):
    T = np.linspace(300.0, 50.0, 120)
    _write(path, "rho_vrh_synth",
           lambda t: VRH_RHO0 * np.exp((VRH_T0 / t) ** 0.25), T)
    return {"t0_k": VRH_T0}


def write_rho_weak(path):
    # falls ~25% linearly over the range: dRho/dT < 0 -> "insulating", but 0.29 e-folds
    T = np.linspace(340.0, 3.0, 150)
    _write(path, "rho_weak_synth", lambda t: 4e-4 * (1.0 - 0.25 * t / 340.0), T)
    return {"efolds": float(np.log(1.0 / 0.75))}


def write_rho_semi_example(path, geometry=(2.0, 2.0), seed=13):
    """examples/ file ONLY — same planted physics, measurement-shaped: 1 K cooling grid,
    0.3% multiplicative instrument noise, real geometry header."""
    T = np.arange(300.0, 80.0 - 1e-9, -1.0)
    _write(path, "resistivity_semiconductor.dat", _rho_arrhenius, T,
           geometry=geometry, noise=0.003, seed=seed)
    return {"ea_mev": EA_EV * 1000, "rho0_ohm_cm": RHO0}


if __name__ == "__main__":
    import pathlib
    d = pathlib.Path(__file__).parent
    print(write_rho_semi(d / "rho_semi_synth.dat"))
    print(write_rho_vrh(d / "rho_vrh_synth.dat"))
    print(write_rho_weak(d / "rho_weak_synth.dat"))
