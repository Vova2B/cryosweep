"""Generate `hall_onesided_synth.dat` — the Stage-A-only Hall branch (final-review F6).

One temperature (10 K), **positive fields only**, so `_antisymmetrize` finds no symmetric
overlap and returns empty: `anti is None` and both R_H and its sigma must fall back to the
raw Stage-A fit. Mutation M14 (`rh_sig_trusted = pt.r_h_sigma` — dropping the fallback)
survived the whole suite because no fixture reached this branch.

The R_xy signal is deliberately noise-dominated (slope 1e-6 Ohm/T under 2e-5 Ohm scatter,
seed 7) so the relative residual sigma clears the 50 % `_REL_SIGMA_WARN` threshold — Stage A
is the noisier stage (it retains the even-in-H R_xx admixture), so it is exactly the branch
the "always-on" warning must cover.

Oracles measured from this file are pinned in `tests/core/test_hall_sigma.py`.
"""
import numpy as np

H = np.linspace(0.0, 90000.0, 21)          # Oe, POSITIVE only
rng = np.random.default_rng(7)
R = 1e-6 * (H / 1e4) + 2e-5 * rng.standard_normal(H.size)

lines = [
    "[Header]",
    "TITLE,hall_onesided_synth",
    "BYAPP, Resistivity, 2.0, 1.0",
    "INFO, , Sample1 Name",
    "INFO, 1, Sample1 Cross Section",
    "INFO, 1, Sample1 Length",
    "[Data]",
    "Comment,Time Stamp (sec),Status (code),Temperature (K),Magnetic Field (Oe),"
    "Sample Position (degrees),Bridge 1 Resistivity (Ohm-m),Bridge 2 Resistivity (Ohm-m),"
    "Number of Readings,Bridge 1 Resistance (Ohms),Bridge 2 Resistance (Ohms)",
]
for h, r in zip(H, R):
    lines.append(f",0,,10.0000,{h:.4f},90.0,{r:.8e},1.00000000e-06,25,"
                 f"{r:.8e},1.00000000e-06")

with open("hall_onesided_synth.dat", "w") as f:
    f.write("\n".join(lines) + "\n")
