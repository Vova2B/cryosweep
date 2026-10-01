# CryoSweep Analysis Report

- **Probe:** vsm
- **Status:** ok
- **Confidence:** 1.000

## Fit — Curie Weiss (`curie_weiss`)

| param | value | ± σ | unit |
|---|---|---|---|
| C | 0.5 | 6.5e-06 | emu*K/(mol*Oe) |
| theta | -9.997 | 0.0024 | K |
| mu_eff | 1.999 | 1.3e-05 | mu_B |

R² = 1.00000, n = 300

## Fit — Modified Curie-Weiss (`curie_weiss_modified`)

| param | value | ± σ | unit |
|---|---|---|---|
| C | 0.5 | 9e-06 | emu*K/(mol*Oe) |
| theta | -10 | 0.00028 | K |
| chi0 | declined |  | emu/(mol*Oe) |
| mu_eff | 1.999 | 1.8e-05 | mu_B |

R² = 1.00000, n = 300, fitted 2–300 K
Flags: chi0_unresolved

## Warnings

- CW fit window extends below |theta| (T_min = 2.0 K < 10.0 K) — low-T rows likely outside the paramagnetic regime; see cw_ladder