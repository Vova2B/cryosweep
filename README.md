# cryosweep

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22307209.svg)](https://doi.org/10.5281/zenodo.22307209)

Analysis of Quantum Design **PPMS** and **MPMS** measurement files (`.dat`) — a Qt-free
analysis core with a JSON contract, a command-line interface, and a desktop GUI built on the
same analyzers.

One file in, physics out: cryosweep detects which measurement it is looking at, separates the
sweeps, fits the appropriate models, and reports what it found — including when it *cannot*
report something, and why.

![The cryosweep heat-capacity tab: low-temperature Cp/T models, full-range Debye-Einstein fit, entropy S(T), and per-field gamma and Debye temperature](https://raw.githubusercontent.com/Vova2B/cryosweep/main/docs/images/heat-capacity-multifield.png)

*That last clause is the point. In the status bar above, cryosweep reports θ_D drifting 160 %
across fields — the lattice should be field-independent — and a Sommerfeld coefficient γ that has
gone negative. Neither is handed back as a result; both are flagged as physics that does not hold.*

| Probe | What it fits |
|---|---|
| **Magnetization (VSM)** — QD + MPMS formats | χ, 1/χ, Curie-Weiss (θ, C, μ_eff) with a window-sensitivity ladder |
| **AC susceptibility (ACMS)** | χ′/χ″, superconducting screening step, T_c, χ″ peak / freezing temperature |
| **Thermal transport (TTO)** | κ, Seebeck, ZT, Wiedemann-Franz decomposition, Lorenz ratio, κ_ph power law |
| **Resistivity** — QD Resistivity + AC Transport | ρ(T), RRR ± σ, magnetoresistance, resistive T_c, power-law / Fermi-liquid fit |
| **Heat capacity** | Debye-Einstein Cp(T), low-T Cp/T vs T², spin-fluctuation models, entropy S(T), Schottky |
| **Hall effect** | Antisymmetrized R_xy, R_H, carrier density, mobility, R_H(T) |

The physics behind each — the models, the fitted quantities, and the criteria the detectors
use — is documented in [`docs/physics-reference.md`](https://github.com/Vova2B/cryosweep/blob/main/docs/physics-reference.md).

## Install

```bash
pip install cryosweep
```

That gives you the analysis core and the `cryosweep` command line. To add the desktop app:

```bash
pip install 'cryosweep[gui]'
```

The GUI is optional because the analysis core and CLI are Qt-free by design, and Qt is by far
the heaviest dependency here — leaving it out keeps an agent or CI install a quarter of the size.
Installing without it still gives you every analyzer; only `cryosweep-gui` needs the extra, and it
says so if you run it.

From a clone, for development:

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[gui]'
```

Python ≥ 3.11 (developed and tested on 3.14).

## Quickstart

The [`examples/`](https://github.com/Vova2B/cryosweep/tree/main/examples) folder has a runnable file per probe, so you can try
everything before pointing it at your own data ([`examples/README.md`](https://github.com/Vova2B/cryosweep/blob/main/examples/README.md)
says what each file shows and which tab/inputs to use — the Hall examples in particular
open on the Resistivity tab first, because Hall measurements share the resistivity file
format).

Those files ship with the repository rather than the wheel, so after a `pip install` fetch one
first — or clone the repo and run the commands as written:

```bash
curl -O https://raw.githubusercontent.com/Vova2B/cryosweep/main/examples/heat_capacity.dat
cryosweep analyze heat_capacity.dat
```

```bash
cryosweep analyze examples/magnetization_vsm.dat
cryosweep analyze examples/heat_capacity.dat
cryosweep analyze examples/thermal_transport.dat
cryosweep report examples/resistivity_superconductor.dat
cryosweep-gui
```

The first fits Curie-Weiss and reports θ = −10 K; `report` prints a Markdown summary instead
of JSON; `cryosweep-gui` opens the desktop app.

Some measurements need inputs the file does not carry. MPMS files hold no molar mass or sample
mass, so the analyzer **gates** rather than guessing — supply them and it proceeds:

```bash
cryosweep analyze examples/magnetization_mpms.dat --molar-mass 200 --mass-mg 10
```

Hall measurements use the same file format as ordinary resistivity — only the wiring differs —
so the Hall analyzer is invoked explicitly, with the channel and the sample thickness:

```bash
cryosweep hall examples/hall_field_sweeps.dat \
    --hall-channel 1 --thickness 0.5 --thickness-unit mm --long-channel 2
```

which reports `R_H = -2.500e-07 m^3/C`.

## Built to be driven by a program

Every command prints **one JSON object** on stdout (logs go to stderr) and sets a meaningful
exit code, so cryosweep is usable from a script, a pipeline, or an LLM agent without screen
scraping:

| Command | What it gives you |
|---|---|
| `cryosweep probes` | available measurement types and what each one needs |
| `cryosweep schema analyze:vsm` | JSON Schema for the result shape |
| `cryosweep export <file> --out result` | tidy long-format CSVs, units in the headers |
| `cryosweep run pipeline.json` | batch |

Exit codes distinguish *no result* from *a result you should look at*: `0` ok, `10` gated (a
required input is missing — the payload names the flag), `11` low confidence, `2` bad input.
Codes 10 and 11 still print a full JSON envelope. Output is byte-stable for the same input, so
it diffs and caches cleanly. A ready-made agent guide ships in
[`skill/cryosweep/SKILL.md`](https://github.com/Vova2B/cryosweep/blob/main/skill/cryosweep/SKILL.md).

## Example data

Everything in `examples/` is written by `tools/make_examples.py`.

**Eight of the ten files are synthetic**, generated by the same builders that produce the test
fixtures. The numbers are physically consistent and each file places a feature where it
exercises the relevant analyzer — but they are **not measurements of any real material**.

**Two are anonymized subsets of real measurements** — `magnetization_vsm_multifield.dat` and
`heat_capacity_multifield.dat` — because no synthetic file reproduces what real multi-field
data does to the segmentation and window-selection paths. They are decimated, and everything
that identified the sample, the operator or the instrument has been replaced: sample material
and comment, all five instrument serial numbers, the acquisition date (the absolute
`Time Stamp` column is rebased to zero, since it decodes to the measurement instant), the
calibration free text in the `Comment` column, and the formula weight and sample mass, which
are neutral values chosen only so the fits report a plausible moment. The generators carry an
`assert_no_identity_leak` post-condition that refuses to write the file if any token of the
source identity survives. The measured *shapes* are real; the sample metadata is not, and **no
scientific conclusion should be drawn from any file here**.

Regenerate with (the two real-derived files are skipped, and left untouched, on any machine
without the private source data):

```bash
.venv/bin/python tools/make_examples.py
```

## Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest
```

Run it from the repository root — some tests build fixture paths relative to the working
directory and launch the CLI as a subprocess.

The suite ships with synthetic `.dat` fixtures only. Tests that need real measurement files, or
the maintainer's local reference gallery, **skip** when those are absent, so a fresh clone is
green with no data at all. To exercise the optional real-data tests against your own files,
create `real_data_map.json` in the repository root mapping logical keys to paths.

## File format and provenance

The `.dat` files this program reads are produced by Quantum Design instrument software and
retain their original `; Copyright …, Quantum Design, Inc.` header lines; those are part of the
format and are preserved rather than stripped. The format is **read**, not redistributed.

This project is not affiliated with, endorsed by, or supported by Quantum Design, Inc. "PPMS"
and "MPMS" are used only to identify the instruments and file formats cryosweep is compatible
with. PPMS is a registered trademark of Quantum Design, Inc.

## Licence

cryosweep is released under the [PolyForm Noncommercial License 1.0.0](https://github.com/Vova2B/cryosweep/blob/main/LICENSE).

**Free for research and education** — universities, public research institutes, national
laboratories, government and nonprofit organizations, and individuals, *regardless of how the
work is funded*. Use it, modify it, redistribute it, publish results with it.

**Commercial use requires a licence.** If you are a company, see
[COMMERCIAL.md](https://github.com/Vova2B/cryosweep/blob/main/COMMERCIAL.md) — it is a short email to `info@cryosweep.org`.

Third-party dependency licences, and what the LGPL requires of the Qt binding, are recorded in
[THIRD-PARTY-LICENSES.md](https://github.com/Vova2B/cryosweep/blob/main/THIRD-PARTY-LICENSES.md).

## Known issues

Known defects are listed in [KNOWN-ISSUES.md](https://github.com/Vova2B/cryosweep/blob/main/KNOWN-ISSUES.md). Most are display or
ergonomics issues found by inspecting every rendered example, and each of those names the
example file that reproduces it; none of them changes a fitted number or an exported value.
Two exceptions are called out explicitly there — a temperature-setpoint binning bug in the
Hall analyzer that can fabricate a carrier-density point, and the derived-quantity fallback
that lets it reach the CSV. Both were found on real data and reproduce on no shipped example.

## Versioning

Semantic versioning. While the version is **0.x** the interfaces are not frozen: the CLI JSON
envelope, the exit codes and the CSV columns may change between releases. Tagging **1.0.0** is
what freezes them.

Plot appearance, GUI layout and the internal Python API are not covered by that promise at any
version — figures are expected to improve, and doing so is not a breaking change.

## Project documents

- [CHANGELOG.md](https://github.com/Vova2B/cryosweep/blob/main/CHANGELOG.md) — what is in each release
- [ROADMAP.md](https://github.com/Vova2B/cryosweep/blob/main/ROADMAP.md) — what is planned for 1.0, with measured costs where they were measured
- [KNOWN-ISSUES.md](https://github.com/Vova2B/cryosweep/blob/main/KNOWN-ISSUES.md) — known defects found and deferred, with what reproduces each
- [CONTRIBUTING.md](https://github.com/Vova2B/cryosweep/blob/main/CONTRIBUTING.md) and [CLA.md](https://github.com/Vova2B/cryosweep/blob/main/CLA.md) — how to contribute, and the one-time agreement
- [SECURITY.md](https://github.com/Vova2B/cryosweep/blob/main/SECURITY.md) — reporting a vulnerability
- [CODE_OF_CONDUCT.md](https://github.com/Vova2B/cryosweep/blob/main/CODE_OF_CONDUCT.md)

## Contributing

Bug reports, files that break the loader, and pull requests are welcome — see
[CONTRIBUTING.md](https://github.com/Vova2B/cryosweep/blob/main/CONTRIBUTING.md). Because the project is dual-licensed, a pull request needs a
one-time [Contributor License Agreement](https://github.com/Vova2B/cryosweep/blob/main/CLA.md); you keep your copyright, and a bot walks you
through it on your first PR.

## Citing

If cryosweep contributes to work you publish, please cite it — see
[CITATION.cff](https://github.com/Vova2B/cryosweep/blob/main/CITATION.cff).
