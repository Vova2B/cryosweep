# CLAUDE.md — cryosweep

Guidance for AI coding assistants (and humans) working in this repository.

## What this is

`cryosweep` analyses Quantum Design PPMS/MPMS measurement files (`.dat`) across seven probes:
VSM magnetometry, AC susceptibility, thermal transport, resistivity, heat capacity, and Hall
(field-sweep and temperature-dependent). It loads a file, detects the probe, segments the sweeps,
fits physical models, and emits tidy CSVs, publication-quality figures and a machine-readable
JSON envelope.

| Module | Role |
|---|---|
| `cryosweep_core/` | Analysis, fitting, plotting, IO. **Imports no Qt** — that is an invariant, not an accident. |
| `cryosweep_cli/` | Headless CLI. One JSON object per command on stdout; logs to stderr. |
| `cryosweep_gui/` | PySide6 desktop app. A thin shell over the core. |
| `tests/core`, `tests/gui` | Two independent suites, each with its own `conftest.py`. |
| `examples/` | One runnable `.dat` per scenario. |
| `tools/` | Development utilities; not part of the installed package. |
| `docs/physics-reference.md` | The formulas. New physics goes here. |
| `skill/` | Agent-facing guide to driving the CLI unattended. |

## Architectural invariants

**The core stays Qt-free.** `cryosweep_core` must be importable and fully testable with no GUI
toolkit present. If analysis code needs to talk to the GUI, invert the dependency.

**Capabilities are gated, never assumed.** Results carry `data.capabilities`: a list of
`{name, applicable, reason}`. `applicable: false` covers both "a required input is missing" and
"recognized but not yet implemented". Never infer that an analysis ran — read the list.

**Missing inputs gate; they do not guess.** When a file lacks something a fit requires (molar
mass, sample mass, thickness), the result is `status: "gated"` with a `gate[]` entry naming the
flag and an example. Inventing a plausible default is worse than declining.

## The physics-integrity rules

These are the point of the project. Treat them as load-bearing.

**Decline rather than fabricate.** A fitted parameter pinned at a search bound is not a
measurement, and neither is one whose σ exceeds its own magnitude. Where that happens the fit
reports *nothing* — blank CSV cells, `null` in JSON — plus a machine-readable flag saying why.
Reporting the bound as if it were a result is the failure mode this discipline exists to prevent.

**A spread is not an error bar.** Fits are re-run across a ladder of fit windows, and the
`max−min` spread across rungs is reported *next to* the statistical σ, never merged with it. On
real data the window frequently moves a parameter by far more than σ does. Quote both, and label
which is which.

**Say what was measured, not what would be convenient.** A maximum at the edge of the measured
range is not an observed peak, and is flagged as such. A warning that fires on real data is
usually correct; suppressing it is not a fix.

**Never present a hand-set value as a fit.** If a user edits model parameters, whatever is drawn
is a model, not a fit, and every surface — figure, legend, export — must keep them distinct.

## Running it

```bash
pip install -e .                                   # editable install
QT_QPA_PLATFORM=offscreen pytest                   # both suites, headless
cryosweep analyze examples/magnetization_vsm.dat   # or: python -m cryosweep_cli
```

**Around 200 tests skip in a fresh checkout. This is by design — please do not "fix" it.**
Those tests exercise real measurement files that are not distributed (they carry sample
identities), and they resolve their inputs through a local-only map. Absent input is a *skip*,
never a failure: see the docstrings in `tests/core/conftest.py`. A green run with ~200 skips is
the expected result.

## Conventions

- **Conventional commits** (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`). Plain messages.
- **Tests first.** For a behaviour change, write the test and watch it fail before fixing it. A
  test written afterwards proves the code does what it does, not what it should.
- **Never commit a `.dat`** beyond `tests/core/fixtures/` and `examples/`, both of which are
  synthetic and generated. `.gitignore` enforces this; do not weaken it.
- **Helper duplication between the two test suites is deliberate.** `tests/core` and `tests/gui`
  are independent packages and keep their helpers module-local. Do not "DRY" them together.
- **Physics belongs in `docs/physics-reference.md`**, with the formula, not only in code.
- Verify claims about behaviour by running the code, not by reading it. Where a figure is the
  output, look at the figure.

## Contributing

Pull requests go to `main`. A one-time CLA is required — see [CLA.md](CLA.md) and
[CONTRIBUTING.md](CONTRIBUTING.md); the bot will prompt you, or you can comment:

> I have read the cryosweep CLA and I hereby sign the Individual CLA.

Known defects are in [KNOWN-ISSUES.md](KNOWN-ISSUES.md), each with what reproduces it; planned
work is in [ROADMAP.md](ROADMAP.md). Both are good places to find a first contribution.

Licensing questions: **info@cryosweep.org**. The licence is PolyForm Noncommercial 1.0.0 —
free for academic and other noncommercial use; commercial use requires a separate licence.
