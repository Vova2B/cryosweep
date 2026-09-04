# Third-party licences

cryosweep itself is licensed under the [PolyForm Noncommercial License 1.0.0](LICENSE).
It does **not** bundle or redistribute any of the packages below — they are declared as
dependencies in `pyproject.toml` and installed by pip from PyPI, each under its own
licence. This file records what those licences are and what they require.

| Package | Licence | Notes |
|---|---|---|
| [numpy](https://numpy.org) | BSD-3-Clause (with vendored 0BSD / MIT / Zlib components) | Permissive |
| [pandas](https://pandas.pydata.org) | BSD-3-Clause | Permissive |
| [scipy](https://scipy.org) | BSD-3-Clause | Permissive |
| [pydantic](https://docs.pydantic.dev) | MIT | Permissive |
| [matplotlib](https://matplotlib.org) | Matplotlib licence (PSF-based, BSD-compatible) | Permissive |
| [PySide6 / Qt for Python](https://doc.qt.io/qtforpython/) | **LGPL-3.0-only** (also available under GPL-2.0 / GPL-3.0, or a commercial Qt licence) | See below |

All of these are compatible with distributing cryosweep under a noncommercial
source-available licence. None of them imposes copyleft on cryosweep's own source.

## PySide6 and the LGPL — why this matters

The GUI uses **PySide6 (Qt for Python)** under the **LGPL-3.0**. This choice is
deliberate and load-bearing: the alternative Python binding, PyQt6, is offered only
under the GPL or a paid commercial licence, and the GPL option would force cryosweep's
own source to be GPL — which is incompatible with the licensing model here.

The LGPL permits combining a work under a licence of your choosing with the library,
provided the recipient can replace the LGPL-licensed library with their own version.
cryosweep satisfies this by construction:

- it is distributed **as source**, and PySide6 is installed separately by pip;
- nothing is statically linked, and no Qt libraries are bundled or modified;
- any user can upgrade, patch, or substitute their own PySide6 build at any time.

The analysis core (`cryosweep_core`) imports no Qt at all — a property enforced by a
test (`tests/core/test_qt_free.py`) — so the CLI and the library are usable with no Qt
present on the system.

**If cryosweep is ever redistributed as a frozen binary** (PyInstaller, Nuitka, an app
bundle, or any single-file executable), the LGPL relinking obligation stops being
satisfied automatically and must be addressed explicitly: ship the Qt libraries as
separate shared objects the user can replace, include the LGPL text and the required
notices, and offer the corresponding source for the Qt components. Do not freeze a
release without handling this.

## Obtaining the licence texts

Each package ships its own licence in its distribution:

```bash
pip show -f <package>          # locate the installed package
python -c "import importlib.metadata as m; print(m.metadata('<package>')['License'])"
```

Canonical texts: [BSD-3-Clause](https://opensource.org/license/bsd-3-clause),
[MIT](https://opensource.org/license/mit),
[LGPL-3.0](https://www.gnu.org/licenses/lgpl-3.0.html),
[matplotlib](https://matplotlib.org/stable/project/license.html).

## Not a dependency: Quantum Design

cryosweep reads the `.dat` files produced by Quantum Design PPMS and MPMS instruments.
It contains no Quantum Design code and is not affiliated with, endorsed by, or a
product of Quantum Design, Inc. "PPMS" and "MPMS" are used only to describe the file
formats and instruments the software is compatible with. PPMS is a registered
trademark of Quantum Design, Inc.
