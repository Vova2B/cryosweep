import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")   # MUST be set before QApplication
import pathlib
import pytest
from PySide6.QtWidgets import QApplication

FIX = pathlib.Path(__file__).resolve().parent.parent / "core" / "fixtures"

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app

@pytest.fixture
def vsm_path():
    return FIX / "vsm_synth.dat"

@pytest.fixture
def hall_path():
    return FIX / "hall_synth.dat"

@pytest.fixture
def hc_synth_path():
    return FIX / "hc_synth.dat"

@pytest.fixture
def hall_long_synth_path():
    return FIX / "hall_long_synth.dat"

@pytest.fixture(autouse=True)
def _isolate_preset_store(tmp_path, monkeypatch):
    import cryosweep_gui.presets_io as pio
    monkeypatch.setattr(pio, "default_store_path", lambda: tmp_path / "presets.json")
    # BLOCKER B3: default_store_path alone is NOT enough once a legacy fallback exists. The
    # patched new path is absent in every GUI test, so an unpatched legacy_store_path would
    # send all 20+ MainWindow-building tests to the developer's real
    # ~/.ppms_v2_plot_presets.json — measured: 5 failures, and a suite that is a function of
    # $HOME (green here, red in CI, able to mask a real regression).
    monkeypatch.setattr(pio, "legacy_store_path", lambda: tmp_path / "legacy-absent.json")
    assert not (tmp_path / "presets.json").exists()
    assert not (tmp_path / "legacy-absent.json").exists()
    yield

_HALL_TDEP_SYNTH = FIX / "hall_tdep_synth.dat"


@pytest.fixture
def hall_tdep_synth_path():
    assert _HALL_TDEP_SYNTH.exists(), _HALL_TDEP_SYNTH
    return _HALL_TDEP_SYNTH


# Same logical-key indirection as tests/core/conftest.py. Duplicated rather than imported:
# the two suites are independent packages and this project keeps helpers module-local.
#: Local-only development markers; see tools/real_data.py for the same list and rationale.
#: NOT a `.git` walk — cryosweep is its own repo nested in the dev tree, so the nearest
#: `.git` is the app's own and every key would silently resolve to None.
_DEV_MARKERS = ("real_data_map.json", "docs/superpowers/pq-reference-gallery")


def _real(key):
    import json
    p = pathlib.Path(__file__).resolve()
    root = next((d for d in (p, *p.parents)
                 if any((d / m).exists() for m in _DEV_MARKERS)), p.parents[2])
    m = root / "real_data_map.json"
    rel = ""
    if m.exists():
        try:
            rel = json.loads(m.read_text()).get(key) or ""
        except (ValueError, OSError):
            rel = ""
    f = (root / rel) if rel else None
    return f if (f is not None and f.exists()) else None


def require_real(key):
    f = _real(key)
    if f is None:
        pytest.skip(f"local-only measurement file for key {key!r} is not available")
    return f


@pytest.fixture
def hall_real_path():
    return require_real("hall")


@pytest.fixture
def vsm_real_path():
    return require_real("vsm")


@pytest.fixture
def tto_real_path():
    return require_real("tto")


@pytest.fixture
def hc_path():
    return require_real("hc")
