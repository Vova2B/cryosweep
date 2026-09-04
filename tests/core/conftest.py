import json
import pathlib
import pytest

# Prime pyplot's backend once, at session start (conftest imports before any test module).
# Tests that render via FigureCanvasAgg never populate pyplot's cached `_get_backend_mod`, so
# whichever test creates the *first* pyplot figure triggers a lazy switch_backend — which under
# Python 3.14 + this matplotlib can fail to resolve `matplotlib.backend_bases`. Creating and
# closing one real pyplot figure here caches the backend successfully up front, making the suite
# order-independent (previously the failure surfaced only when an early plot test deferred the
# first pyplot figure to a downstream test).
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as _plt  # noqa: E402
_plt.close(_plt.figure())


#: Local-only development markers; see tools/real_data.py for the same list and rationale.
DEV_MARKERS = ("real_data_map.json", "docs/superpowers/pq-reference-gallery")


def repo_root(start=__file__):
    """The dev-tree root: the nearest ancestor holding a local-only development marker.

    Not parents[N] — this file moves in the two-app split, and not a `.git` walk either:
    cryosweep is its own repository nested inside the private dev tree, so the nearest
    `.git` is cryosweep's own while the markers sit above it. A `.git` walk therefore
    returns the app directory, and real_data()/manifest_path() below resolve to None for
    every key — 200+ tests skipping while the suite still exits 0. A public checkout has
    no marker at any level and resolves None by design: that is the skip-not-fail path.
    """
    p = pathlib.Path(start).resolve()
    for d in (p, *p.parents):
        if any((d / m).exists() for m in DEV_MARKERS):
            return d
    return p.parents[2]


def real_data(key):
    """Resolve a local-only measurement file by logical key. None when unavailable.

    The map (real_data_map.json at the repo root) is gitignored, so a public checkout resolves
    every key to None and every real-data test skips. This is the ONLY indirection: no
    measurement filename appears in any tracked file (spec §2c/§2e/§2f).
    """
    root = repo_root()
    m = root / "real_data_map.json"
    if not m.exists():
        return None
    try:
        rel = json.loads(m.read_text()).get(key)
    except (ValueError, OSError):
        return None
    if not rel:
        return None
    p = root / rel
    return p if p.exists() else None


def manifest_path():
    """The PQ reference-gallery manifest, or None when it is not present.

    It lives at the REPO root, OUTSIDE the app directory, and it can never ship: it spells
    seven real measurement filenames and points at copyrighted journal figures. So the
    published repo (the app directory alone) simply does not have it, and every test that
    reads it must skip rather than fail -- the same rule as an absent local-only data file.
    """
    p = repo_root() / "docs/superpowers/pq-reference-gallery/manifest.json"
    return p if p.exists() else None


def require_manifest():
    """Return the manifest path, or skip. Never assert -- an absent gallery manifest is a
    skip, not a failure, for the same reason as require_real (spec §2a)."""
    p = manifest_path()
    if p is None:
        pytest.skip("gallery manifest is not available (it is not part of the published tree)")
    return p


def require_real(key):
    """Return the file for `key`, or skip the test. Never assert — an absent local-only file
    is a skip, not a failure (spec §2a)."""
    p = real_data(key)
    if p is None:
        pytest.skip(f"local-only measurement file for key {key!r} is not available")
    return p


FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
HALL_SYNTH = FIX / "hall_synth.dat"
HC_SYNTH = FIX / "hc_synth.dat"
HALL_LONG_SYNTH = FIX / "hall_long_synth.dat"
ACT_SYNTH = FIX / "act_synth.dat"
MPMS_SYNTH = FIX / "mpms_synth.dat"
HALL_TDEP_SYNTH = FIX / "hall_tdep_synth.dat"


# --- real-data fixtures: every one skips when the file is unavailable ---

@pytest.fixture
def hc_path():
    return require_real("hc")

@pytest.fixture
def res_path():
    return require_real("res")

@pytest.fixture
def hall_real_path():
    return require_real("hall")

@pytest.fixture
def act_real_path():
    return require_real("act")

@pytest.fixture
def mpms_real_path():
    return require_real("mpms")

@pytest.fixture
def tto_real_path():
    return require_real("tto")

@pytest.fixture
def acms_real_path():
    return require_real("acms")

@pytest.fixture
def vsm_real_path():
    return require_real("vsm")

@pytest.fixture
def dc_rho_path():
    return require_real("dc_rho")

@pytest.fixture
def hc_fields_path():
    return require_real("hc_fields")

@pytest.fixture
def hc_lowmass_path():
    return require_real("hc_lowmass")

@pytest.fixture
def hc_lowt_path():
    return require_real("hc_lowt")


# --- committed synthetic fixtures: assert, they must always be there ---

@pytest.fixture
def hall_synth_path():
    assert HALL_SYNTH.exists(), HALL_SYNTH
    return HALL_SYNTH

@pytest.fixture
def hall_long_synth_path():
    assert HALL_LONG_SYNTH.exists(), HALL_LONG_SYNTH
    return HALL_LONG_SYNTH

@pytest.fixture
def hc_synth_path():
    assert HC_SYNTH.exists(), HC_SYNTH
    return HC_SYNTH

@pytest.fixture
def act_synth_path():
    assert ACT_SYNTH.exists(), ACT_SYNTH
    return ACT_SYNTH

@pytest.fixture
def mpms_synth_path():
    assert MPMS_SYNTH.exists(), MPMS_SYNTH
    return MPMS_SYNTH

@pytest.fixture
def hall_tdep_synth_path():
    assert HALL_TDEP_SYNTH.exists(), HALL_TDEP_SYNTH
    return HALL_TDEP_SYNTH
