"""The anonymized real Hall example must keep the properties it ships FOR.

KNOWN-ISSUES 18/19/20 were real correctness bugs, and none reproduced on any shipped
example: the synthetic Hall files are field-symmetric and sit exactly on setpoint, so the
fixed paths had no public regression coverage. `hall_mixed_sweeps.dat` is a decimated,
anonymized subset of a real Hall-wired measurement chosen precisely because it is messy —
temperature setpoints that drift across the old round(·,1) bin edge, and temperatures
covered by a single ± field pair.

Naive decimation destroys exactly these properties (measured: a global every-6th-row
subset loses the ±40 kOe ramps to the segmenter entirely, and every-3rd loses one of
them), so the generator decimates structure-aware and THESE TESTS pin the payload — if a
regeneration thins the file differently, they fail rather than silently shipping a file
that no longer exercises anything.
"""
import numpy as np
import pathlib
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

EXAMPLE = pathlib.Path(__file__).resolve().parents[2] / "examples" / "hall_mixed_sweeps.dat"
_REG = build_default_registry()
_CACHE = {}


def _analyze(probe):
    if probe not in _CACHE:
        cfg = RunConfig.load(probe_override=probe)
        cfg.hall.hall_channel = 1
        cfg.hall.thickness_mm = 0.07
        cfg.hall.longitudinal_channel = 2
        _CACHE[probe] = analyze_file(load_dat(str(EXAMPLE)), cfg, _REG)
    return _CACHE[probe]


def _body_rows():
    lines = EXAMPLE.read_text(encoding="latin-1").splitlines()
    di = next(i for i, ln in enumerate(lines) if ln.strip() == "[Data]")
    import csv
    return [r for r in csv.reader(ln for ln in lines[di + 2:] if ln.strip())]


# ---------------- raw payload properties (independent of any analyzer) ----------------

def test_the_200k_loop_straddles_the_old_bin_edge():
    # item 19's trigger: setpoint drift ACROSS round(T, 1)'s 199.95 bin edge. Both
    # sub-populations must survive decimation, else the file no longer reproduces the
    # defect the fix is guarded against.
    T = np.array([float(r[3]) for r in _body_rows()])
    tt = T[np.abs(T - 200.0) < 0.5]
    lo, hi = tt[tt < 199.95], tt[tt >= 199.95]
    assert len(lo) >= 5 and len(hi) >= 5, (len(lo), len(hi))
    assert np.median(hi) - np.median(lo) > 0.05


def test_no_identity_token_survives():
    # This file's identity lived in its SOURCE FILENAME, not its header. The forbidden
    # tokens are therefore derived from the untracked source at runtime — hardcoding them
    # here would put the identity INTO the public tree, defeating the very anonymisation
    # this test verifies (the identity gate caught exactly that, 2026-09-05: a check must
    # not contain the thing it checks for). Off the owner machine the source map is
    # absent and the test skips, the established real-data convention.
    import re
    from tests.core.conftest import require_real
    src = require_real("hall")
    benign = {"resistivity", "option", "hall"}     # format words, legitimate in any file
    toks = {t.lower() for t in re.split(r"[^A-Za-z0-9]+", pathlib.Path(src).stem)
            if len(t) >= 4} - benign
    assert toks, "source filename yielded no tokens — this check would be vacuous"
    blob = EXAMPLE.read_text(encoding="latin-1").lower()
    leaked = sorted(t for t in toks if t in blob)
    assert not leaked, f"identity token(s) survive in the shipped example: {leaked}"


# ---------------- item 18: a single +-pair fits as antisym ----------------

def test_single_pair_temperatures_fit_as_antisym_with_full_confidence():
    r = _analyze("hall_tdep")
    assert r.status == "ok"
    assert r.confidence >= 0.9                      # was 0.0 before the fix (item 18)
    pts = r.data["points"]
    single = [p for p in pts if p["antisym_points"] == 1]
    assert len(single) >= 80, len(single)
    assert all(p["r_h_method"] == "antisym" for p in single)
    # the honest fallback still exists and is labeled: unpaired coverage -> 2point + low conf
    two_pt = [p for p in pts if p["r_h_method"] == "2point"]
    assert all(p["low_confidence"] for p in two_pt)


def test_all_five_field_curves_survive_decimation():
    # the +-40 kOe ramps are what naive decimation loses first (measured); with them gone
    # the two-pair temperatures vanish and item 18's single-pair path is all that is left
    r = _analyze("hall_tdep")
    fields = sorted(c["field_oe"] for c in r.data["interp_curves"])
    assert fields == [-90000.0, -40000.0, 0.0, 40000.0, 90000.0]
    assert any(p["antisym_points"] == 2 for p in r.data["points"])


# ---------------- item 19: setpoint clustering fabricates no phantom ----------------

def test_drifting_200k_setpoint_is_one_group_not_two():
    r = _analyze("hall")
    pts = r.data["points"]
    near200 = [p for p in pts if 199.0 < p["temperature"] < 201.0]
    assert len(near200) == 1, [p["temperature"] for p in near200]
    temps = sorted(p["temperature"] for p in pts)
    assert len(temps) == 9                          # 2/5/10/15/20/50/100/200/300 K
    close = [(a, b) for a, b in zip(temps, temps[1:]) if b - a < 0.15]
    assert not close, f"phantom setpoint pair(s): {close}"


def test_tdep_grid_has_no_phantom_temperatures():
    pts = _analyze("hall_tdep").data["points"]
    temps = sorted(p["temperature"] for p in pts)
    close = [(a, b) for a, b in zip(temps, temps[1:]) if b - a < 0.15]
    assert not close, f"phantom grid pair(s): {close}"


# ---------------- item 20: nothing derived without the R_H it derives from ----------------

@pytest.mark.parametrize("probe", ["hall", "hall_tdep"])
def test_no_carrier_density_without_a_hall_coefficient(probe):
    # the pre-fix failure mode was carrier_n = 1.06e30 published at a temperature whose
    # R_H was None. The pure decline path is pinned by the fix's own unit tests; this file
    # pins the joint invariant that was actually violated on real data.
    for p in _analyze(probe).data["points"]:
        if p.get("carrier_n") is not None:
            assert p.get("R_H") is not None, p["temperature"]


# ---------------- the geometry-unset warning, first public reproducer ----------------

def test_geometry_unset_warning_fires_on_the_resistivity_probe():
    r = _analyze("resistivity")
    assert any("geometry" in w.lower() for w in r.warnings), r.warnings
