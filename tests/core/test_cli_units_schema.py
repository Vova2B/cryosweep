"""Slice 3, Bugs 3 & 4: inv_chi unit strings + strengthened schema round-trip.

Validates a REAL `cryosweep analyze` ok payload against the analyze:vsm schema
(VSMData) and the full envelope against the result schema (Result), replacing
the old key-presence-only check.
"""
import json
import subprocess
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIX = str(ROOT / "tests/core/fixtures/vsm_synth.dat")


def _run(*args):
    return subprocess.run([sys.executable, "-m", "cryosweep_cli", *args],
                          capture_output=True, text=True, cwd=ROOT)


# --- Bug 3: inv_chi_unit strings ---

def test_si_inv_chi_unit_is_well_formed():
    r = _run("analyze", FIX, "--unit-system", "SI")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)["data"]
    assert data["inv_chi_unit"] == "mol/m^3"


def test_cgs_inv_chi_unit_unchanged():
    r = _run("analyze", FIX, "--unit-system", "CGS")
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)["data"]
    assert data["inv_chi_unit"] == "mol*Oe/emu"


# --- Strengthened: real ok payload round-trips through both models ---

def test_ok_payload_validates_against_schemas():
    from cryosweep_core.analyzers.mag import VSMData
    from cryosweep_core.result import Result
    r = _run("analyze", FIX)
    assert r.returncode == 0, r.stderr
    env = json.loads(r.stdout)
    assert env["status"] == "ok"
    Result(**env)                       # full envelope -> result schema
    vd = VSMData(**env["data"])         # data -> analyze:vsm schema
    assert vd.fit is not None
    assert len(vd.temperature) == len(vd.inv_chi) >= 3
