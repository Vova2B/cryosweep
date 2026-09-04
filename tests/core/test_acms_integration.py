"""End-to-end ACMS integration: real He-3 file dispatch, gallery manifest, CLI molar flags.

Pins the real single-frequency AC file all the way through the registry/dispatch path
(status ok, 3 amplitude groups / 6 curves, both feature detectors decline) and verifies
the global --molar-mass/--mass-mg plumbing reaches the acms analyzer THROUGH the CLI
(not just via dataclasses.replace).
"""
import json
import pathlib
import subprocess
import sys

import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file

ROOT = pathlib.Path(__file__).resolve().parents[2]
from tests.core.conftest import repo_root

REPO = repo_root()   # the repo root (docs/, skill/ and the real data live there, not in the app folder)



def test_real_file_end_to_end_dispatch(acms_real_path):
    rt = load_dat(acms_real_path)
    r = analyze_file(rt, RunConfig(), build_default_registry())
    assert r.data["probe"] == "acms" and r.status == "ok"
    assert r.data["sc_transition"] is None                      # null SC search
    assert r.data["chi_dprime_peaks"] == []                     # featureless
    assert len({round(c["amplitude_oe"], 4) for c in r.data["curves"]}) == 3
    assert len(r.data["curves"]) == 6
    caps = {c["name"]: c["applicable"] for c in r.data["capabilities"]}
    assert caps["ac_susceptibility"] is True
    assert caps["superconducting_screening"] is False and caps["chi_dprime_peak"] is False
    # No sample mass/molar mass in the bare file -> molar ladder not applicable.
    assert caps["molar_normalization"] is False
    assert r.data["curves"][0]["chi_prime_molar"] is None
    # Dropped-group contract: the lone 0.4979 Oe point is reported, not silently eaten.
    assert any(round(g["amplitude_oe"], 4) == 0.4979 and g["n_points"] == 1
               and g["reason"] == "< 5 points" for g in r.data["dropped_groups"])


def test_manifest_has_acms_entries():
    from tests.core.conftest import require_manifest
    m = json.loads(require_manifest().read_text())
    ids = {e["id"] for e in m}
    assert {"acms_chi_t", "acms_chi_t_sc"} <= ids
    by_id = {e["id"]: e for e in m}
    # Real-file entry points at the local-only AC-susceptibility dat and the headline stacked
    # kind. The filename is resolved through the untracked map (no measurement name is spelled
    # in a tracked file); when the file is unavailable the manifest shape is still asserted.
    assert by_id["acms_chi_t"]["v2_kind"] == "acms_chi_t"
    from tests.core.conftest import real_data
    src = real_data("acms")
    if src is not None:
        assert by_id["acms_chi_t"]["dat"].endswith(src.name)
    # SC synth entry reuses the headline kind on the Tc-marker fixture.
    assert by_id["acms_chi_t_sc"]["v2_kind"] == "acms_chi_t"
    assert by_id["acms_chi_t_sc"]["dat"].endswith("acms_sc_synth.dat")


def test_sc_synth_exercises_tc_marker():
    """The acms_chi_t_sc gallery fixture must yield a real SC transition (Tc marker path)."""
    rt = load_dat("tests/core/fixtures/acms_sc_synth.dat")
    r = analyze_file(rt, RunConfig(), build_default_registry())
    assert r.status == "ok"
    sc = r.data["sc_transition"]
    assert sc is not None
    assert abs(sc["tc_mid_k"] - 5.0) < 0.1   # oracle Tc_mid 5.000 K


def test_cli_molar_flags_reach_acms(acms_real_path):
    r = subprocess.run([sys.executable, "-m", "cryosweep_cli", "analyze", str(acms_real_path),
                        "--molar-mass", "200.0", "--mass-mg", "5.0"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["data"]["probe"] == "acms"
    assert payload["data"]["curves"][0]["chi_prime_molar"] is not None
    caps = {c["name"]: c["applicable"] for c in payload["data"]["capabilities"]}
    assert caps["molar_normalization"] is True
