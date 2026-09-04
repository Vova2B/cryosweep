"""DC-mode ACMS files: recover the magnetisation the ACMS probe cannot use.

Real ACMS files are sometimes recorded with the AC drive never engaged: the instrument
still writes the full ACMS column set, but Frequency / Amplitude / M' / M'' are EMPTY in
every row while M-DC and M-Std.Dev. carry a perfectly good M(T) or M(B) sweep. The ACMS
analyzer correctly declines (`gated`, need="ac_data"), and the file's data was then
discarded entirely -- on the owner's two real files, 2503 usable rows.

Two mechanisms, tested separately:
  * mag.py resolves its moment column from `moment` OR, when that is absent or unusable,
    `m_dc`. The "or unusable" half is load-bearing: `Moment (emu)` is PRESENT in these
    files' headers and empty in every row, so an `absent`-only test never fires and the
    analyzer reports "no temperature sweep found" forever.
  * dispatch reroutes to the runner-up probe when the winner gates on something the user
    cannot supply (an empty `remedy`), and says so in the result.
"""
from __future__ import annotations

import pathlib

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.detect.probe import detect_probe, detect_probe_ranked

FIX = pathlib.Path(__file__).parent / "fixtures"
DCONLY = FIX / "acms_dconly_synth.dat"


def _analyze(**cfg_kw):
    rt = load_dat(str(DCONLY))
    import dataclasses

    if cfg_kw.get("molar_mass") or cfg_kw.get("mass_mg"):
        rt = dataclasses.replace(rt, header=dataclasses.replace(
            rt.header, molar_mass=cfg_kw.get("molar_mass"), mass_mg=cfg_kw.get("mass_mg")))
    return analyze_file(rt, RunConfig.load(), build_default_registry())


def test_file_still_detects_as_acms():
    """The reroute must not be achieved by breaking detection -- the file IS an ACMS file."""
    rt = load_dat(str(DCONLY))
    score, key = detect_probe(rt.header, set(rt.df.columns), build_default_registry())
    assert key == "acms" and score >= 0.5, (key, score)


def test_ranked_detection_exposes_the_runner_up():
    rt = load_dat(str(DCONLY))
    ranked = detect_probe_ranked(rt.header, set(rt.df.columns), build_default_registry())
    keys = [k for _s, k in ranked]
    assert keys[0] == "acms"
    assert "vsm" in keys[1:], keys


def test_reroutes_to_vsm_and_asks_for_the_inputs_it_needs():
    """Before: gated on need='ac_data' with an empty remedy -- a dead end.
    After: gated on molar_mass / sample_mass, which the user CAN supply."""
    r = _analyze()
    needs = {g.need for g in (r.gate or [])}
    assert "ac_data" not in needs, needs
    assert needs == {"molar_mass", "sample_mass"}, needs
    assert r.data.get("probe") == "vsm"
    assert any(g.remedy for g in r.gate), "a reroute must land on user-fixable gates"


def test_reroute_is_reported_never_silent():
    r = _analyze()
    blob = " ".join(r.warnings or [])
    assert "acms" in blob and "vsm" in blob, r.warnings
    assert r.data.get("rerouted_from") == "acms", r.data.get("rerouted_from")


def test_with_inputs_supplied_it_analyses_the_dc_moment():
    r = _analyze(molar_mass=200.0, mass_mg=5.0)
    assert r.status in ("ok", "low_confidence"), (r.status, r.gate, r.errors)
    assert r.data.get("moment_source") == "m_dc", r.data.get("moment_source")
    blocks = r.data.get("t_blocks") or r.data.get("loops") or []
    assert blocks, f"no sweep recovered from 120 DC rows: {list(r.data)}"


def test_healthy_vsm_file_is_untouched():
    """The fallback must be invisible to a file whose `moment` column is usable."""
    import glob

    cands = sorted(glob.glob(str(FIX / "*vsm*.dat"))) or sorted(glob.glob(str(FIX / "*mpms*.dat")))
    if not cands:
        return  # no vsm fixture in this tree; covered by the gallery gate instead
    rt = load_dat(cands[0])
    r = analyze_file(rt, RunConfig.load(), build_default_registry())
    assert r.data.get("rerouted_from") is None
    assert r.data.get("moment_source") in (None, "moment"), r.data.get("moment_source")
