"""Where did the molar mass come from -- the file, or a person?

mu_eff = 2.827*sqrt(C) and C scales with the molar mass, so a molar mass someone typed
propagates straight into a reported moment. Both entry points (the CLI's --molar-mass and
the GUI's box) patched it onto HeaderMeta, after which it was indistinguishable from a value
the instrument wrote. The result recorded neither the number nor its origin, so a mu_eff
could not be checked against the input that produced it.

That is not hypothetical: this project's own handoff quoted mu_eff = 3.211 for a file whose
header says 386 g/mol. The 3.211 came from running the documentation's PLACEHOLDER
`--molar-mass 200`. It survived review because theta and r2 are invariant under a rescaling
of chi -- the two numbers a reader would cross-check are exactly the two that cannot
disagree, and the one number that carried the error was recorded nowhere.

These tests pin value AND source, at every site that can supply one.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.io.header import apply_sample_inputs, sample_input_provenance
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry

ROOT = pathlib.Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
VSM = EXAMPLES / "magnetization_vsm.dat"        # header carries MOLWGHT 200.0 / MASS 5.0
MPMS = EXAMPLES / "magnetization_mpms.dat"      # header carries neither


def _analyze(rt):
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())


def test_a_value_the_instrument_wrote_is_marked_as_coming_from_the_file():
    prov = sample_input_provenance(load_dat(str(VSM)).header)
    assert prov["molar_mass"] == {"value": 200.0, "source": "header"}


def test_a_value_a_person_supplied_is_marked_as_such():
    rt = apply_sample_inputs(load_dat(str(MPMS)), {"molar_mass": 386.0, "mass_mg": 1.1})
    prov = sample_input_provenance(rt.header)
    assert prov["molar_mass"] == {"value": 386.0, "source": "user"}
    assert prov["mass_mg"] == {"value": 1.1, "source": "user"}


def test_overriding_a_header_value_records_the_override_not_the_file():
    """The dangerous case: the file HAS a molar mass and someone passes a different one."""
    rt = apply_sample_inputs(load_dat(str(VSM)), {"molar_mass": 999.0})
    prov = sample_input_provenance(rt.header)
    assert prov["molar_mass"] == {"value": 999.0, "source": "user"}
    assert prov["mass_mg"]["source"] == "header", "an untouched input must not be relabelled"


def test_an_input_nobody_supplied_is_omitted_rather_than_reported_as_none():
    assert "molar_mass" not in sample_input_provenance(load_dat(str(MPMS)).header)


def test_an_empty_patch_does_not_mark_anything_as_user_supplied():
    rt = apply_sample_inputs(load_dat(str(VSM)), {})
    assert sample_input_provenance(rt.header)["molar_mass"]["source"] == "header"


def test_the_vsm_result_records_the_molar_mass_that_produced_its_mu_eff():
    res = _analyze(load_dat(str(VSM)))
    assert res.data["sample_inputs"]["molar_mass"] == {"value": 200.0, "source": "header"}


def test_two_different_molar_masses_are_distinguishable_in_the_result():
    """The property that would have caught the 3.211: mu_eff moves, and the number that
    moved it is now on the record next to it."""
    seen = {}
    for mol in (200.0, 386.0):
        res = _analyze(apply_sample_inputs(load_dat(str(MPMS)),
                                           {"molar_mass": mol, "mass_mg": 1.1}))
        seen[mol] = (res.data["sample_inputs"]["molar_mass"]["value"],
                     res.data["fit"]["params"]["mu_eff"])
    assert seen[200.0][0] == 200.0 and seen[386.0][0] == 386.0
    assert seen[200.0][1] != seen[386.0][1], "fixture drifted: mu_eff must scale with molar mass"


def test_acms_records_it_too(tmp_path):
    """ACMS scales chi_mol by the same two inputs; the gap was not VSM-specific."""
    rt = load_dat(str(EXAMPLES / "ac_susceptibility.dat"))
    res = analyze_file(rt, RunConfig.load(probe_override="acms"), build_default_registry())
    assert res.data["sample_inputs"]["molar_mass"]["source"] == "header"


def test_the_cli_marks_a_flag_supplied_value_as_user_supplied():
    """End to end: an agent reading the JSON can see the number it passed."""
    out = subprocess.run([sys.executable, "-m", "cryosweep_cli", "analyze", str(MPMS),
                          "--molar-mass", "386", "--mass-mg", "1.1"],
                         capture_output=True, text=True, cwd=str(ROOT))
    payload = json.loads(out.stdout)
    assert payload["data"]["sample_inputs"]["molar_mass"] == {"value": 386.0, "source": "user"}


def test_the_gui_box_marks_a_typed_value_as_user_supplied():
    """The GUI patches through AnalysisState; it must not bypass the provenance record."""
    pytest.importorskip("PySide6")
    from cryosweep_gui.state import AnalysisState
    st = AnalysisState()
    st.load(str(MPMS))
    rt = st.patched_raw({"molar_mass": 386.0, "mass_mg": 1.1})
    assert sample_input_provenance(rt.header)["molar_mass"]["source"] == "user"


def test_the_exported_sidecar_records_the_inputs_behind_the_numbers(tmp_path):
    """The CSVs that carry mu_eff (`fit_params`, `derived`) name no molar mass, so a reader
    holding only the export cannot check the moment against the number that produced it.
    The `.meta.json` sidecar is the provenance file -- it belongs there."""
    from cryosweep_core.io.export import export_result
    res = _analyze(apply_sample_inputs(load_dat(str(MPMS)),
                                       {"molar_mass": 386.0, "mass_mg": 1.1}))
    outs = export_result(res, str(tmp_path / "e"))
    meta = json.loads(pathlib.Path(outs["meta"]).read_text())
    assert meta["sample_inputs"]["molar_mass"] == {"value": 386.0, "source": "user"}


def test_the_sidecar_omits_the_key_when_there_is_nothing_to_record(tmp_path):
    """An absent input must not appear as an empty object -- that reads as 'we checked'."""
    from cryosweep_core.io.export import export_result
    res = analyze_file(load_dat(str(EXAMPLES / "resistivity_semiconductor.dat")),
                       RunConfig.load(), build_default_registry())
    outs = export_result(res, str(tmp_path / "e"))
    meta = json.loads(pathlib.Path(outs["meta"]).read_text())
    assert "sample_inputs" not in meta or meta["sample_inputs"]
