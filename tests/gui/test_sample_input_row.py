"""The molar mass behind a mu_eff has to be visible where the mu_eff is.

`data["sample_inputs"]` records the value and whether a person supplied it, but the field/
value table only emits top-level scalars, so a nested record is silently dropped -- and GUI
users are precisely the people now being asked to type these numbers. A provenance record
nobody can see is not provenance.
"""
import pathlib

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.io.header import apply_sample_inputs
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry
from cryosweep_gui.output_panel import flatten_rows

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
VSM = EXAMPLES / "magnetization_vsm.dat"     # header carries MOLWGHT 200.0


def _rows(rt):
    res = analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())
    return dict(flatten_rows(res.data or {}))


def _find(rows, needle):
    return [f"{k} = {v}" for k, v in rows.items() if needle in k]


def test_a_header_supplied_molar_mass_is_shown_and_attributed_to_the_file():
    rows = _rows(load_dat(str(VSM)))
    hits = _find(rows, "molar_mass")
    assert hits, "the molar mass behind mu_eff is not in the table at all"
    assert "200.0" in hits[0] and "file" in hits[0].lower()


def test_a_typed_molar_mass_is_marked_as_not_coming_from_the_file():
    """The case that produced a 39%-wrong published mu_eff. It must read differently from a
    header value at a glance, not merely be present."""
    rows = _rows(apply_sample_inputs(load_dat(str(VSM)), {"molar_mass": 999.0}))
    hits = _find(rows, "molar_mass")
    assert hits and "999.0" in hits[0]
    assert "not from the file" in hits[0].lower()


def test_the_two_cases_do_not_render_identically():
    header = _find(_rows(load_dat(str(VSM))), "molar_mass")[0]
    typed = _find(_rows(apply_sample_inputs(load_dat(str(VSM)), {"molar_mass": 200.0})),
                  "molar_mass")[0]
    assert header != typed, "same number, different origin -- the table must distinguish them"
