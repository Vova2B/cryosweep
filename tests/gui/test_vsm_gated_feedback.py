"""A gated result must not be reported to the user as an empty selection.

A VSM file whose header carries no molar mass / sample mass returns status="gated" with
`data={"probe": "vsm"}` and no series at all, so every plot kind is unbacked and the plot
area is empty. It used to render the fixed string "(no plots selected)", which says the
opposite of what happened -- it blames the user for unchecking boxes, and sends them to a
Plots checklist that is empty for a reason they are never told. The real diagnosis existed
only in the bottom banner, phrased as CLI flags (`--molar-mass`), naming a command-line
switch rather than the two boxes in the panel two inches away.

These tests pin the three surfaces that have to agree: the empty-state text, the input
placeholders, and the label the core hands the GUI for each unmet need.
"""
import pathlib

import pytest
from PySide6.QtWidgets import QLabel

from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.config import RunConfig
from cryosweep_core.io.loader import load_dat
from cryosweep_core.registry import build_default_registry
from cryosweep_gui.inputs.vsm import VSMInputPanel
from cryosweep_gui.output_panel import _empty_message

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def gated_vsm_result():
    """The shipped MPMS example gates on both mass inputs -- no real data needed."""
    res = analyze_file(load_dat(str(EXAMPLES / "magnetization_mpms.dat")),
                       RunConfig.load(probe_override="vsm"), build_default_registry())
    assert res.status == "gated", "fixture drifted: this example is supposed to gate"
    assert {g.need for g in res.gate} == {"molar_mass", "sample_mass"}
    return res


def test_gated_empty_message_names_both_missing_inputs(gated_vsm_result):
    msg = _empty_message(gated_vsm_result)
    assert "Molar mass" in msg and "Sample mass" in msg


def test_gated_empty_message_does_not_blame_the_selection(gated_vsm_result):
    """'(no plots selected)' is false here and sends the user to an empty checklist."""
    assert "no plots selected" not in _empty_message(gated_vsm_result).lower()


def test_gated_empty_message_points_at_the_panel_not_a_cli_flag(gated_vsm_result):
    msg = _empty_message(gated_vsm_result)
    assert "--molar-mass" not in msg and "--mass-mg" not in msg
    assert "panel" in msg.lower()


def test_unselected_plots_still_read_as_an_empty_selection():
    """A non-gated result with nothing ticked keeps the original wording -- there the user
    really did uncheck everything."""
    class _Ok:
        status = "ok"
        gate = []
    assert _empty_message(_Ok()) == "(no plots selected)"


def test_empty_message_survives_a_gate_carrying_no_field_label():
    """Other probes' gates have no GUI field label; fall back to the need name, never crash."""
    class _G:
        need = "thickness"
        reason = "thickness required"
        remedy = {"flag": "--thickness"}

    class _Gated:
        status = "gated"
        gate = [_G()]
    msg = _empty_message(_Gated())
    assert "thickness" in msg.lower()
    assert "no plots selected" not in msg.lower()


def test_vsm_placeholders_do_not_call_a_required_input_optional(qapp):
    """These two are required for exactly the files that gate; '(optional)' is the lie that
    kept the user from typing them."""
    panel = VSMInputPanel()
    for edit in (panel.molar_mass_edit, panel.mass_mg_edit):
        assert "optional" not in edit.placeholderText().lower()


def test_panel_labels_match_the_labels_the_core_hands_the_gui(qapp, gated_vsm_result):
    """One fact, one spelling: the empty-state message names a box by reading `remedy.field`
    from the core, so a renamed widget label must fail here rather than silently send the
    user hunting for a box that no longer exists."""
    panel = VSMInputPanel()
    shown = {lbl.text().rstrip(" *:") for lbl in panel.findChildren(QLabel)}
    for g in gated_vsm_result.gate:
        field = (g.remedy or {}).get("field")
        assert field, f"gate {g.need!r} carries no GUI field label"
        assert field in shown, f"core says {field!r}; panel shows {sorted(shown)}"


def test_banner_names_the_box_not_only_the_cli_flag(qapp, gated_vsm_result):
    """The banner carried the only real diagnosis on screen, phrased as `--molar-mass`.
    A GUI user has no command line; name the box, keep the flag for agent users."""
    from cryosweep_gui.status_banner import StatusBanner
    b = StatusBanner()
    b.show_result(gated_vsm_result)
    assert "Molar mass" in b.text() and "Sample mass" in b.text()
    assert "field=" not in b.text(), "the GUI label must be phrased, not dumped as a k=v pair"


def test_banner_format_unchanged_for_a_gate_without_a_field_label(qapp):
    """Probes whose gates carry no GUI label keep the existing rendering byte-for-byte."""
    from cryosweep_core.result import Gate, Provenance, Result
    from cryosweep_gui.status_banner import StatusBanner
    r = Result(status="gated", confidence=0.5, data={},
               gate=[Gate(need="hall_channel", reason="no channel",
                          remedy={"flag": "--hall-channel", "example": "--hall-channel 1"})],
               provenance=Provenance(file="x.dat", sha256="ab", app_version="t", config={}))
    b = StatusBanner()
    b.show_result(r)
    assert "gated[hall_channel]: no channel → flag=--hall-channel example=--hall-channel 1" in b.text()
