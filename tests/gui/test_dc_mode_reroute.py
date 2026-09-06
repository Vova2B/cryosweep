"""A DC-mode ACMS file must reach the Magnetization tab in the GUI, as it does in the CLI.

Some ACMS files are recorded with the AC drive never engaged: the instrument writes the full
ACMS column set, but Frequency / Amplitude / M' / M'' are empty in every row while M-DC
carries a good M(T) or M(B) sweep. Detection is correct -- the file IS an ACMS file by token
and by column set -- and `cryosweep_core.analyzers.dispatch` already reroutes it to the
runner-up probe, because the ACMS analyzer gates on something no user input can fix. The CLI
returns `rerouted_from: "acms"` and a usable VSM result.

The GUI could not, for two independent reasons: every tab forced `probe_override`, so
dispatch returned before it ever reached the detection-and-reroute path; and the window
seeded only the top-ranked probe's tab, so the Magnetization tab's file list stayed empty
even if the user found it. The file therefore landed on AC Susceptibility, gated with
"no usable AC data", and looked like a file the app could not open.

The routing rule stays in the core. The window's only job is to ASK when a result comes back
gated, and to render the answer.
"""
import pathlib

import pytest

from cryosweep_gui.main_window import MainWindow

FIX = pathlib.Path(__file__).resolve().parent.parent / "core" / "fixtures"
DCONLY = FIX / "acms_dconly_synth.dat"
ACMS = FIX / "acms_peak_synth.dat"


@pytest.fixture
def win(qapp):
    w = MainWindow()
    yield w
    w.close()


def _probe_of_current_tab(w):
    return w.tabs.currentWidget().probe


def _files_on(w, probe):
    for i in range(w.tabs.count()):
        tab = w.tabs.widget(i)
        if tab.probe == probe:
            return list(tab._files) if hasattr(tab, "_files") else tab.file_list()
    raise AssertionError(f"no tab for probe {probe!r}")


def test_dc_mode_acms_file_lands_on_magnetization(win):
    win.load_path(str(DCONLY))
    assert _probe_of_current_tab(win) == "vsm"


def test_dc_mode_acms_file_seeds_the_magnetization_file_list(win):
    """Landing there is useless if the tab has no file: the user sees an empty list."""
    win.load_path(str(DCONLY))
    assert _files_on(win, "vsm"), "Magnetization tab was never seeded with the file"


def test_the_reroute_is_reported_not_silent(win):
    """A runner-up score sits below confidence_min by construction, so a silent hop would be
    the app quietly overruling its own detector."""
    win.load_path(str(DCONLY))
    surfaced = (win.chip.text() + " " + win.banner.text()).lower()
    assert "acms" in surfaced and "vsm" in surfaced or "magnetization" in surfaced


def test_a_real_acms_file_still_lands_on_ac_susceptibility(win):
    """The reroute must not become a general escape hatch: a genuine AC file is unaffected."""
    win.load_path(str(ACMS))
    assert _probe_of_current_tab(win) == "acms"


def test_a_gated_vsm_file_does_not_hop_away_from_magnetization(win):
    """Gated-with-a-remedy is not a dead end -- the user can fix it in the panel. Hopping
    would take them away from the two boxes that solve it."""
    examples = pathlib.Path(__file__).resolve().parents[2] / "examples"
    win.load_path(str(examples / "magnetization_mpms.dat"))
    assert _probe_of_current_tab(win) == "vsm"
    assert win.tabs.currentWidget()._last_result.status == "gated"
