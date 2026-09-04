"""C2 Task 8 — per-file analysis params: panel as a view of the focused entry."""
from __future__ import annotations
import pathlib

FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"


def _win(qapp):
    from cryosweep_gui.main_window import MainWindow
    return MainWindow()


def test_panel_state_roundtrips_per_file(qapp):
    win = _win(qapp); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "vsm_synth.dat"), str(FIX / "vsm_synth.dat")])
    tab.focus_file(0); tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    tab.commit_focused_params()                       # snapshot file 0's params into its entry
    tab.focus_file(1); tab.panel.molar_mass_edit.setText("400"); tab.panel.mass_mg_edit.setText("9")
    tab.commit_focused_params()
    tab.focus_file(0)
    assert tab.panel.molar_mass_edit.text() == "200"  # file 0's params restored on focus
    assert tab._files[1].state["molar_mass"] == "400"  # file 1's params persisted in its entry


def test_per_file_geometry_gives_distinct_rho(qapp):
    # two resistivity files with DIFFERENT geometry -> different rho overlaid (correctness).
    # ADAPTATION: act_synth.dat is ACT format (only Res. chN ohm-cm columns, no raw Bridge Resistance),
    # so geometry recompute is unavailable for it.  hall_synth.dat is QD-Resistivity format and has
    # Bridge N Resistance (Ohms), enabling geometry recompute.  That file has only field sweeps so we
    # use rho_h_curves (not rho_t_curves).  rho_source is per-bridge ("geometry"), not top-level.
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "hall_synth.dat"), str(FIX / "hall_synth.dat")])
    tab.focus_file(0); tab.panel.width_edit.setText("2"); tab.panel.thickness_edit.setText("0.5")
    tab.panel.length_edit.setText("3"); tab.commit_focused_params()
    tab.focus_file(1); tab.panel.width_edit.setText("4"); tab.panel.thickness_edit.setText("0.5")
    tab.panel.length_edit.setText("3"); tab.commit_focused_params()
    tab.analyze_and_render()
    r0, r1 = tab._files[0].result, tab._files[1].result
    # bridge-level rho_source (top-level is also "geometry" here, but bridge is authoritative)
    assert r0.data["bridges"][0]["rho_source"] == "geometry"
    assert r1.data["bridges"][0]["rho_source"] == "geometry"
    # different width -> different rho magnitudes (geometry recompute used each file's own width)
    b0 = r0.data["bridges"][0]["rho_h_curves"][0]["rho"][0]
    b1 = r1.data["bridges"][0]["rho_h_curves"][0]["rho"][0]
    assert b0 != b1


def test_switching_focus_does_not_corrupt_other_files_result(qapp):
    # switching focus to file 1, editing its params, and re-rendering must NOT change file 0's result.
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "hall_synth.dat"), str(FIX / "hall_synth.dat")])
    tab.focus_file(0); tab.panel.width_edit.setText("2"); tab.panel.thickness_edit.setText("0.5")
    tab.panel.length_edit.setText("3"); tab.commit_focused_params()
    tab.focus_file(1); tab.panel.width_edit.setText("4"); tab.panel.thickness_edit.setText("0.5")
    tab.panel.length_edit.setText("3"); tab.commit_focused_params()
    tab.analyze_and_render()
    rho0 = tab._files[0].result.data["bridges"][0]["rho_h_curves"][0]["rho"][0]
    # now focus file 1, change its width, re-render — file 0's result must be stable
    tab.focus_file(1); tab.panel.width_edit.setText("8"); tab.commit_focused_params()
    tab.analyze_and_render()
    rho0_after = tab._files[0].result.data["bridges"][0]["rho_h_curves"][0]["rho"][0]
    assert rho0_after == rho0, (
        f"file 0 rho changed after editing file 1: {rho0} -> {rho0_after}"
    )


def test_remove_file_does_not_clobber_survivor_state(qapp):
    # removing file 0 must not let uncommitted/index-shifted edits land in the survivor.
    win = _win(qapp); win.select_probe("resistivity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(FIX / "hall_synth.dat"), str(FIX / "hall_synth.dat")])
    tab.focus_file(0); tab.panel.width_edit.setText("2"); tab.panel.thickness_edit.setText("0.5")
    tab.panel.length_edit.setText("3"); tab.commit_focused_params()
    tab.focus_file(1); tab.panel.width_edit.setText("4"); tab.panel.thickness_edit.setText("0.5")
    tab.panel.length_edit.setText("3"); tab.commit_focused_params()
    tab.remove_file(0)                                 # survivor was file 1 (width=4)
    tab.analyze_and_render()
    assert tab._files[0].state["width"] == "4"        # survivor's own state intact, not clobbered
