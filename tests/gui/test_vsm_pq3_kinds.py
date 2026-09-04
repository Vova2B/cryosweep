# tests/gui/test_vsm_pq3_kinds.py — PQ-3 Task 4: GUI checklist + preset/layout
# round-trip coverage for the VSM journal-pass kinds.
#
# Kinds under test (all shipped across PQ-3 Tasks 1-4):
#   vsm_moment_t   "Moment vs T"   (ramp-split ZFC/FC branches on multi-ramp files)
#   vsm_chi_t      "χ vs T"        (twin χ/χ⁻¹ axis)
#   vsm_chi_t_product "χT vs T"
#   inverse_chi    "1/χ vs T"      (Curie-Weiss journal upgrade)
#   vsm_mh         "M vs H"        (gated ON only when the file carries M(H) loops)
#
# NOTE: most assertions here exercise already-shipped Task 1-3 behaviour surfaced through
# the GUI (they pass immediately); the Task-4-specific coverage is (a) the ramp-key
# curves-subset surviving reconcile_layout and (b) vsm_mh gating on a loops-bearing file.
import types

from cryosweep_core.plotting.presets import reconcile_layout
from cryosweep_core.plotting.spec import PlotEntry, PlotLayout
from cryosweep_core.registry import build_default_registry


def _vsm_synth_tab(win):
    """vsm probe on the pure-M(T) synth fixture (no loops -> vsm_mh gated OFF)."""
    win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.show_result(tab.analyze())
    return tab


def _vsm_n_tab(win, path):
    """vsm probe on the real loops-bearing VSM_N file (needs molar mass/mass 300/1.1)."""
    win.load_path(str(path))
    win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("300")
    tab.panel.mass_mg_edit.setText("1.1")
    tab.show_result(tab.analyze())
    return tab


# ---------------------------------------------------------------------------
# (a) M(T)-family kinds appear in the Plots checklist with their labels
# ---------------------------------------------------------------------------

def test_vsm_checklist_has_mt_kinds(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(vsm_path))
    tab = _vsm_synth_tab(win)
    for k in ("vsm_moment_t", "vsm_chi_t", "vsm_chi_t_product", "inverse_chi"):
        assert k in tab.controls._boxes, k
    labels = {cb.text() for cb in tab.controls._boxes.values()}
    assert {"Moment vs T", "χ vs T", "χT vs T", "1/χ vs T"} <= labels


# ---------------------------------------------------------------------------
# (b) vsm_mh gating: omitted on the loops-less synth file, present on VSM_N
# ---------------------------------------------------------------------------

def test_vsm_mh_omitted_without_loops(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(vsm_path))
    tab = _vsm_synth_tab(win)
    # pure M(T) synth -> no field-sweep loops -> vsm_mh series() == [] -> no checkbox
    assert "vsm_mh" not in tab.controls._boxes
    assert "vsm_mh" not in tab.controls.enabled_kinds()


def test_vsm_mh_present_on_loops_file_no_crash(qapp, vsm_real_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    tab = _vsm_n_tab(win, vsm_real_path)         # must not raise (offscreen render)
    assert tab._last_result.status in ("ok", "low_confidence")
    assert "vsm_mh" in tab.controls._boxes
    assert tab.controls._boxes["vsm_mh"].text() == "M vs H"


# ---------------------------------------------------------------------------
# (c) preset/layout round-trip: vsm_mh + a curves-subset of ramp keys survives
#     reconcile_layout (kind known -> kept; ramp keys are valid series keys).
# ---------------------------------------------------------------------------

def _ramp_result():
    """Synthetic 2-ramp VSM result whose vsm_moment_t series keys are curve:r0/curve:r1."""
    n = 6
    T = list(range(n)) + list(range(n, 0, -1))
    d = {"probe": "vsm", "temperature": [float(t) for t in T],
         "moment_per_fu": [float(t) for t in T],
         "chi_molar_cgs": [1.0 + t for t in T], "inv_chi": [1.0 / (1.0 + t) for t in T],
         "inv_chi_unit": "mol*Oe/emu", "loops": [],
         "ramps": [{"direction": "warming", "i0": 0, "i1": n - 1},
                   {"direction": "cooling", "i0": n, "i1": 2 * n - 1}]}
    return types.SimpleNamespace(data=d, status="ok")


def test_ramp_key_curves_subset_survives_reconcile(qapp):
    reg = build_default_registry()
    res = _ramp_result()
    # a restored layout selecting only the warming branch of vsm_moment_t + a gated vsm_mh;
    # known covers every vsm kind, so the subset reads as deliberate (nothing gets appended)
    layout = PlotLayout(plots=[
        PlotEntry(kind="vsm_moment_t", spec={"curves": ["curve:r0"]}),
        PlotEntry(kind="vsm_mh")],
        known=[k.key for k in reg.plot_kinds_for("vsm")])
    restored = PlotLayout.model_validate(layout.model_dump())
    reconciled = reconcile_layout(restored, res, reg)
    kinds = {e.kind for e in reconciled.plots}
    assert kinds == {"vsm_moment_t", "vsm_mh"}                 # both kept
    mt = next(e for e in reconciled.plots if e.kind == "vsm_moment_t")
    # curve:r0 IS a valid ramp series key -> curves subset PRESERVED (not reset to None)
    assert mt.spec.curves == ["curve:r0"]


def test_single_ramp_stale_ramp_key_reset_but_entry_kept(qapp, vsm_path):
    # on a single-ramp real file the only series key is "curve"; a stale "curve:r1" subset
    # is reset to None by reconcile, but the entry itself survives (kind known).
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(str(vsm_path))
    tab = _vsm_synth_tab(win)
    reg = build_default_registry()
    layout = PlotLayout(plots=[PlotEntry(kind="vsm_moment_t", spec={"curves": ["curve:r1"]})])
    reconciled = reconcile_layout(PlotLayout.model_validate(layout.model_dump()),
                                  tab._last_result, reg)
    mt = next(e for e in reconciled.plots if e.kind == "vsm_moment_t")
    assert mt.spec.curves is None                              # stale ramp key dropped
