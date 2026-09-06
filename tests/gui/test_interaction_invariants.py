"""Interaction-level invariants, asserted generically across probe tabs.

This harness scripts realistic sequences (load -> toggle -> refit -> export) and asserts
INVARIANTS about the interaction surface, not instances:

I.  Pending freeze — while an analysis worker is in flight, every result-consumer button
    is disabled. Consumers are DISCOVERED, not named: a consumer is any button the tab
    itself disables when there is no result (`_gate_buttons(None)`), so a future consumer
    that follows the gating pattern is covered without editing this file. This is the
    class of the export-stale-state defect (async overlay add + immediate Export CSV
    silently wrote the pre-add state): on the pre-guard code the consumers stayed enabled
    in flight, and this test fails there.

II. Checkbox ownership — every toggle on a plot card's AxisStrip must round-trip (its
    artist inventory is restored EXACTLY when toggled back), and toggles that are
    presented as a row of peers (same parent widget) must own pairwise-DISJOINT artist
    footprints. This is the class of the fit-line-key regression (unchecking one
    model@field box dropped ALL fit lines): there, every sibling's footprint was the
    same 16 lines, so the disjointness invariant fails without knowing the instance.

III. Kind ownership — unchecking a plot-kind checkbox removes exactly that card and
    leaves every other card's artist inventory untouched; re-checking restores all.

IV. Fresh export — after switching the file and letting the async refit settle, the
    Export CSV button (through its real handler) writes bytes identical to a from-scratch
    analysis+export of the new file: the export path holds no stale state.

Limitations (deliberate, stated): a checkbox whose fit declined (nothing drawn) has an
empty footprint and is only round-trip-checked; hall/hall_tdep tabs need per-file inputs
(thickness/channel) and are exercised by their own tests, not this generic sweep.
"""
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import hashlib
import pathlib

import pytest
from PySide6.QtWidgets import QCheckBox, QPushButton

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"

# (probe, primary example, second example of the SAME probe for the refit/export scenario)
CASES = [
    ("vsm",           "magnetization_vsm.dat",        "magnetization_vsm_multifield.dat"),
    ("heatcapacity",  "heat_capacity_multifield.dat", "heat_capacity.dat"),
    ("resistivity",   "resistivity_superconductor.dat", "resistivity_semiconductor.dat"),
    ("tto",           "thermal_transport.dat",        None),
    ("acms",          "ac_susceptibility.dat",        None),
]
_IDS = [c[0] for c in CASES]


def _tab(qapp, probe, filename):
    from cryosweep_gui.main_window import MainWindow
    path = EXAMPLES / filename
    assert path.exists(), path
    win = MainWindow()
    win.state.load(str(path))
    win.select_probe(probe)
    tab = win.tabs.currentWidget()
    assert tab.probe == probe
    tab.set_files([str(path)])
    tab.analyze_and_render()                      # sync seam: settled result, cards rendered
    qapp.processEvents()
    return win, tab


def _join(tab, qapp, timeout_s=30):
    for w in (tab._worker, tab._batch_worker):
        if w is not None:
            w.wait(int(timeout_s * 1000))
    qapp.processEvents()


def _h(b: bytes) -> str:
    return hashlib.blake2b(b, digest_size=12).hexdigest()


def _artists(fig):
    """Identity set of the figure's data-carrying artists (colour deliberately excluded:
    removing a series legitimately re-slots the colour cycle of its neighbours)."""
    out = []
    for ax in fig.axes:
        for ln in ax.lines:
            lbl = str(ln.get_label())
            if lbl.startswith("_"):
                # matplotlib auto-labels ('_child3', '_nolegend_') are POSITION-dependent:
                # removing an earlier line renumbers them. Identity = gid + data.
                lbl = ""
            out.append(("line", lbl, ln.get_gid(), _h(ln.get_xydata().tobytes())))
        for c in ax.collections:
            verts = b"".join(p.vertices.tobytes() for p in c.get_paths())
            out.append(("coll", type(c).__name__, c.get_gid(), _h(verts)))
        for p in ax.patches:
            out.append(("patch", type(p).__name__, p.get_gid()))
        for t in ax.texts:
            out.append(("text", t.get_text()))
    return out


def _inventory(fig):
    """Round-trip identity: artists + the view (limits/scales), per axes."""
    view = tuple((ax.get_xlim(), ax.get_ylim(), ax.get_xscale(), ax.get_yscale())
                 for ax in fig.axes)
    return (tuple(_artists(fig)), view)


def _card_map(tab):
    return {c.entry.kind: c for c in tab.output._cards}


# ---- toggle-unit discovery (generic: no instance names) ----

def _strip_units(strip):
    """(group_key, description, is_on, set_on) for every toggle the strip presents.
    Groups: QCheckBoxes grouped by their immediate parent widget; the curve checklist's
    checkable items form one group of peers."""
    units = []
    for cb in strip.findChildren(QCheckBox):
        units.append((f"cbrow:{id(cb.parentWidget())}", cb.text(),
                      cb.isChecked(), cb.setChecked))
    cl = getattr(strip, "checklist", None)
    if cl is not None:
        from PySide6.QtCore import Qt
        for it in cl._all_data_items():
            def _set(on, it=it):
                it.setCheckState(Qt.CheckState.Checked if on else Qt.CheckState.Unchecked)
            units.append(("checklist", it.text(), it.checkState() == Qt.CheckState.Checked, _set))
    return units


# ---- I. pending freeze ----------------------------------------------------------------

def _discover_consumers(tab):
    """A consumer is any button the tab's own gate disables when there is no result."""
    settled = tab._last_result
    assert settled is not None
    enabled_with_result = {b for b in tab.findChildren(QPushButton) if b.isEnabled()}
    tab._gate_buttons(None)
    consumers = {b for b in enabled_with_result if not b.isEnabled()}
    tab._gate_buttons(settled)                    # restore
    assert {b for b in tab.findChildren(QPushButton) if b.isEnabled()} == enabled_with_result
    return consumers


@pytest.mark.parametrize("probe,fname,_alt", CASES, ids=_IDS)
def test_no_result_consumer_enabled_while_analysis_pending(qapp, probe, fname, _alt):
    win, tab = _tab(qapp, probe, fname)
    consumers = _discover_consumers(tab)
    assert consumers, "premise: the tab gates at least one result-consumer button"
    tab.request_analyze_and_render()              # async refit: worker in flight
    assert tab._worker_running(), "premise: the analysis is actually pending"
    stale_reachable = [b.text() for b in consumers if b.isEnabled()]
    _join(tab, qapp)
    assert not stale_reachable, (
        f"result consumers enabled while an analysis was pending: {stale_reachable} — "
        "clicking them acts on the PREVIOUS result")
    # and the freeze lifts: the completion render re-enables every consumer
    assert all(b.isEnabled() for b in _discover_consumers(tab) | consumers)
    win.close(); qapp.processEvents()


# ---- II. strip-checkbox ownership -----------------------------------------------------

@pytest.mark.parametrize("probe,fname,_alt", CASES, ids=_IDS)
def test_strip_toggles_round_trip_and_peers_own_disjoint_artists(qapp, probe, fname, _alt):
    win, tab = _tab(qapp, probe, fname)
    checked = 0
    for kind, card in _card_map(tab).items():
        if card.figure is None:
            continue
        inv0 = _inventory(card.figure)
        footprints = {}                            # (group, desc) -> set of artist ids
        for group, desc, on, set_on in _strip_units(card.strip):
            set_on(not on)                         # flip (synchronous re-render)
            fig1 = card.figure
            # unchecking a card's ONLY series legitimately yields the placeholder
            # (NothingToPlot) — that is "every artist removed", not a defect
            a0 = set(inv0[0])
            a1 = set(_artists(fig1)) if fig1 is not None else set()
            footprints[(group, desc)] = (a0 - a1) | (a1 - a0)
            set_on(on)                             # flip back
            assert card.figure is not None and _inventory(card.figure) == inv0, (
                f"{kind}: toggling {desc!r} off/on did not restore the plot exactly")
            checked += 1
        groups = {}
        for (group, desc), fp in footprints.items():
            groups.setdefault(group, []).append((desc, fp))
        for group, members in groups.items():
            for i, (da, fa) in enumerate(members):
                for db, fb in members[i + 1:]:
                    overlap = fa & fb
                    assert not overlap, (
                        f"{kind}: peer toggles {da!r} and {db!r} both control "
                        f"{sorted(overlap)[:4]}{'…' if len(overlap) > 4 else ''} — each "
                        "checkbox must toggle exactly its own artists")
    assert checked, "premise: at least one toggle was exercised"
    win.close(); qapp.processEvents()


# ---- III. kind-checkbox ownership -----------------------------------------------------

@pytest.mark.parametrize("probe,fname,_alt", CASES, ids=_IDS)
def test_kind_checkbox_removes_exactly_its_card(qapp, probe, fname, _alt):
    win, tab = _tab(qapp, probe, fname)
    baseline = {k: _inventory(c.figure) for k, c in _card_map(tab).items()
                if c.figure is not None}
    kinds = list(tab.controls._boxes)
    assert kinds, "premise: the probe has plot-kind checkboxes"
    for k in kinds:
        tab.controls.set_kind_enabled(k, False)
        now = _card_map(tab)
        assert k not in now, f"unchecking {k} left its card in place"
        assert set(now) == set(kinds) - {k}, (
            f"unchecking {k} changed OTHER cards: {sorted(set(kinds) ^ set(now) ^ {k})}")
        for ok, c in now.items():
            if c.figure is not None and ok in baseline:
                assert _inventory(c.figure) == baseline[ok], (
                    f"unchecking {k} altered the {ok} plot")
        tab.controls.set_kind_enabled(k, True)
        restored = {kk: _inventory(c.figure) for kk, c in _card_map(tab).items()
                    if c.figure is not None}
        assert restored == baseline, f"re-checking {k} did not restore the original plots"
    win.close(); qapp.processEvents()


# ---- IV. fresh export after refit -----------------------------------------------------

@pytest.mark.parametrize("probe,fname,alt",
                         [c for c in CASES if c[2] is not None],
                         ids=[c[0] for c in CASES if c[2] is not None])
def test_export_after_refit_carries_no_stale_state(qapp, tmp_path, monkeypatch, probe,
                                                   fname, alt):
    from cryosweep_gui import probe_tab as pt
    from cryosweep_core.io.export import export_result

    win, tab = _tab(qapp, probe, fname)
    # switch the file: fires the realistic async path (file_manager.changed)
    tab.set_files([str(EXAMPLES / alt)])
    tab.file_manager.changed.emit()
    _join(tab, qapp)

    # the button's REAL handler, dialog monkeypatched to a tmp stem
    stem = tmp_path / "gui_export"
    monkeypatch.setattr(pt.QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (str(stem), "")))
    assert tab.export_btn.isEnabled()
    tab.export_btn.click()
    gui_files = {p.name.replace("gui_export", "X"): p.read_bytes()
                 for p in tmp_path.glob("gui_export*")}
    assert gui_files, "the export button wrote nothing"

    # staleness probe 1: the export's provenance names the file NOW loaded, not the
    # one loaded before the refit (the shipped defect exported the pre-switch result)
    meta = next((v for k, v in gui_files.items() if k.endswith("meta.json")), None)
    assert meta is not None, "export wrote no meta.json to check provenance against"
    assert alt.encode() in meta and fname.encode() not in meta, (
        "the export's provenance still names the PREVIOUS file — stale result exported")

    # staleness probe 2: two independent state paths agree — the entry's stored result
    # (written by the batch completion) and _last_result (what the button exports)
    tstem = tmp_path / "truth"
    export_result(tab._files[0].result, str(tstem), fmt="csv")
    truth_files = {p.name.replace("truth", "X"): p.read_bytes()
                   for p in tmp_path.glob("truth*")}
    assert gui_files == truth_files, (
        "the export button and the entry's stored result disagree — the export path "
        "holds state the refit did not refresh")
    win.close(); qapp.processEvents()
