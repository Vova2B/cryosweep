"""ROADMAP item 4 — analysis off the GUI thread for the refit / file-list paths.

The Analyze button already runs async (test_probe_tab_async.py). These tests pin the
remaining path: `refit_requested` and `file_manager.changed` now go through
`request_analyze_and_render` (BatchAnalyzeWorker), with the busy indicator and a
coalesced (never dropped) re-request. `analyze_and_render` stays the sync seam.

Responsiveness is MEASURED, not assumed: the test heartbeats the GUI thread (timestamp +
processEvents every ~1 ms) while the ~240 ms heat-capacity analysis is in flight and
asserts the heartbeats actually ran and never stalled. On the old sync wiring the emit
blocks the event loop for the whole flight, so ZERO heartbeats land during it (verified
failing before the fix). The post-analysis render legitimately stays on the GUI thread —
item 4 moves the ANALYSIS off it — so the measured window is the worker flight.
"""
import os; os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import copy
import pathlib
import time

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
HC_EXAMPLE = EXAMPLES / "heat_capacity.dat"


def _hc_tab(qapp):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow()
    win.state.load(str(HC_EXAMPLE))
    win.select_probe("heatcapacity")
    tab = win.tabs.currentWidget()
    tab.set_files([str(HC_EXAMPLE)])
    return win, tab


def _join(tab, qapp, timeout_s=15):
    w = getattr(tab, "_batch_worker", None)
    if w is not None:
        w.wait(int(timeout_s * 1000))
    qapp.processEvents()


def test_refit_signal_keeps_gui_thread_live(qapp):
    """THE measurement: event-loop turns keep happening on the GUI thread between the
    refit request and its result. Measured on this machine: ~21 turns during a ~230 ms
    analysis, typical inter-turn gap 7-12 ms (CPython GIL handoff), worst gap the final
    delivery+render, which legitimately runs on the GUI thread. On the old sync wiring
    the emit call contains the WHOLE flight, so zero turns happen inside it (verified
    failing before the fix: "0 timer ticks delivered")."""
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()                      # seed a first result synchronously
    # a refit seeded with fitted params converges in ~60 ms; reset entry AND panel to a
    # fresh panel's defaults (set_state leaves unnamed keys untouched, so {} resets
    # nothing) to get the full-length (~240 ms measured) fit the roadmap describes
    from cryosweep_gui.inputs.hc import HCInputPanel
    default_state = HCInputPanel().get_state()
    tab._files[0].state = default_state
    tab.panel.set_state(default_state)
    qapp.processEvents()                          # drain the render backlog before measuring
    before = tab._files[0].result
    beats = 0                                     # GUI-thread event-loop turns in flight
    beats_while_analyzing = 0                     # ... of which the worker was computing
    t0 = time.monotonic()
    tab.panel.refit_requested.emit()              # the user pressing a refit button
    while tab._files[0].result is before and time.monotonic() - t0 < 15:
        beats += 1
        if tab._batch_worker is not None and tab._batch_worker.isRunning():
            beats_while_analyzing += 1
        qapp.processEvents()
    done = time.monotonic()
    _join(tab, qapp)
    assert tab._files[0].result is not before, "refit never delivered a result"
    flight = done - t0
    # "not blocked" = the GUI thread's event loop keeps turning while the analysis
    # computes. The sync wiring runs the whole flight INSIDE the emit, so both counts
    # are EXACTLY ZERO there, always, by construction — that is the discriminator.
    # Thresholds are deliberately minimal: turn counts scale with machine load (measured
    # unloaded: ~21 turns / 240 ms flight; under a full parallel suite: 2 turns / 626 ms),
    # and a smoothness bound would measure the GIL and the box, not the wiring.
    assert beats >= 2, (
        f"GUI thread blocked during the {flight*1000:.0f} ms refit flight: "
        f"{beats} event-loop turns")
    assert beats_while_analyzing >= 1, (
        f"no event-loop turns while the worker analyzed ({beats} total turns in "
        f"{flight*1000:.0f} ms)")


def test_refit_async_result_matches_sync(qapp):
    """Same seed state -> the async path computes exactly the sync path's result.
    (Consecutive refits deliberately differ — 2(a) seeds each fit with the last fitted
    params — so the comparison restores the state snapshot between the two runs.)"""
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()                      # first fit; state now carries its params
    snap = copy.deepcopy(tab._files[0].state)
    tab.analyze_and_render()                      # second fit, sync = the reference
    sync2 = tab._files[0].result
    tab._files[0].state = copy.deepcopy(snap)     # rewind to the same seed state
    tab.panel.set_state(tab._files[0].state)      # and the panel: commit_focused_params snapshots it
    tab.request_analyze_and_render()
    t0 = time.monotonic()
    while tab._files[0].result is sync2 and time.monotonic() - t0 < 15:
        qapp.processEvents(); time.sleep(0.001)
    _join(tab, qapp)
    assert tab._files[0].result.model_dump_json() == sync2.model_dump_json()


def test_refit_busy_indicator_shown_and_cleared(qapp):
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()
    tab.request_analyze_and_render()
    busy_disabled = not tab.analyze_btn.isEnabled()   # capture BEFORE waiting
    busy_text = win.banner.text().lower()
    marker = object()
    t0 = time.monotonic()
    while tab._batch_worker is not None and tab._batch_worker.isRunning() \
            and time.monotonic() - t0 < 15:
        qapp.processEvents(); time.sleep(0.001)
    _join(tab, qapp)
    assert busy_disabled, "Analyze button not disabled while a refit runs"
    assert "analyz" in busy_text
    assert tab.analyze_btn.isEnabled()
    assert "ok" in win.banner.text().lower()


def test_refit_request_coalesced_not_dropped(qapp):
    """A refit requested mid-flight must not be silently dropped (a stale display would
    stand); it is coalesced into one rerun after the running one completes."""
    win, tab = _hc_tab(qapp)

    class _FakeRunning:
        def isRunning(self): return True
        def wait(self, *_): return True
    tab._batch_worker = _FakeRunning()
    tab.request_analyze_and_render()
    assert tab._pending_rerun is True
    assert isinstance(tab._batch_worker, _FakeRunning)   # did not clobber the running worker
    tab._batch_worker = None; tab._pending_rerun = False


def test_close_event_joins_running_batch_worker(qapp):
    win, tab = _hc_tab(qapp)
    tab.analyze_and_render()
    tab.request_analyze_and_render()
    assert tab._batch_worker is not None
    win.close()                                   # closeEvent -> stop_worker() joins
    assert not tab._batch_worker.isRunning()


def test_signal_wiring_targets_async_wrapper(qapp):
    """The invariant, not the symptom: the refit/file-change signals connect to the async
    wrapper. (Direct sync calls to analyze_and_render remain valid for tests/main_window.)"""
    win, tab = _hc_tab(qapp)
    import inspect
    src = inspect.getsource(type(tab).__init__)
    assert "changed.connect(self.request_analyze_and_render)" in src
    assert "refit_requested.connect(self.request_analyze_and_render)" in src
