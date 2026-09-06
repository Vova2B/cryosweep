from __future__ import annotations
from PySide6.QtCore import QThread, Signal
from cryosweep_core.analyzers.dispatch import analyze_file

def run_analysis(rt, cfg, registry):
    """The slow, Qt-free analysis (runs on a worker thread). Never raises -> always a Result."""
    try:
        return analyze_file(rt, cfg, registry)
    except Exception as e:                       # belt-and-suspenders (analyze_file already guards)
        from cryosweep_core.result import Result, Provenance
        return Result(status="error",
                      errors=[f"analyze failed: {type(e).__name__}: {e}"],
                      data={"probe": getattr(cfg, "probe_override", None) or "?"},
                      provenance=Provenance(file="", sha256="", app_version="", config={}))

class AnalyzeWorker(QThread):
    done = Signal(object)                    # emits the Result (delivered on the GUI thread, queued)

    def __init__(self, rt, cfg, registry, parent=None):
        super().__init__(parent)
        self._rt, self._cfg, self._registry = rt, cfg, registry

    def run(self):                               # executes on the worker thread
        self.done.emit(run_analysis(self._rt, self._cfg, self._registry))

class BatchAnalyzeWorker(QThread):
    """ROADMAP item 4: the analyze_and_render path (refits, file-list changes) analyzes
    every included file; this worker runs those analyses off the GUI thread. Takes the
    ALREADY-PREPARED (rt, cfg) list — widget reads stay on the GUI thread — and emits a
    result list aligned with it (None where preparation failed)."""
    done = Signal(object)                    # emits list[Result | None] (queued -> GUI thread)

    def __init__(self, preps, registry, parent=None):
        super().__init__(parent)
        self._preps, self._registry = preps, registry

    def run(self):                               # executes on the worker thread
        self.done.emit([run_analysis(p[0], p[1], self._registry) if p else None
                        for p in self._preps])
