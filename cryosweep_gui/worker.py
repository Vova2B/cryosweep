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
