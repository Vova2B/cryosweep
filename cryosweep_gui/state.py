from __future__ import annotations
import dataclasses
from cryosweep_core.io.loader import load_dat

class AnalysisState:
    """Holds the single loaded RawTable + the last Result per probe key.
    The cached RawTable is treated as immutable: header patches produce COPIES."""
    def __init__(self):
        self._raw = None
        self._results: dict[str, object] = {}

    def load(self, path) -> None:
        self._raw = load_dat(str(path))
        self._results.clear()

    def get_raw(self):
        return self._raw

    def patched_raw(self, header_patch: dict):
        """Return a copy of the cached RawTable with header fields overridden.
        Empty patch -> the cached object itself (no copy)."""
        rt = self._raw
        if rt is None or not header_patch:
            return rt
        return dataclasses.replace(rt, header=dataclasses.replace(rt.header, **header_patch))

    def cache_result(self, probe: str, result) -> None:
        self._results[probe] = result

    def get_result(self, probe: str):
        return self._results.get(probe)
