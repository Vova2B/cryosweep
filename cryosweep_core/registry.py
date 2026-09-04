from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

@dataclass(frozen=True)
class Need:
    key: str
    scope: Literal["header", "file", "sample"] = "file"
    provider_probe: str | None = None
    observable: str | None = None
    units: str | None = None
    match_axes: tuple = ()
    required: bool = True

@runtime_checkable
class Detector(Protocol):
    key: str
    def matches(self, h, cols) -> float: ...
    def axes(self, h, cols) -> list: ...

class Registry:
    def __init__(self):
        self._detectors = {}
        self._analyzers = {}
        self._fitmodels = {}
        self._observables = {}
        self._plotkinds = {}
    def register_detector(self, d): self._detectors[d.key] = d
    def register_analyzer(self, probe, a): self._analyzers[probe] = a
    def register_fitmodel(self, m): self._fitmodels[m.key] = m
    def register_observable(self, o): self._observables[o.key] = o
    def register_plotkind(self, p): self._plotkinds[p.key] = p
    def detector_keys(self): return sorted(self._detectors)
    def detectors(self): return list(self._detectors.values())
    def analyzer_keys(self): return sorted(self._analyzers)
    def get_analyzer(self, probe): return self._analyzers.get(probe)
    def fitmodel_keys(self): return sorted(self._fitmodels)
    def get_fitmodel(self, key): return self._fitmodels.get(key)
    def fitmodels(self): return list(self._fitmodels.values())
    def observable_keys(self): return sorted(self._observables)
    def plotkind_keys(self): return sorted(self._plotkinds)
    def plot_kinds_for(self, probe):
        return [p for p in self._plotkinds.values() if p.probe == probe]

def build_default_registry() -> Registry:
    from cryosweep_core.analyzers.builtins import (BUILTIN_DETECTORS, BUILTIN_ANALYZERS,
                                              BUILTIN_FITMODELS, BUILTIN_PLOTKINDS)
    reg = Registry()
    for d in BUILTIN_DETECTORS:
        reg.register_detector(d)
    for probe, a in BUILTIN_ANALYZERS:
        reg.register_analyzer(probe, a)
    for m in BUILTIN_FITMODELS:
        reg.register_fitmodel(m)
    for p in BUILTIN_PLOTKINDS:
        reg.register_plotkind(p)
    return reg
