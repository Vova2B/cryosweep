from __future__ import annotations
import dataclasses


def _need_dict(n):
    return dataclasses.asdict(n) if dataclasses.is_dataclass(n) else dict(n)


def discover(registry) -> dict:
    probes = []
    seen = set()
    for key in registry.detector_keys():
        a = registry.get_analyzer(key)
        needs = [_need_dict(n) for n in getattr(a, "needs", ())] if a else []
        probes.append({"key": key, "has_analyzer": a is not None, "has_detector": True, "needs": needs})
        seen.add(key)
    for key in registry.analyzer_keys():            # analyzers with no detector (e.g. hall)
        if key in seen:
            continue
        a = registry.get_analyzer(key)
        needs = [_need_dict(n) for n in getattr(a, "needs", ())]
        probes.append({"key": key, "has_analyzer": True, "has_detector": False, "needs": needs})
    fits = [{"key": m.key, "params": list(getattr(m, "params", []))} for m in registry.fitmodels()]
    plots = sorted(
        [{"key": p.key, "label": p.label, "probe": p.probe,
          "default_xscale": p.default_xscale, "default_yscale": p.default_yscale}
         for p in registry._plotkinds.values()],
        key=lambda p: p["key"])
    return {"probes": probes,
            "fits": sorted(fits, key=lambda f: f["key"]),
            "plots": plots,
            "observables": [{"key": k} for k in registry.observable_keys()]}
