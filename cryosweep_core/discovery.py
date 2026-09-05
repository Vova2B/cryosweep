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


def discover_for(registry, result) -> dict:
    """File-aware discovery (KNOWN-ISSUES #10): the global `discover` dump filtered to the
    file actually in hand. `plots` keeps only the detected probe's kinds, each carrying
    `available` — whether its series builder yields anything against THIS analyzed result,
    the same predicate `reconcile_layout` uses for "backed". That answers the question a
    render refusal poses ("plot kind 'X' unavailable: no series selected") without guessing.
    `probes`/`fits`/`observables` stay the global registry facts — they do not depend on the
    file — and the no-file `discover` dump is untouched (byte-identical for existing callers).
    """
    d = discover(registry)
    probe = (getattr(result, "data", None) or {}).get("probe")
    plots = []
    for k in registry.plot_kinds_for(probe):
        try:
            available = bool(k.series(result))
        except Exception:            # a builder that cannot even run on this file: not drawable
            available = False
        plots.append({"key": k.key, "label": k.label, "probe": k.probe,
                      "default_xscale": k.default_xscale, "default_yscale": k.default_yscale,
                      "available": available})
    d["plots"] = sorted(plots, key=lambda p: p["key"])
    d["probe"] = probe
    return d
