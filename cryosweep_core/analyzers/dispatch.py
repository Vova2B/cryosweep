from __future__ import annotations
import hashlib, pathlib
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.probe import detect_probe, detect_probe_ranked
from cryosweep_core.result import Result, Provenance

def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()[:16] if path else ""

def _err(rawtable, cfg, key, msg) -> Result:
    path = getattr(rawtable, "path", None)
    prov = Provenance(file=str(path or rawtable.header.title or ""), sha256=_sha(path),
                      app_version=rawtable.header.app_version, config=cfg.model_dump(mode="json"))
    return Result(status="error", errors=[msg], data={"probe": key}, provenance=prov)

def _safe_analyze(analyzer, rawtable, cfg) -> Result:
    try:
        return analyzer.analyze(rawtable, cfg)
    except Exception as e:                                  # defense-in-depth: no analyzer may crash dispatch
        key = getattr(analyzer, "probe", "unknown")
        return _err(rawtable, cfg, key, f"analyzer '{key}' failed: {type(e).__name__}: {e}")

def analyze_file(rawtable, cfg, registry) -> Result:
    df, cmap = canonicalize_columns(rawtable.df, rawtable.header)
    override = getattr(cfg, "probe_override", None)
    if override:
        analyzer = registry.get_analyzer(override)
        if analyzer is None:                               # unknown override: error, do NOT fall through (silent mis-route)
            return _err(rawtable, cfg, override, f"unknown probe_override '{override}'")
        return _safe_analyze(analyzer, rawtable, cfg)
    ranked = detect_probe_ranked(rawtable.header, set(df.columns), registry)
    score, key = ranked[0] if ranked else (0.0, None)
    # score<0.5 => unknown (avoid routing an unknown file to the alphabetically-last analyzer when all scores are 0)
    analyzer = registry.get_analyzer(key) if (key and score >= 0.5) else None
    if analyzer is None:
        return _err(rawtable, cfg, key, f"no analyzer for probe '{key}' (score {score:.2f})")
    result = _safe_analyze(analyzer, rawtable, cfg)
    if _is_dead_end(result):
        alt = _try_runner_up(result, ranked, key, rawtable, cfg, registry)
        if alt is not None:
            return alt
    return result


def _is_dead_end(result) -> bool:
    """True when the analyzer gated on something the USER CANNOT SUPPLY.

    A Gate carrying a `remedy` names a flag the user can pass (--molar-mass, ...), so the
    file is on the right probe and merely needs an input. A gate with an EMPTY remedy --
    e.g. acms `need="ac_data"`, "no usable AC data" -- is a dead end on this probe: nothing
    the user types will fix it, which is exactly the signal that the file's BYAPP token and
    its contents disagree. Only then is trying another probe justified.

    SCOPE CAVEAT (adversarial review, 2026-08-31): this test is broader than the case it
    was written for. TTO also builds remedy-less gates (tto.py), so a TTO file with a
    blanked Conductivity column reaches here as a "dead end" too. What actually stops it
    being rerouted is `_try_runner_up`'s `alt_score <= 0.0` skip -- on such a file the
    ranked list is [(1.0, 'tto'), (0.0, 'vsm'), (0.0, 'resistivity')] and every alternative
    scores zero. That skip is therefore LOAD-BEARING, not a cheap guard: removing it would
    let unrelated probes claim files whose real probe honestly declined. If this ever needs
    narrowing, key it on the specific `need` rather than on remedy-emptiness.
    """
    if getattr(result, "status", None) != "gated":
        return False
    gates = getattr(result, "gate", None) or []
    return bool(gates) and not any(g.remedy for g in gates)


def _try_runner_up(primary, ranked, primary_key, rawtable, cfg, registry):
    """Re-run under the next scoring probe; return its Result only if it does better.

    "Better" means it either analysed the file, or gated on something the user CAN supply.
    A second dead end is not an improvement and is discarded, so the primary probe's own
    diagnosis is what the user sees. The reroute is always reported -- runner-up scores sit
    below `confidence_min` by construction, so this must never be silent.
    """
    for alt_score, alt_key in ranked:
        if alt_key == primary_key or not alt_key or alt_score <= 0.0:
            continue
        alt = registry.get_analyzer(alt_key)
        if alt is None:
            continue
        cand = _safe_analyze(alt, rawtable, cfg)
        if cand.status == "error" or _is_dead_end(cand):
            continue
        note = (f"rerouted {primary_key} -> {alt_key}: the {primary_key} analyzer found no "
                f"usable {primary_key} signal in this file ("
                f"{'; '.join(g.reason for g in (primary.gate or []))}), but its columns "
                f"carry {alt_key} data (detector score {alt_score:.2f})")
        data = dict(cand.data or {})
        data["rerouted_from"] = primary_key
        return cand.model_copy(update={
            "warnings": [note, *(cand.warnings or [])],
            "data": data,
        })
    return None
