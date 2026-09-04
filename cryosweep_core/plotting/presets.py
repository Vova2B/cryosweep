from __future__ import annotations
import json, os, pathlib
from pydantic import BaseModel, Field, ValidationError
from cryosweep_core.plotting.spec import GlobalStyle, PlotLayout, PlotEntry

class NamedPreset(BaseModel):
    name: str
    probe: str
    layout: PlotLayout

_JOURNAL_KINDS = {
    "vsm": ["vsm_moment_t", "vsm_chi_t", "vsm_mh"],
    "resistivity": ["resistivity_rho_t", "resistivity_mr_pct"],
    "hall": ["hall_asym_vs_B", "hall_rh_t"],
    "heatcapacity": ["hc_full_cp_t", "cp_over_t", "hc_entropy_vs_t"],
}


def _all_kinds(probe):
    from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
    return [k.key for k in BUILTIN_PLOTKINDS if k.probe == probe]


def _preset(name, probe, kinds):
    return NamedPreset(name=name, probe=probe,
                       layout=PlotLayout(plots=[PlotEntry(kind=k) for k in kinds]))


def _build_builtins():
    out: dict[str, list[NamedPreset]] = {}
    for probe in ("vsm", "resistivity", "hall", "heatcapacity", "hall_tdep"):
        presets = []
        if probe in _JOURNAL_KINDS:
            presets.append(_preset("Journal", probe, _JOURNAL_KINDS[probe]))
        presets.append(_preset("All plots", probe, _all_kinds(probe)))
        out[probe] = presets
    return out


BUILTIN_PRESETS: dict[str, list[NamedPreset]] = _build_builtins()


def builtin_presets(probe: str) -> list[NamedPreset]:
    """Return deep copies of the built-in presets for `probe` (empty list if none)."""
    return [p.model_copy(deep=True) for p in BUILTIN_PRESETS.get(probe, [])]


class PresetStore(BaseModel):
    version: int = 1                                  # informational marker; loader is version-agnostic
    global_style: GlobalStyle = Field(default_factory=GlobalStyle)
    last_used: dict[str, PlotLayout] = Field(default_factory=dict)
    presets: list[NamedPreset] = Field(default_factory=list)

def load_store(path) -> PresetStore:
    """Best-effort, element-wise tolerant load. Never raises; salvages valid entries."""
    try:
        raw = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return PresetStore()
    if not isinstance(raw, dict):
        return PresetStore()
    store = PresetStore()
    if isinstance(raw.get("version"), int):
        store.version = raw["version"]
    try:
        store.global_style = GlobalStyle.model_validate(raw.get("global_style") or {})
    except ValidationError:
        store.global_style = GlobalStyle()
    for probe, lay in (raw.get("last_used") or {}).items():
        try:
            store.last_used[str(probe)] = PlotLayout.model_validate(lay)
        except ValidationError:
            continue
    for item in (raw.get("presets") or []):
        try:
            store.presets.append(NamedPreset.model_validate(item))
        except ValidationError:
            continue
    return store

def save_store(store: PresetStore, path) -> bool:
    """Atomic best-effort write (temp + os.replace). OSError -> False, never raises;
    cleans up the temp file if the write/replace fails partway."""
    p = pathlib.Path(path)
    tmp = p.with_suffix(p.suffix + ".tmp")
    try:
        tmp.write_text(store.model_dump_json(indent=2))
        os.replace(tmp, p)
        return True
    except OSError:
        try:
            tmp.unlink(missing_ok=True)          # no stray .tmp left behind
        except OSError:
            pass
        return False

def reconcile_overlay_layout(layout: PlotLayout, results, registry, overlay,
                             add_newly_backed: bool = True) -> PlotLayout:
    """Overlay-aware reconcile: validate each entry's spec.curves (file-qualified keys) against the
    UNION of f'{file_id}::{s.key}' over all results; reset to None when stale; drop unknown kinds.
    add_newly_backed as in reconcile_layout (False for deliberate named-preset subsets)."""
    probe = (results[0].data or {}).get("probe") if results else None
    kinds = registry.plot_kinds_for(probe)
    by_key = {k.key: k for k in kinds}
    out = []
    for e in layout.plots:
        kind = by_key.get(e.kind)
        if kind is None:
            continue
        valid = set()
        for r, of in zip(results, overlay):
            valid |= {f"{of.file_id}::{s.key}" for s in kind.series(r)}
        spec = e.spec
        if spec.curves is not None and not set(spec.curves) <= valid:
            spec = spec.model_copy(update={"curves": None})
        out.append(PlotEntry(kind=e.kind, spec=spec))
    backed = [k.key for k in kinds if any(k.series(r) for r in results)]
    if add_newly_backed:
        _append_newly_backed(out, layout, backed)
    return PlotLayout(plots=out, known=backed)

def _append_newly_backed(out: list, layout: PlotLayout, backed: list[str]) -> None:
    """Append a default entry for every kind backed NOW but not known at save time (e.g. the
    R_H(T)/mobility kinds once a thickness is supplied). known=None (stores predating the field)
    can't distinguish 'unbacked then' from 'user unchecked', so recover by treating the saved
    plots as the known set — missing backed kinds reappear once, then the refreshed known
    records any deliberate uncheck."""
    known = layout.known if layout.known is not None else [e.kind for e in layout.plots]
    have = {e.kind for e in out}
    out.extend(PlotEntry(kind=k) for k in backed if k not in known and k not in have)

def reconcile_layout(layout: PlotLayout, result, registry, add_newly_backed: bool = True) -> PlotLayout:
    """Reset each entry's stale curve keys to None (curve keys are file-specific); drop entries
    whose kind isn't known for this result's probe. Keeps unbacked-but-known kinds (placeholder);
    appends kinds that became backed since the layout was saved (see _append_newly_backed) —
    pass add_newly_backed=False for curated named presets, whose subset is deliberate."""
    probe = (result.data or {}).get("probe")
    kinds = registry.plot_kinds_for(probe)
    by_key = {k.key: k for k in kinds}                            # registry has no get_plotkind()
    out = []
    for e in layout.plots:
        kind = by_key.get(e.kind)
        if kind is None:
            continue
        valid = {s.key for s in kind.series(result)}
        spec = e.spec
        if spec.curves is not None and not set(spec.curves) <= valid:
            spec = spec.model_copy(update={"curves": None})
        out.append(PlotEntry(kind=e.kind, spec=spec))
    backed = [k.key for k in kinds if k.series(result)]
    if add_newly_backed:
        _append_newly_backed(out, layout, backed)
    return PlotLayout(plots=out, known=backed)
