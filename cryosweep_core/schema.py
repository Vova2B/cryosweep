from __future__ import annotations
from cryosweep_core.result import Result, FitResult
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.mag import VSMData
from cryosweep_core.analyzers.hc import HCData
from cryosweep_core.analyzers.resistivity import ResistivityData
from cryosweep_core.analyzers.hall import HallData
from cryosweep_core.analyzers.hall_tempdep import HallTempDepData

SCHEMA_NAMES = ("result", "fit", "config", "analyze:vsm", "analyze:hc", "analyze:resistivity",
                "analyze:hall", "analyze:hall_tdep")


def unknown_keys(model_cls, data, prefix: str = "") -> list[str]:
    """Dotted paths of keys in *data* that no field of *model_cls* accepts.

    Pydantic's default is to IGNORE extra keys, which turns a typo'd option file
    (``{"errorband": true}``) or a wrong shape (a spec boolean on the entry instead of
    under ``"spec"``) into a silent no-op. Callers use this to warn — never to reject:
    option files written against other versions must keep loading.

    Recurses into fields whose annotation is (or contains) a BaseModel, and into
    ``list[BaseModel]`` fields; plain dict-valued fields (e.g. RunConfig's
    ``heatcapacity.full_init``) are accepted as-is. Non-dict *data* yields [].
    """
    from typing import get_args, get_origin
    from pydantic import BaseModel

    def _model_of(ann):
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            return ann
        for arg in get_args(ann):                      # Optional[Model] and unions
            if isinstance(arg, type) and issubclass(arg, BaseModel):
                return arg
        return None

    def _list_item_model_of(ann):
        anns = [ann, *get_args(ann)]                   # list[...] possibly inside Optional
        for cand in anns:
            if get_origin(cand) is list:
                args = get_args(cand)
                if args:
                    return _model_of(args[0])
        return None

    if not isinstance(data, dict):
        return []
    out: list[str] = []
    fields = model_cls.model_fields
    for key, value in data.items():
        if key not in fields:
            out.append(prefix + key)
            continue
        ann = fields[key].annotation
        sub = _model_of(ann)
        if sub is not None and isinstance(value, dict):
            out += unknown_keys(sub, value, f"{prefix}{key}.")
            continue
        item = _list_item_model_of(ann)
        if item is not None and isinstance(value, list):
            for i, el in enumerate(value):
                out += unknown_keys(item, el, f"{prefix}{key}[{i}].")
    return out


def get_schema(name: str) -> dict:
    table = {"result": Result, "fit": FitResult, "config": RunConfig,
             "analyze:vsm": VSMData, "analyze:hc": HCData,
             "analyze:resistivity": ResistivityData,
             "analyze:hall": HallData,
             "analyze:hall_tdep": HallTempDepData}
    if name not in table:
        raise KeyError(f"unknown schema '{name}'; known: {sorted(table)}")
    return table[name].model_json_schema()
