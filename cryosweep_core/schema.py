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


def get_schema(name: str) -> dict:
    table = {"result": Result, "fit": FitResult, "config": RunConfig,
             "analyze:vsm": VSMData, "analyze:hc": HCData,
             "analyze:resistivity": ResistivityData,
             "analyze:hall": HallData,
             "analyze:hall_tdep": HallTempDepData}
    if name not in table:
        raise KeyError(f"unknown schema '{name}'; known: {sorted(table)}")
    return table[name].model_json_schema()
