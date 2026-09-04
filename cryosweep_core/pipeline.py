from __future__ import annotations
import json, pathlib
from typing import Literal
from pydantic import BaseModel
from cryosweep_core.io.loader import load_dat, expand_user_path
from cryosweep_core.result import EXIT_CODES

# EXIT_CODES is {ok: 0, gated: 10, low_confidence: 11, error: 2}: the soft, recoverable
# outcomes deliberately sit in a high band, away from the shell's conventional hard-failure 2.
# So the codes are NOT ordered by severity and max() over them is wrong -- it ranks a step that
# hard-failed BELOW one that merely needs a flag, and the caller (an agent, per the shipped
# skill) then retries with the flag instead of noticing the failure. Rank on this instead, and
# map back to the code only at the end.
_SEVERITY = {"ok": 0, "low_confidence": 1, "gated": 2, "error": 3}
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry


class Step(BaseModel):
    command: Literal["detect", "analyze"]          # YAGNI: only these two this slice
    file: str
    options: dict = {}                             # supported: molar_mass, mass_mg, unit_system


class PipelineCfg(BaseModel):
    steps: list[Step]


def _apply_options(rt, base_cfg, options):
    import dataclasses
    from cryosweep_core.config import RunConfig
    mol = options.get("molar_mass"); mg = options.get("mass_mg")
    if mol is not None or mg is not None:
        h = rt.header
        rt = dataclasses.replace(rt, header=dataclasses.replace(
            h, molar_mass=mol if mol is not None else h.molar_mass,
            mass_mg=mg if mg is not None else h.mass_mg))
    cfg = RunConfig.load(unit_system=options["unit_system"]) if "unit_system" in options else base_cfg
    return rt, cfg


def run_pipeline(path, cfg) -> dict:
    from pydantic import ValidationError
    # Guard the pipeline-file load: missing path or malformed/invalid JSON
    # returns a structured error instead of a traceback.
    try:
        spec = PipelineCfg(**json.loads(pathlib.Path(path).read_text()))
    except (FileNotFoundError, OSError, UnicodeError, ValueError, KeyError, ValidationError) as e:
        return {"results": [], "error": str(e), "exit": 2}
    results = []; worst = "ok"
    for step in spec.steps:
        # One bad step (missing/unreadable file, analyze failure) yields a step
        # error result with exit >= 2; the remaining steps still run.
        try:
            # expand ~ at load only: step results echo step.file exactly as written
            rt = load_dat(expand_user_path(step.file))
            rt, step_cfg = _apply_options(rt, cfg, step.options)
            if step.command == "detect":
                df, cmap = canonicalize_columns(rt.df, rt.header)
                score, key = detect_probe(rt.header, set(df.columns), build_default_registry())
                results.append({"command": "detect", "status": "ok", "probe": key, "score": score})
            else:  # analyze
                from cryosweep_core.analyzers.dispatch import analyze_file
                res = analyze_file(rt, step_cfg, build_default_registry())
                results.append({"command": step.command, "status": res.status,
                                "confidence": res.confidence, "data": res.model_dump(mode="json")["data"]})
                worst = max(worst, res.status, key=lambda s: _SEVERITY.get(s, 3))
        except (FileNotFoundError, OSError, UnicodeError, ValueError, KeyError, ValidationError) as e:
            results.append({"command": step.command, "file": step.file,
                            "status": "error", "error": str(e)})
            worst = max(worst, "error", key=lambda s: _SEVERITY.get(s, 3))
    return {"results": results, "exit": EXIT_CODES.get(worst, 2)}
