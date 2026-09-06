from __future__ import annotations
import dataclasses
import pathlib
from cryosweep_core.model import HeaderMeta, ChannelMeta

_ENCODINGS = ("utf-8", "latin1", "cp1252", "ISO-8859-1")

def _read_lines(path) -> list[str]:
    last = None
    for enc in _ENCODINGS:
        try:
            return pathlib.Path(path).read_text(encoding=enc).splitlines()
        except UnicodeDecodeError as e:
            last = e
    raise ValueError(f"Could not decode {path} with {_ENCODINGS}: {last}")

def _to_float(s: str):
    try:
        return float(s.strip())
    except (ValueError, AttributeError):
        return None

#: Header fields a person can supply, and that a reported quantity scales with.
SAMPLE_INPUT_FIELDS = ("molar_mass", "mass_mg", "n_atoms")


def apply_sample_inputs(rt, patch: dict):
    """Return a copy of *rt* whose header carries *patch*, recorded as USER-supplied.

    The single place a sample input may be overridden. It was previously done with a bare
    `dataclasses.replace` at four sites (CLI, pipeline, GUI state, GUI tab), after which a
    typed molar mass was indistinguishable from one the instrument wrote -- and mu_eff scales
    with it. Route every override through here so the record follows the value.

    A patch entry of None is a no-op, not an override: the CLI passes both flags always.
    Empty patch -> the original object, unmarked.
    """
    real = {k: v for k, v in (patch or {}).items() if v is not None}
    if not real:
        return rt
    h = rt.header
    marked = frozenset(h.user_supplied) | {k for k in real if k in SAMPLE_INPUT_FIELDS}
    return dataclasses.replace(rt, header=dataclasses.replace(
        h, user_supplied=marked, **real))


def sample_input_provenance(header) -> dict:
    """`{field: {"value": v, "source": "header"|"user"}}` for the sample inputs in play.

    Inputs neither the file nor the user supplied are OMITTED rather than reported as None:
    a null entry reads as "we looked and there is none", which is the same shape as a value,
    and the caller cannot tell an absent input from an unrecorded one."""
    out = {}
    for f in SAMPLE_INPUT_FIELDS:
        v = getattr(header, f, None)
        if v is None:
            continue
        out[f] = {"value": v,
                  "source": "user" if f in (header.user_supplied or ()) else "header"}
    return out


def parse_header(path) -> HeaderMeta:
    lines = _read_lines(path)
    data_line = next((i for i, ln in enumerate(lines) if ln.strip() == "[Data]"), -1)
    has_header_marker = any(ln.strip() == "[Header]" for ln in lines)
    bare_csv = (data_line < 0) and (not has_header_marker)
    head = lines[:data_line] if data_line >= 0 else lines
    app = app_version = title = None
    info: dict = {}
    info_rows: list = []
    molar_mass = n_atoms = mass_mg = None
    # QD VSM dialect, collected separately so an explicit `KEY:` row always wins (see below).
    bare_molar_mass = bare_mass_mg = None
    for ln in head:
        parts = [p.strip() for p in ln.split(",")]
        tag = parts[0].upper() if parts else ""
        if tag == "TITLE" and len(parts) > 1:
            title = parts[1]
        elif tag == "BYAPP" and len(parts) > 1:
            app = parts[1]
            app_version = parts[2] if len(parts) > 2 and parts[2] else None
        elif tag == "INFO" and len(parts) >= 3:
            value, rest = parts[1], parts[2]
            key, desc = (rest.split(":", 1) if ":" in rest else (None, rest))
            key = key.strip() if key else None
            desc = desc.strip()
            info_rows.append((key, value, desc))
            info[key or desc] = value
            if key == "MOLWGHT":
                molar_mass = _to_float(value)
            elif key == "ATOMS":
                n_atoms = _to_float(value)
            elif key == "MASS":
                mass_mg = _to_float(value)
            elif key is None:
                # The QD VSM option writes these WITHOUT a "KEY:" prefix --
                # `INFO,<mg>,SAMPLE_MASS` / `INFO,<g/mol>,SAMPLE_MOLECULAR_WEIGHT` -- so the
                # branches above never fire and a file stating both values gates as if it
                # stated neither. SAMPLE_MASS is in mg, the same unit as the MASS: row
                # (the per-row `Mass (grams)` DATA column is an unrelated VSM diagnostic).
                if desc.upper() == "SAMPLE_MASS":
                    bare_mass_mg = _to_float(value)
                elif desc.upper() == "SAMPLE_MOLECULAR_WEIGHT":
                    bare_molar_mass = _to_float(value)
    # A colon-keyed row is explicit and wins regardless of the order the two dialects appear in.
    if molar_mass is None:
        molar_mass = bare_molar_mass
    if mass_mg is None:
        mass_mg = bare_mass_mg
    return HeaderMeta(
        app=app, app_version=app_version, title=title,
        info=info, info_rows=tuple(info_rows), channels={},
        molar_mass=molar_mass, n_atoms=n_atoms, mass_mg=mass_mg,
        data_line=data_line, raw_lines=tuple(lines), bare_csv=bare_csv,
    )
