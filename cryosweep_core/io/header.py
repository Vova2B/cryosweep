from __future__ import annotations
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
    return HeaderMeta(
        app=app, app_version=app_version, title=title,
        info=info, info_rows=tuple(info_rows), channels={},
        molar_mass=molar_mass, n_atoms=n_atoms, mass_mg=mass_mg,
        data_line=data_line, raw_lines=tuple(lines), bare_csv=bare_csv,
    )
