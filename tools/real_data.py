"""Resolve local-only measurement files by logical key, for the dev capture scripts.

Same contract as tests/core/conftest.py's real_data(): the map (real_data_map.json at the
repo root) is gitignored, so no measurement filename appears in any tracked file. Duplicated
rather than imported from the test tree — tools/ and tests/ are independent, and this project
keeps such helpers module-local.
"""
from __future__ import annotations

import json
import pathlib


#: Local-only development markers. Either one identifies the private dev tree; both live
#: at the same root here, but pq_compare needs the gallery on a machine with no data map.
DEV_MARKERS = ("real_data_map.json", "docs/superpowers/pq-reference-gallery")


def repo_root(start=__file__):
    """The dev-tree root: the nearest ancestor holding a local-only development marker.

    Deliberately NOT a `.git` walk. cryosweep is its own repository nested inside the
    private development tree, so the nearest `.git` is cryosweep's own while the markers
    sit one level above it -- a `.git` walk returns the app directory and every lookup
    below silently resolves to None. A public checkout holds no marker at any level, falls
    through to the fallback, and resolves None on purpose: that is the skip-not-fail path.
    """
    p = pathlib.Path(start).resolve()
    for d in (p, *p.parents):
        if any((d / m).exists() for m in DEV_MARKERS):
            return d
    return p.parents[1]


def real_path(key: str):
    """Path for `key`, or None when the map or the file is unavailable."""
    root = repo_root()
    m = root / "real_data_map.json"
    if not m.exists():
        return None
    try:
        rel = json.loads(m.read_text()).get(key)
    except (ValueError, OSError):
        return None
    if not rel:
        return None
    p = root / rel
    return p if p.exists() else None
