from __future__ import annotations
import pathlib

def default_store_path() -> pathlib.Path:
    """Home-dir config file for v2 plot presets (distinct from v1's ~/.ppms_plot_settings.json)."""
    return pathlib.Path.home() / ".cryosweep_plot_presets.json"


def legacy_store_path() -> pathlib.Path:
    """The pre-rename store. READ-ONLY, one-way: existing users keep their presets across the
    rename, but nothing ever writes or deletes this file — the first save goes to the new path.
    Resolve it as a module attribute (`presets_io.legacy_store_path()`), never by importing the
    function into a local name, so tests can monkeypatch it and stay hermetic."""
    return pathlib.Path.home() / ".ppms_v2_plot_presets.json"
