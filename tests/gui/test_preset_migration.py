"""Task 6: the preset store rename must carry existing users across, one-way.

Three cases. The third is the hermeticity one: once the NEW file exists, the legacy path
must not be consulted at all — a fallback phrased as "if the new file is absent, read the
legacy path" is what makes the whole GUI suite a function of $HOME (BLOCKER B3).
"""
import json
import cryosweep_gui.presets_io as pio


def _store_json(field_unit):
    return {"version": 1, "global_style": {"field_unit": field_unit},
            "last_used": {}, "presets": []}


def test_legacy_is_read_when_new_is_absent(qapp, tmp_path, monkeypatch):
    from cryosweep_gui.main_window import MainWindow
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(_store_json("T")))
    new = tmp_path / "new.json"
    monkeypatch.setattr(pio, "default_store_path", lambda: new)
    monkeypatch.setattr(pio, "legacy_store_path", lambda: legacy)

    win = MainWindow()
    assert win.preset_path == new, "preset_path must stay on the NEW path, not the legacy one"
    assert win.preset_store.global_style.field_unit == "T", "legacy values did not load"


def test_save_writes_new_and_leaves_legacy_byte_identical(qapp, tmp_path, monkeypatch):
    from cryosweep_gui.main_window import MainWindow
    from cryosweep_core.plotting.presets import save_store
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(_store_json("T")))
    before = legacy.read_bytes()
    new = tmp_path / "new.json"
    monkeypatch.setattr(pio, "default_store_path", lambda: new)
    monkeypatch.setattr(pio, "legacy_store_path", lambda: legacy)

    win = MainWindow()
    save_store(win.preset_store, win.preset_path)

    assert new.exists(), "the first save must land on the new path"
    assert legacy.read_bytes() == before, "the legacy file must never be written"


def test_new_present_means_legacy_is_never_consulted(qapp, tmp_path, monkeypatch):
    from cryosweep_gui.main_window import MainWindow
    new = tmp_path / "new.json"
    new.write_text(json.dumps(_store_json("Oe")))

    def _boom():
        raise AssertionError("legacy_store_path consulted although the new file exists")
    monkeypatch.setattr(pio, "default_store_path", lambda: new)
    monkeypatch.setattr(pio, "legacy_store_path", _boom)

    win = MainWindow()
    assert win.preset_store.global_style.field_unit == "Oe"
