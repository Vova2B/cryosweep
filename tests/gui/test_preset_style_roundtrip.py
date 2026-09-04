import pathlib
from cryosweep_core.plotting.spec import GlobalStyle, PlotLayout, PlotEntry
from cryosweep_core.plotting.presets import PresetStore


class _StubControls:
    def __init__(self): self.style = GlobalStyle()
    def set_style(self, s): self.style = s


class _StubTab:
    def __init__(self):
        self.controls = _StubControls()
        self._layout_state = PlotLayout(plots=[PlotEntry(kind="vsm_moment_t")])
        self._last_result = None


def _bar(qapp):
    from cryosweep_gui.preset_bar import PresetBar
    bar = PresetBar("vsm")
    store = PresetStore(global_style=GlobalStyle(grid=True, legend_loc="outside", fit_color="red"))
    bar.bind(store, _StubTab(), lambda: None)
    return bar, store


def test_export_then_import_roundtrips_style(qapp, tmp_path):
    bar, store = _bar(qapp)
    out = tmp_path / "p.json"
    bar.export_to(str(out))                                   # writes p.json + p.style.json
    assert out.with_suffix(".style.json").exists()
    # fresh bar with default style imports the pair
    bar2, store2 = _bar(qapp)
    store2.global_style = GlobalStyle()                       # reset to defaults
    bar2._store = store2
    ok = bar2.import_from(str(out), "imported")
    assert ok
    assert store2.global_style.grid is True
    assert store2.global_style.legend_loc == "outside"
    assert store2.global_style.fit_color == "red"
    assert bar2._tab.controls.style.grid is True              # applied to the panel too


def test_import_without_sidecar_does_not_crash(qapp, tmp_path):
    bar, store = _bar(qapp)
    layout_only = tmp_path / "layout.json"
    layout_only.write_text(bar._tab._layout_state.model_dump_json())
    before = store.global_style.model_dump_json()
    ok = bar.import_from(str(layout_only), "lo")
    assert ok                                                 # layout import still succeeds
    assert store.global_style.model_dump_json() == before     # style untouched
