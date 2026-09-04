"""PQ-1 2b GUI: ExportPlotsDialog + single-save reroute + AxisStrip mm overrides."""
import pathlib

from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec

FIX = pathlib.Path(__file__).resolve().parents[1] / "core" / "fixtures"

_ENTRIES = [("inverse_chi", "1/χ vs T"), ("vsm_moment_t", "Moment vs T")]


def _dlg(qapp, entries=None, style=None, **kw):
    from cryosweep_gui.export_dialog import ExportPlotsDialog
    return ExportPlotsDialog(entries or _ENTRIES, style or GlobalStyle(),
                             default_dir="/tmp", default_prefix="sample", **kw)


def _setup_vsm(win):
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    tab.show_result(tab.analyze())
    return tab


# ── dialog widget behavior ───────────────────────────────────────────────────

def test_dialog_lists_layout_kinds_all_checked(qapp):
    d = _dlg(qapp)
    assert set(d.plot_checks) == {"inverse_chi", "vsm_moment_t"}
    assert all(cb.isChecked() for cb in d.plot_checks.values())


def test_all_none_buttons(qapp):
    d = _dlg(qapp)
    d.none_btn.click()
    assert not any(cb.isChecked() for cb in d.plot_checks.values())
    assert not d.export_btn.isEnabled()          # nothing selected -> disabled
    d.all_btn.click()
    assert all(cb.isChecked() for cb in d.plot_checks.values())
    assert d.export_btn.isEnabled()


def test_dpi_seeded_from_style_and_enabled_iff_png(qapp):
    d = _dlg(qapp, style=GlobalStyle(dpi=450))
    assert d.dpi_spin.value() == 450
    assert d.fmt_checks["png"].isChecked() and d.dpi_spin.isEnabled()
    d.fmt_checks["png"].setChecked(False)
    d.fmt_checks["pdf"].setChecked(True)
    assert not d.dpi_spin.isEnabled()


def test_export_disabled_without_formats(qapp):
    d = _dlg(qapp)
    d.fmt_checks["png"].setChecked(False)
    assert not d.export_btn.isEnabled()


def test_assemble_returns_selection(qapp):
    d = _dlg(qapp)
    d.plot_checks["vsm_moment_t"].setChecked(False)
    d.fmt_checks["pdf"].setChecked(True)
    d.tight_cb.setChecked(True)
    d.dpi_spin.setValue(600)
    d.prefix_edit.setText("mysample")
    args = d.assemble()
    assert args["kinds"] == ["inverse_chi"]
    assert args["formats"] == ["png", "pdf"]
    assert args["dpi"] == 600 and args["tight"] is True
    assert args["prefix"] == "mysample"
    assert str(args["out_dir"]) == "/tmp"


def test_tight_default_off(qapp):
    assert not _dlg(qapp).tight_cb.isChecked()


def test_example_label_tracks_prefix_and_format(qapp):
    d = _dlg(qapp)
    d.prefix_edit.setText("abc")
    assert "abc_inverse_chi.png" in d.example_label.text()
    d.fmt_checks["png"].setChecked(False)
    d.fmt_checks["svg"].setChecked(True)
    assert "abc_inverse_chi.svg" in d.example_label.text()


# ── ProbeTab wiring ─────────────────────────────────────────────────────────

def test_probe_tab_has_exportplots_button_gated(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    from cryosweep_gui.probe_tab import ProbeTab
    from cryosweep_gui.inputs.base import build_panel
    from cryosweep_core.registry import build_default_registry
    # bare tab, nothing loaded -> disabled (same gate as saveplot_btn)
    bare = ProbeTab(probe="vsm", panel=build_panel("vsm"), registry=build_default_registry(),
                    get_raw=lambda: None, get_unit=lambda: "CGS")
    assert not bare.exportplots_btn.isEnabled()
    win = MainWindow(); win.load_path(str(vsm_path)); win.select_probe("vsm")
    tab = _setup_vsm(win)
    assert tab.exportplots_btn.isEnabled()


def test_export_plots_invoked_with_assembled_args(qapp, vsm_path, monkeypatch, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    import cryosweep_gui.probe_tab as pt
    win = MainWindow(); win.load_path(str(vsm_path)); win.select_probe("vsm")
    tab = _setup_vsm(win)
    calls = {}

    def fake_export(result, layout, style, out_dir, prefix, kinds=None, formats=("png",), tight=False):
        calls.update(result=result, layout=layout, style=style, out_dir=out_dir,
                     prefix=prefix, kinds=kinds, formats=formats, tight=tight)
        return []

    monkeypatch.setattr(pt, "export_plots", fake_export)

    class FakeDialog:
        def __init__(self, *a, **k): pass
        def exec(self): return 1                     # accepted
        def assemble(self):
            return {"kinds": ["inverse_chi"], "formats": ["png", "pdf"],
                    "dpi": 500, "tight": False, "out_dir": tmp_path, "prefix": "zz"}

    monkeypatch.setattr(pt, "ExportPlotsDialog", FakeDialog)
    tab._on_export_plots()
    assert calls["kinds"] == ["inverse_chi"] and calls["formats"] == ["png", "pdf"]
    assert calls["prefix"] == "zz" and calls["out_dir"] == tmp_path
    assert calls["style"].dpi == 500                  # dpi write-back applied before export
    assert tab.controls.style.dpi == 500              # ...and persisted to the panel style


def test_single_save_routes_through_save_figure(qapp, vsm_path, monkeypatch, tmp_path):
    from cryosweep_gui.main_window import MainWindow
    import cryosweep_gui.probe_tab as pt
    win = MainWindow(); win.load_path(str(vsm_path)); win.select_probe("vsm")
    tab = _setup_vsm(win)
    seen = {}

    def fake_save(fig, path, style, spec=None, fmt=None, tight=False):
        seen.update(fig=fig, path=path, style=style, spec=spec, fmt=fmt)
        return pathlib.Path(path)

    monkeypatch.setattr(pt, "save_figure", fake_save)
    tab._save_plot_to(tmp_path / "x.pdf")
    assert seen["fig"] is tab.output.last_figure
    assert seen["spec"] is not None                   # focused card's spec rides along
    assert seen["style"] is tab.controls.style


# ── AxisStrip mm overrides ──────────────────────────────────────────────────

def _strip(qapp, spec=None):
    from cryosweep_gui.plot_controls import AxisStrip
    from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
    kind = next(k for k in BUILTIN_PLOTKINDS if k.key == "inverse_chi")
    return AxisStrip([], spec if spec is not None else PlotSpec(), kind)


def test_axisstrip_mm_defaults_auto_none(qapp):
    spec = PlotSpec()
    s = _strip(qapp, spec)
    assert s._w_mm.value() == 0 and s._h_mm.value() == 0     # 0 == "auto"
    assert spec.width_mm is None and spec.height_mm is None


def test_axisstrip_mm_commit_and_signal(qapp):
    spec = PlotSpec()
    s = _strip(qapp, spec)
    fired = []
    s.spec_changed.connect(lambda: fired.append(1))
    s._w_mm.setValue(120.0); s._commit_mm()
    assert spec.width_mm == 120.0 and spec.height_mm is None
    assert fired
    s._w_mm.setValue(0.0); s._commit_mm()
    assert spec.width_mm is None                              # back to auto


def test_axisstrip_mm_seeded_from_spec(qapp):
    s = _strip(qapp, PlotSpec(width_mm=85.0, height_mm=60.0))
    assert s._w_mm.value() == 85.0 and s._h_mm.value() == 60.0
