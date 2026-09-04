from __future__ import annotations
import dataclasses
import pathlib
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QFileDialog, QSplitter,
                             QScrollArea, QGroupBox)
from cryosweep_core.config import RunConfig
from cryosweep_gui.worker import AnalyzeWorker, run_analysis
from cryosweep_core.io.export import export_result
from cryosweep_core.plotting.export import save_figure, export_plots
from cryosweep_core.reports import build_report
from cryosweep_gui.export_dialog import ExportPlotsDialog
from cryosweep_gui.output_panel import OutputPanel
from cryosweep_gui.plot_controls import PlotControlsPanel
from cryosweep_gui.status_banner import StatusBanner   # type clarity; the banner is the window's
from cryosweep_core.plotting.presets import reconcile_layout, reconcile_overlay_layout
from cryosweep_core.plotting.spec import PlotLayout, PlotEntry
from cryosweep_gui.preset_bar import PresetBar
from cryosweep_core.io.loader import load_dat
from cryosweep_core.plotting.catalog import OverlayFile
from cryosweep_gui.file_manager import FileManager

_EXPORTABLE = ("ok", "gated", "low_confidence")
_RENDERABLE = ("ok", "low_confidence")          # statuses that carry plottable data (gated/error do not)


class _FileEntry:
    __slots__ = ("file_id", "path", "label", "include", "colour", "result", "state")
    def __init__(self, file_id, path, label):
        self.file_id = file_id; self.path = path; self.label = label
        self.include = True; self.colour = None; self.result = None; self.state = {}

class ProbeTab(QWidget):
    def __init__(self, probe, panel, registry, get_raw, get_unit):
        super().__init__()
        self.probe = probe
        self.panel = panel
        self._registry = registry
        self._get_raw = get_raw
        self._get_unit = get_unit
        self._last_result = None
        self._worker = None
        self._files: list[_FileEntry] = []
        self._next_id = 0
        self._focus = 0                          # index of the focused entry (export/report target)
        self.output = OutputPanel()

        self.controls = PlotControlsPanel(registry)
        self.controls.setMinimumWidth(280)
        self.controls.layout_changed.connect(self._apply_layout)
        self.controls.style_changed.connect(self._apply_style)
        self._layout_state = None
        self._mw = None
        self.output.layout_edited.connect(self._persist_last_used)

        self.preset_bar = PresetBar(self.probe)
        self.file_manager = FileManager(self)
        self.file_manager.changed.connect(self.analyze_and_render)
        if hasattr(self.panel, "refit_requested"):
            self.panel.refit_requested.connect(self.analyze_and_render)

        # ── 3-zone splitter: [left-inputs | center-output | right-controls] ──
        left_widget = QWidget()
        left = QVBoxLayout(left_widget)
        left.addWidget(self.file_manager)
        _preset_box = QGroupBox("Plot preset"); _pbl = QVBoxLayout(_preset_box)
        _pbl.setContentsMargins(6, 6, 6, 6); _pbl.addWidget(self.preset_bar)
        left.addWidget(_preset_box)
        left.addWidget(self.panel)
        self.analyze_btn = QPushButton("Analyze")
        self.export_btn = QPushButton("Export CSV")
        self.report_btn = QPushButton("Save report")
        self.saveplot_btn = QPushButton("Save plot")
        self.exportplots_btn = QPushButton("Export plots…")
        for b in (self.analyze_btn, self.export_btn, self.report_btn, self.saveplot_btn,
                  self.exportplots_btn):
            left.addWidget(b)
        left.addStretch(1)                        # pack inputs to top; no controls here anymore
        self._toggle_btn = QPushButton("◀ Controls")
        self._toggle_btn.setFixedHeight(22)
        self._toggle_btn.setToolTip("Show/hide the plot controls panel")
        left.addWidget(self._toggle_btn)

        self._left_scroll = QScrollArea()
        self._left_scroll.setWidgetResizable(True)
        self._left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._left_scroll.setWidget(left_widget)

        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._left_scroll)   # pane 0: inputs (scrollable)
        self._splitter.addWidget(self.output)     # pane 1: plots
        self._splitter.addWidget(self.controls)   # pane 2: plot controls (right)
        self._splitter.setStretchFactor(0, 0)     # left: fixed
        self._splitter.setStretchFactor(1, 1)     # center: absorbs extra space
        self._splitter.setStretchFactor(2, 0)     # right: fixed
        self._splitter.setSizes([300, 800, 300])  # sensible initial widths
        self._cached_splitter_sizes: list[int] | None = None
        self._controls_visible: bool = True       # logical state (default open)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._splitter)

        self.analyze_btn.clicked.connect(self._on_analyze)
        self.export_btn.clicked.connect(self._on_export)
        self.report_btn.clicked.connect(self._on_report)
        self.saveplot_btn.clicked.connect(self._on_saveplot)
        self.exportplots_btn.clicked.connect(self._on_export_plots)
        self._toggle_btn.clicked.connect(self._on_toggle_controls)
        self._gate_buttons(None)

        # the window injects this so the tab can push status to the shared banner
        self.banner: StatusBanner | None = None

    # ---- the seam: pure, synchronous, returns a Result (caller renders) ----
    def _build_cfg(self, rt):
        """Apply panel overrides/patch to *rt* and return (rt_patched, RunConfig). Pure helper."""
        overrides = {"unit_system": self._get_unit(), "probe_override": self.probe}
        overrides.update(self.panel.build_overrides())
        patch = self.panel.build_header_patch()
        if patch:
            rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, **patch))
        return rt, RunConfig.load(**overrides)

    def _prepare(self):
        """GUI-thread: read the panel widgets -> (rt, cfg), or None if no file is loaded."""
        rt = self._get_raw()
        if rt is None:
            return None
        return self._build_cfg(rt)

    def analyze(self):
        """Synchronous analysis (used by tests, the sync fallback, and main_window auto-reanalyze)."""
        prep = self._prepare()
        if prep is None:
            return None
        return run_analysis(prep[0], prep[1], self._registry)

    def request_analysis(self):
        """Async: run analyze_file off the GUI thread; render on the GUI thread when done."""
        if self._worker is not None and self._worker.isRunning():
            return                               # ignore re-entrant requests while one is running
        prep = self._prepare()
        if prep is None:
            if self.banner is not None:
                self.banner.show_message("Load a .dat file first.")
            return
        self._set_busy(True)
        self._worker = AnalyzeWorker(prep[0], prep[1], self._registry)
        self._worker.done.connect(self._on_analyzed)
        self._worker.start()

    def _on_analyzed(self, result):              # GUI thread (queued delivery)
        self._set_busy(False)
        restore = self._mw.preset_store.last_used.get(self.probe) if self._mw else None
        self.show_result(result, restore_layout=restore)

    def _set_busy(self, busy):
        self.analyze_btn.setEnabled(not busy)
        if busy and self.banner is not None:
            self.banner.show_message("analyzing…")

    def stop_worker(self):
        """Join a running worker (called on window close so no QThread is destroyed mid-run)."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.wait()

    def _backed_kinds(self, results) -> list[str]:
        """Kinds with data behind them for these results (stamped into PlotLayout.known so a
        restore can tell newly-backed kinds from user-unchecked ones)."""
        probe = (results[0].data or {}).get("probe")
        return [k.key for k in self._registry.plot_kinds_for(probe)
                if any(k.series(r) for r in results)]

    def show_result(self, result, restore_layout=None, restore_exact=False) -> None:
        """restore_exact=True applies restore_layout as-is (named presets: the subset is
        deliberate); False (last_used) also brings in kinds that became backed since save."""
        self._last_result = result
        self.controls.blockSignals(True)
        try:
            self.controls.set_result(result, self.probe)          # checkboxes for backed kinds (all on)
            if restore_layout is not None:
                restored = reconcile_layout(restore_layout, result, self._registry,
                                            add_newly_backed=not restore_exact)
                self.controls.set_enabled_set([e.kind for e in restored.plots])
                self._layout_state = restored
            else:
                self._layout_state = self.controls.current_layout().model_copy(
                    update={"known": self._backed_kinds([result])})
        finally:
            self.controls.blockSignals(False)
        self.output.style = self.controls.style
        self.output.show_result(result, self._layout_state)
        if hasattr(self.panel, "show_comparison"):
            self.panel.show_comparison((result.data or {}).get("comparison"))
        self._gate_buttons(result)
        if self.banner is not None:
            self.banner.show_result(result)

    def _apply_layout(self, layout):
        new_kinds = [e.kind for e in layout.plots]
        old = {e.kind: e for e in (self._layout_state.plots if self._layout_state else [])}
        merged = [old.get(k) or PlotEntry(kind=k) for k in new_kinds]   # preserve edited specs across toggles
        self._layout_state = PlotLayout(
            plots=merged,
            known=self._layout_state.known if self._layout_state else None)  # keep backed-at-save set
        if self._is_overlay():
            self._render_files(); self._persist_last_used(); return
        if self._last_result is not None:
            self.output.style = self.controls.style
            self.output.show_result(self._last_result, self._layout_state)
            self._gate_buttons(self._last_result)
            self._persist_last_used()

    def _apply_style(self, style):
        self.output.style = style
        if self._is_overlay():
            self._render_files(); return
        if self._last_result is not None and self._layout_state is not None:
            self.output.show_result(self._last_result, self._layout_state)

    def _persist_last_used(self):
        if self._mw is not None and self._layout_state is not None:
            self._mw.preset_store.last_used[self.probe] = self._layout_state.model_copy(deep=True)
            self._mw._save_presets_soon()

    def bind_window(self, mw):
        """Injected by MainWindow after the store exists (like `banner`)."""
        self._mw = mw
        self.preset_bar.bind(mw.preset_store, self, mw._save_presets_soon)

    # ---- file list (C1: shared params via self.panel) ----
    def _dedup_label(self, path):
        base = pathlib.Path(path).stem
        taken = {e.label for e in self._files}
        if base not in taken:
            return base
        i = 2
        while f"{base} ({i})" in taken:
            i += 1
        return f"{base} ({i})"

    def add_file(self, path):
        e = _FileEntry(self._next_id, str(path), self._dedup_label(path))
        self._next_id += 1
        self._files.append(e)
        return e

    def set_files(self, paths):
        self._files = []; self._next_id = 0; self._focus = 0
        for p in paths:
            self.add_file(p)
        if getattr(self, "file_manager", None) is not None:
            self.file_manager.refresh()

    def add_overlay_path(self, path):
        self.add_file(path)
        self.file_manager.refresh()
        self.file_manager.changed.emit()

    def remove_file(self, idx):
        self.commit_focused_params()                 # snapshot edits before any index shift
        if 0 <= idx < len(self._files):
            self._files.pop(idx)
            self._focus = min(self._focus, max(0, len(self._files) - 1))

    def set_include(self, idx, on):
        if 0 <= idx < len(self._files):
            self._files[idx].include = bool(on)

    def set_colour(self, idx, colour):
        if 0 <= idx < len(self._files):
            self._files[idx].colour = colour

    def _included(self):
        return [e for e in self._files if e.include]

    def _is_overlay(self):
        return len(self._included()) >= 2

    def commit_focused_params(self):
        """Read the panel widgets into the focused entry's saved state."""
        if 0 <= self._focus < len(self._files):
            self._files[self._focus].state = self.panel.get_state()

    def focus_file(self, idx):
        """Bind the panel to entry idx: commit the old focus, then load idx's saved state into the panel."""
        if not (0 <= idx < len(self._files)):
            return
        self.commit_focused_params()
        self._focus = idx
        self.panel.set_state(self._files[idx].state)

    def _prepare_entry(self, entry):
        """Build (rt, cfg) for one entry using the entry's own saved params (C2). Returns None on load failure."""
        try:
            rt = load_dat(entry.path)
        except Exception:
            return None
        self.panel.set_state(entry.state)                     # transiently load the entry's params
        return self._build_cfg(rt)

    def analyze_and_render(self):
        """Sync analyze of all included entries -> render (A/B if 1, overlay if ≥2)."""
        self.commit_focused_params()                          # snapshot live edits before the transient set_state loop
        inc = self._included()
        if not inc:
            return
        jobs = [(self._prepare_entry(e), e) for e in inc]
        for prep, e in jobs:
            e.result = run_analysis(prep[0], prep[1], self._registry) if prep else None
        self._render_files()
        if self._files:                                       # restore the panel to the focused entry
            self.panel.set_state(self._files[self._focus].state)

    def sync_panel_to_focus(self):
        """Load the focused entry's saved params into the panel (no commit). Used after a remove."""
        if 0 <= self._focus < len(self._files):
            self.panel.set_state(self._files[self._focus].state)

    def _render_files(self):
        inc = self._included()
        usable = [e for e in inc if e.result is not None and e.result.status in _RENDERABLE]
        note = None
        if usable and len(usable) < len(inc):                 # rendered something but hid some files
            dropped = [e.label for e in inc if e not in usable]
            note = f"{len(dropped)} of {len(inc)} file(s) not plotted (gated/error/load failure): {', '.join(dropped)}"
        if not usable:
            # nothing plottable -> show the first included result (if any) so its status/gate is visible
            first = next((e.result for e in inc if e.result is not None), None)
            if first is not None:
                self.show_result(first, restore_layout=self._restore_for())
            return
        if len(usable) == 1:
            self.show_result(usable[0].result, restore_layout=self._restore_for())
            if note and self.banner is not None:
                self.banner.show_result(usable[0].result, notes=(note,))
            return
        results = [e.result for e in usable]
        overlay = [OverlayFile(e.file_id, e.label, e.colour) for e in usable]
        foc = self._files[self._focus] if 0 <= self._focus < len(self._files) else None
        focused = foc.result if (foc is not None and foc in usable) else results[0]  # export/report target
        self._show_overlay(results, overlay, restore_layout=self._restore_for(), note=note, focused=focused)

    def _restore_for(self):
        return self._mw.preset_store.last_used.get(self.probe) if self._mw else None

    def _show_overlay(self, results, overlay, restore_layout=None, note=None, focused=None):
        focused = focused if focused is not None else results[0]
        self._last_result = focused                              # export/report act on the focused file (spec §6)
        self.controls.blockSignals(True)
        try:
            self.controls.set_result(results[0], self.probe)         # plot on/off checkboxes (backed kinds)
            if restore_layout is not None:
                restored = reconcile_overlay_layout(restore_layout, results, self._registry, overlay)
                self.controls.set_enabled_set([e.kind for e in restored.plots])
                self._layout_state = restored
            else:
                self._layout_state = self.controls.current_layout().model_copy(
                    update={"known": self._backed_kinds(results)})
        finally:
            self.controls.blockSignals(False)
        self.output.style = self.controls.style
        self.output.show_result(results, self._layout_state, overlay=overlay)
        self._gate_buttons(focused)
        if self.banner is not None:
            self.banner.show_result(results[0], notes=(note,) if note else ())

    def _gate_buttons(self, result) -> None:
        ok = result is not None and result.status in _EXPORTABLE
        self.export_btn.setEnabled(ok)
        self.report_btn.setEnabled(ok)
        self.saveplot_btn.setEnabled(ok and self.output.last_figure is not None)
        self.exportplots_btn.setEnabled(ok and self.output.last_figure is not None)

    # ---- collapsible right controls pane ----

    @property
    def controls_visible(self) -> bool:
        """True when the right PlotControlsPanel is logically visible (default True).
        Tracks the intended state even when the parent tab hasn't been shown yet."""
        return self._controls_visible

    def set_controls_visible(self, visible: bool) -> None:
        """Show or hide the right controls pane, caching/restoring splitter sizes."""
        if visible == self._controls_visible:
            return
        self._controls_visible = visible
        if not visible:
            # Cache current sizes before hiding so we can restore exactly.
            self._cached_splitter_sizes = self._splitter.sizes()
            self.controls.setVisible(False)
            self._toggle_btn.setText("▶ Controls")
        else:
            self.controls.setVisible(True)
            if self._cached_splitter_sizes is not None:
                self._splitter.setSizes(self._cached_splitter_sizes)
            self._toggle_btn.setText("◀ Controls")

    def _on_toggle_controls(self) -> None:
        self.set_controls_visible(not self._controls_visible)

    def _on_analyze(self):
        self.request_analysis()

    def _on_export(self):
        if self._last_result is None:
            return
        stem, _ = QFileDialog.getSaveFileName(self, "Export CSV", f"{self.probe}_out")
        if stem:
            export_result(self._last_result, stem, fmt="csv")

    def _on_report(self):
        if self._last_result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save report", f"{self.probe}_report.md")
        if path:
            with open(path, "w") as f:
                f.write(build_report(self._last_result)["markdown"])

    def _save_plot_to(self, path, fmt=None):
        if self.output.last_figure is not None:
            fig = self.output.last_figure
            # last_figure is the first rendered card's figure; pass that card's spec
            # so a per-plot mm override rides along to save_figure.
            spec = next((c.entry.spec for c in self.output._cards if c.figure is fig), None)
            save_figure(fig, path, self.controls.style, spec=spec, fmt=fmt)

    def _on_saveplot(self):
        if self.output.last_figure is None:
            return
        path, sel = QFileDialog.getSaveFileName(
            self, "Save plot", f"{self.probe}_plot.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            fmt = {"PNG (*.png)": "png", "PDF (*.pdf)": "pdf", "SVG (*.svg)": "svg"}.get(sel)
            self._save_plot_to(path, fmt=fmt)

    def _on_export_plots(self):
        if self._last_result is None or self._layout_state is None:
            return
        from cryosweep_gui.output_panel import _KIND_MAP
        entries = [(e.kind, _KIND_MAP[e.kind].label if e.kind in _KIND_MAP else e.kind)
                   for e in self._layout_state.plots]
        src = self._files[self._focus].path if self._files else None
        default_dir = pathlib.Path(src).parent if src else pathlib.Path.cwd()
        prefix = pathlib.Path(src).stem[:25] if src else "export"
        dlg = ExportPlotsDialog(entries, self.controls.style, default_dir, prefix, parent=self)
        if not dlg.exec():
            return
        args = dlg.assemble()
        if "png" in args["formats"]:
            # write the chosen DPI back through the panel widget so style + presets stay in sync
            self.controls._dpi.setValue(args["dpi"])
        paths = export_plots(self._last_result, self._layout_state, self.controls.style,
                             args["out_dir"], args["prefix"], kinds=args["kinds"],
                             formats=args["formats"], tight=args["tight"])
        if self.banner is not None:
            self.banner.setText(f"Exported {len(paths)} file(s) → {args['out_dir']}")
