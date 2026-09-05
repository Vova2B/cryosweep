"""KNOWN-ISSUES #22 — "Save plot" must save the plot on screen, not the first one.

`last_figure` was captured once while the layout rendered (the first card that had a
figure, guarded by `and self.last_figure is None`) and nothing updated it afterwards, so
navigating in Focus mode and pressing Save plot silently wrote a different figure.
"""


def _tab_with_plots(qapp, vsm_path):
    from cryosweep_gui.main_window import MainWindow
    win = MainWindow(); win.load_path(vsm_path); win.select_probe("vsm")
    tab = win.tabs.currentWidget()
    tab.panel.molar_mass_edit.setText("200"); tab.panel.mass_mg_edit.setText("5")
    tab.analyze_and_render(); qapp.processEvents()
    assert len(tab.output._cards) >= 2          # premise: more than one plot in the layout
    return win, tab


def test_focus_navigation_moves_the_saved_figure(qapp, vsm_path):
    win, tab = _tab_with_plots(qapp, vsm_path)
    out = tab.output
    out._on_focus_mode(); qapp.processEvents()
    figs = [c.figure for c in out._cards]
    assert out.last_figure is figs[out._focus_index]
    out._step_focus(+1); qapp.processEvents()
    assert out._focus_index == 1
    assert out.last_figure is figs[1]           # the figure on screen IS the one saved
    assert out.last_figure is not figs[0]


def test_grid_mode_keeps_the_first_figure(qapp, vsm_path):
    win, tab = _tab_with_plots(qapp, vsm_path)
    out = tab.output
    assert out._mode == "grid"
    first = next(c.figure for c in out._cards if c.figure is not None)
    assert out.last_figure is first


def test_save_plot_writes_the_focused_figure(qapp, vsm_path, monkeypatch):
    win, tab = _tab_with_plots(qapp, vsm_path)
    out = tab.output
    out._on_focus_mode(); out._step_focus(+1); qapp.processEvents()
    seen = {}
    import cryosweep_gui.probe_tab as pt
    monkeypatch.setattr(pt, "save_figure",
                        lambda fig, path, style, spec=None, fmt=None: seen.update(fig=fig))
    tab._save_plot_to("unused.png")
    assert seen["fig"] is out._cards[1].figure
