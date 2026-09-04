def _panel(qapp):
    from cryosweep_core.registry import build_default_registry
    from cryosweep_gui.plot_controls import PlotControlsPanel
    return PlotControlsPanel(build_default_registry())

def test_tier1_tier2_controls_map_to_fields(qapp):
    p = _panel(qapp)
    p._family.setCurrentText("serif");     assert p.style.font_family == "serif"
    p._cmap.setCurrentText("viridis");      assert p.style.colormap == "viridis"
    p._cmap_rev.setChecked(True);           assert p.style.colormap_reverse is True
    p._edge_color.setCurrentText("black");  assert p.style.edge_color == "black"
    p._dpi.setValue(150);                   assert p.style.dpi == 150
    p._label_sz.setValue(14);               assert p.style.label_size == 14

def test_sentinels_map_to_none(qapp):
    p = _panel(qapp)
    p._family.setCurrentText("serif"); p._family.setCurrentText("(default)")
    assert p.style.font_family is None
    p._cmap.setCurrentText("viridis"); p._cmap.setCurrentText("(none)")
    assert p.style.colormap is None
    p._label_sz.setValue(14); p._label_sz.setValue(0.0)     # 0 == "auto"
    assert p.style.label_size is None

def test_each_control_emits_style_changed(qapp):
    p = _panel(qapp)
    seen = []
    p.style_changed.connect(lambda s: seen.append(1))
    p._dpi.setValue(150); p._cmap.setCurrentText("plasma"); p._label_sz.setValue(12)
    assert len(seen) >= 3

def test_reset_styling_restores_defaults_and_emits(qapp):
    from cryosweep_core.plotting.spec import GlobalStyle
    p = _panel(qapp)
    p._dpi.setValue(150); p._family.setCurrentText("serif")
    seen = []
    p.style_changed.connect(lambda s: seen.append(s))
    p._reset_styling()
    assert p.style == GlobalStyle() and p._dpi.value() == 300 and seen[-1] == GlobalStyle()

def test_set_style_syncs_widgets_without_emit(qapp):
    from cryosweep_core.plotting.spec import GlobalStyle
    p = _panel(qapp)
    seen = []
    p.style_changed.connect(lambda s: seen.append(1))
    p.set_style(GlobalStyle(dpi=200, colormap="magma"))
    assert p._dpi.value() == 200 and p._cmap.currentText() == "magma" and seen == []
