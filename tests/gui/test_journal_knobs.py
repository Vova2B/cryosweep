from cryosweep_core.plotting.spec import GlobalStyle


def _panel(qapp):
    from cryosweep_gui.plot_controls import PlotControlsPanel
    from cryosweep_core.registry import build_default_registry
    return PlotControlsPanel(build_default_registry())


def test_journal_widgets_exist_with_defaults(qapp):
    p = _panel(qapp)
    s = GlobalStyle()
    assert p._tick_dir.currentText() == s.tick_direction          # "in"
    assert p._minor_ticks.isChecked() is s.minor_ticks            # True
    assert p._ticks_top.isChecked() is s.ticks_top                # True
    assert p._ticks_right.isChecked() is s.ticks_right            # True
    assert p._grid.isChecked() is s.grid                          # False
    assert p._grid_style.currentText() == s.grid_style            # "--"
    assert abs(p._grid_alpha.value() - s.grid_alpha) < 1e-9       # 0.4
    assert p._legend_on.isChecked() is s.legend_on                # True
    assert p._legend_loc.currentText() == s.legend_loc            # "best"
    assert p._legend_frame.isChecked() is s.legend_frame          # False
    assert p._connect_lines.isChecked() is s.connect_lines        # True
    assert p._thousands.isChecked() is s.thousands_sep            # False
    assert p._spine_w.value() == 0.0                              # 0.0 shows "auto" (None)
    assert p._fit_color.currentText() == "(auto)"                 # None -> "(auto)"
    assert p._fit_linestyle.currentText() == s.fit_linestyle      # "-"


def test_journal_group_collapsed_by_default_and_toggles(qapp):
    p = _panel(qapp)
    assert p._journal_box.isCheckable()
    assert p._journal_box.isChecked() is False                    # collapsed at start
    # use isHidden() not isVisible(): offscreen, an unshown ancestor makes
    # isVisible() False even for shown widgets; isHidden() reflects the explicit flag.
    assert p._journal_content.isHidden() is True
    p._journal_box.setChecked(True)                               # expand
    assert p._journal_content.isHidden() is False


def test_each_journal_setter_emits_with_value(qapp):
    p = _panel(qapp)
    seen = []
    p.style_changed.connect(lambda s: seen.append(s))
    p._set_grid(True);                 assert p.style.grid is True
    p._set_tick_direction("out");      assert p.style.tick_direction == "out"
    p._set_minor_ticks(False);         assert p.style.minor_ticks is False
    p._set_ticks_top(False);           assert p.style.ticks_top is False
    p._set_ticks_right(False);         assert p.style.ticks_right is False
    p._set_grid_style(":");            assert p.style.grid_style == ":"
    p._set_grid_alpha(0.2);            assert abs(p.style.grid_alpha - 0.2) < 1e-9
    p._set_legend_on(False);           assert p.style.legend_on is False
    p._set_legend_loc("outside");      assert p.style.legend_loc == "outside"
    p._set_legend_frame(True);         assert p.style.legend_frame is True
    p._set_connect_lines(False);       assert p.style.connect_lines is False
    p._set_fit_linestyle("--");        assert p.style.fit_linestyle == "--"
    p._set_thousands(True);            assert p.style.thousands_sep is True
    assert len(seen) == 13             # one emit per setter call above


def test_auto_sentinels_map_to_none(qapp):
    p = _panel(qapp)
    p._set_spine_width(1.5);  assert p.style.spine_width == 1.5
    p._set_spine_width(0.0);  assert p.style.spine_width is None        # 0 -> auto/None
    p._set_fit_color("red");  assert p.style.fit_color == "red"
    p._set_fit_color("(auto)"); assert p.style.fit_color is None       # "(auto)" -> None


def test_set_style_restores_journal_widgets_without_emit(qapp):
    p = _panel(qapp)
    emits = []
    p.style_changed.connect(lambda s: emits.append(s))
    restored = GlobalStyle(
        spine_width=1.25, tick_direction="out", minor_ticks=False,
        ticks_top=False, ticks_right=False, grid=True, grid_style=":",
        grid_alpha=0.15, legend_on=False, legend_loc="outside",
        legend_frame=True, connect_lines=False, fit_color="red",
        fit_linestyle="--", thousands_sep=True,
    )
    p.set_style(restored)
    assert p._spine_w.value() == 1.25
    assert p._tick_dir.currentText() == "out"
    assert p._minor_ticks.isChecked() is False
    assert p._ticks_top.isChecked() is False
    assert p._ticks_right.isChecked() is False
    assert p._grid.isChecked() is True
    assert p._grid_style.currentText() == ":"
    assert abs(p._grid_alpha.value() - 0.15) < 1e-9
    assert p._legend_on.isChecked() is False
    assert p._legend_loc.currentText() == "outside"
    assert p._legend_frame.isChecked() is True
    assert p._connect_lines.isChecked() is False
    assert p._fit_color.currentText() == "red"
    assert p._fit_linestyle.currentText() == "--"
    assert p._thousands.isChecked() is True
    assert emits == []                       # restore must NOT emit style_changed


def test_set_style_none_spine_and_fit_color_show_auto(qapp):
    p = _panel(qapp)
    p.set_style(GlobalStyle(spine_width=None, fit_color=None))
    assert p._spine_w.value() == 0.0             # None -> "auto" (0.0)
    assert p._fit_color.currentText() == "(auto)"


def test_legend_loc_combo_offers_explicit_positions(qapp):
    # KNOWN-ISSUES 4, manual half: a user who can see the right spot must be able to say
    # "upper left", not just "inside". Three modes + the nine matplotlib positions.
    p = _panel(qapp)
    items = [p._legend_loc.itemText(i) for i in range(p._legend_loc.count())]
    assert items[:3] == ["best", "inside", "outside"]
    for loc in ("upper right", "upper left", "lower right", "lower left",
                "upper center", "lower center", "center left", "center right", "center"):
        assert loc in items
    p._set_legend_loc("upper left")
    assert p.style.legend_loc == "upper left"


def test_explicit_legend_loc_roundtrips_through_style_sync(qapp):
    p = _panel(qapp)
    s = GlobalStyle(legend_loc="lower center")
    p.set_style(s)
    assert p._legend_loc.currentText() == "lower center"
    assert p.style.legend_loc == "lower center"
