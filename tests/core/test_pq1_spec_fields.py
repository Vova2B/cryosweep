from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec, ReferenceLine

def test_globalstyle_journal_defaults():
    s = GlobalStyle()
    assert s.tick_direction == "in"
    assert s.minor_ticks is True and s.ticks_top is True and s.ticks_right is True
    assert s.grid is False and s.grid_style == "--"
    assert s.connect_lines is True
    assert s.legend_on is True and s.legend_loc == "best" and s.legend_frame is False
    assert s.fit_color is None and s.fit_linestyle == "-"
    assert s.thousands_sep is False
    assert s.spine_width is None

def test_plotspec_overrides_default_none():
    p = PlotSpec()
    assert p.grid is None and p.connect_lines is None
    assert p.legend_on is None and p.legend_loc is None
    assert p.reference_lines is None

def test_reference_line_model():
    rl = ReferenceLine(axis="h", value=0.0, label="M=0")
    assert rl.axis == "h" and rl.value == 0.0 and rl.label == "M=0"
    assert rl.color == "black" and rl.linestyle == "-" and rl.linewidth == 0.8
    p = PlotSpec(reference_lines=[ReferenceLine(axis="v", value=30.8, label="T_N")])
    assert p.reference_lines[0].axis == "v"
