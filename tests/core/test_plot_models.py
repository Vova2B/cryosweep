from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec, PlotEntry, PlotLayout

def test_globalstyle_defaults_preserve_today():
    s = GlobalStyle()
    assert (s.width_mm, s.height_mm, s.dpi) == (90.0, 70.0, 300)
    assert s.font_pt == 9.0 and s.marker == "o" and s.marker_size == 3.0
    assert s.line_width == 1.0 and s.font_family is None and s.palette is None and s.color is None

def test_plotspec_is_per_plot_only():
    p = PlotSpec()
    assert (p.xmin, p.xmax, p.ymin, p.ymax) == (None, None, None, None)
    assert p.xscale is None and p.yscale is None
    assert p.curves is None and p.fit_line is True and p.title is None
    # styling fields moved OUT to GlobalStyle; width_mm/height_mm returned in
    # PQ-1 2b as nullable per-plot EXPORT-size overrides (None -> GlobalStyle)
    assert not hasattr(p, "marker") and not hasattr(p, "font_pt")
    assert p.width_mm is None and p.height_mm is None

def test_layout_roundtrip_is_serializable():           # the B contract
    lay = PlotLayout(plots=[PlotEntry(kind="inverse_chi",
                                      spec=PlotSpec(yscale="log", curves=["b1:T:0:0"]))])
    again = PlotLayout.model_validate(lay.model_dump())
    assert again.plots[0].kind == "inverse_chi"
    assert again.plots[0].spec.yscale == "log"
    assert again.plots[0].spec.curves == ["b1:T:0:0"]

def test_two_entries_have_independent_specs():         # default_factory, not shared mutable
    a, b = PlotEntry(kind="x"), PlotEntry(kind="y")
    a.spec.curves = ["z"]
    assert b.spec.curves is None

def test_globalstyle_roundtrip():
    from cryosweep_core.plotting.spec import GlobalStyle
    s = GlobalStyle(palette=["#ff0000"], line_width=2.0, color="#ff0000")
    assert GlobalStyle.model_validate(s.model_dump()) == s

def test_plotspec_rejects_invalid_scale():
    import pytest
    from cryosweep_core.plotting.spec import PlotSpec
    with pytest.raises(Exception):
        PlotSpec(xscale="symlog")

def test_plotspec_fit_lines_defaults_none():
    from cryosweep_core.plotting.spec import PlotSpec
    assert PlotSpec().fit_lines is None

def test_plotspec_fit_lines_accepts_subset_and_empty():
    from cryosweep_core.plotting.spec import PlotSpec
    assert PlotSpec(fit_lines=("linear",)).fit_lines == ("linear",)
    assert PlotSpec(fit_lines=()).fit_lines == ()

def test_plotspec_fit_lines_roundtrips_json():
    from cryosweep_core.plotting.spec import PlotSpec
    s = PlotSpec(fit_lines=("linear", "power_law"))
    s2 = PlotSpec.model_validate_json(s.model_dump_json())
    assert s2.fit_lines == ("linear", "power_law")

def test_plotspec_without_fit_lines_validates_backcompat():
    # an old persisted spec dict (no fit_lines key) must still validate
    from cryosweep_core.plotting.spec import PlotSpec
    s = PlotSpec.model_validate({"curves": None, "fit_line": True})
    assert s.fit_lines is None
