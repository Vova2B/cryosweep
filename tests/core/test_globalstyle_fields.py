import pytest
from pydantic import ValidationError
from cryosweep_core.plotting.spec import GlobalStyle

def test_new_fields_default_to_none_or_false():
    s = GlobalStyle()
    assert s.label_size is None and s.title_size is None and s.tick_size is None and s.legend_size is None
    assert s.edge_color is None and s.edge_width is None
    assert s.colormap is None and s.colormap_reverse is False

def test_existing_defaults_unchanged():
    s = GlobalStyle()
    assert (s.width_mm, s.height_mm, s.dpi, s.font_pt) == (90.0, 70.0, 300, 9.0)
    assert s.marker == "o" and s.marker_size == 3.0 and s.line_width == 1.0

def test_positivity_constraints_reject_nonpositive():
    for bad in ({"dpi": 0}, {"width_mm": -1}, {"font_pt": 0}, {"label_size": 0}, {"edge_width": -2}):
        with pytest.raises(ValidationError):
            GlobalStyle(**bad)

def test_roundtrip_preserves_new_fields():
    s = GlobalStyle(label_size=12, colormap="viridis", colormap_reverse=True, edge_color="black", edge_width=0.5)
    again = GlobalStyle.model_validate_json(s.model_dump_json())
    assert again.label_size == 12 and again.colormap == "viridis" and again.colormap_reverse is True
    assert again.edge_color == "black" and again.edge_width == 0.5

def test_robust_view_fields_default_on():
    from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec
    s = GlobalStyle()
    assert s.robust_view is True and s.robust_k == 8.0
    assert PlotSpec().robust_view is None

def test_field_unit_defaults_to_oe():
    s = GlobalStyle()
    assert s.field_unit == "Oe"

def test_field_unit_roundtrips():
    s = GlobalStyle(field_unit="T")
    again = GlobalStyle.model_validate_json(s.model_dump_json())
    assert again.field_unit == "T"

def test_field_unit_rejects_unknown():
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GlobalStyle(field_unit="Gauss")
