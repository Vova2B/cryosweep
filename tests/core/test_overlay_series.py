from cryosweep_core.plotting.catalog import Series, OverlayFile, overlay_series

class _Kind:
    def series(self, result, field_unit="Oe"):   # field_unit threaded by Task 3/5
        return result   # test stub: result IS the series list

def test_overlay_series_file_qualifies_keys_and_groups():
    rA = [Series(key="curve", label="1/χ", x=[1], y=[1], default_on=True)]
    rB = [Series(key="curve", label="1/χ", x=[2], y=[2], default_on=True)]
    ov = [OverlayFile(file_id=0, label="sampleA"), OverlayFile(file_id=1, label="sampleB")]
    out = overlay_series(_Kind(), [rA, rB], ov)
    assert [s.key for s in out] == ["0::curve", "1::curve"]      # same raw key, distinct effective keys
    assert [s.group for s in out] == ["sampleA", "sampleB"]      # grouped by file label
    assert [s.label for s in out] == ["sampleA · 1/χ", "sampleB · 1/χ"]
    assert all(s.default_on for s in out)

def test_overlay_file_is_frozen_with_optional_colour():
    of = OverlayFile(file_id=3, label="x")
    assert of.colour is None and of.file_id == 3
