from cryosweep_core.plotting.catalog import Series, PlotKind, series_label, select_series
from cryosweep_core.plotting.spec import PlotSpec

def _series():
    return [Series(key="a", label="A", x=[1], y=[1], group="G1", default_on=True),
            Series(key="b", label="B", x=[2], y=[2], group="G1", default_on=False),
            Series(key="c", label="C", x=[3], y=[3], group="G2", default_on=True)]

def test_select_none_uses_default_on():
    sel = select_series(_series(), PlotSpec())          # curves=None
    assert [s.key for s in sel] == ["a", "c"]

def test_select_empty_list_selects_none():
    assert select_series(_series(), PlotSpec(curves=[])) == []

def test_select_explicit_keys_in_order_of_series():
    sel = select_series(_series(), PlotSpec(curves=["c", "b"]))
    assert [s.key for s in sel] == ["b", "c"]

def test_series_label_is_plain_label_in_A():
    s = _series()[0]
    assert series_label(object(), s) == "A"

def test_plotkind_carries_metadata():
    k = PlotKind(key="k", label="K", probe="vsm", series=lambda r: [], default_yscale="log")
    assert k.probe == "vsm" and k.default_yscale == "log" and k.default_xscale == "linear"
