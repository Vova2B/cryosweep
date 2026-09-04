from cryosweep_core.plotting.catalog import Series
from cryosweep_gui.plot_controls import CurveChecklist

def _series():
    return [Series(key="a1", label="alpha 1", x=[0], y=[0], group="GroupA", default_on=True),
            Series(key="a2", label="alpha 2", x=[0], y=[0], group="GroupA", default_on=True),
            Series(key="b1", label="beta 1",  x=[0], y=[0], group="GroupB", default_on=True)]

def _header(w, name):
    for i in range(w._list.count()):
        if w._list.item(i).text() == f"— {name} —":
            return w._list.item(i)
    raise AssertionError(f"header {name} not found")

def test_group_header_hidden_when_all_members_filtered(qapp):
    w = CurveChecklist(_series())
    w.set_filter("alpha")                              # only GroupA members match
    assert _header(w, "GroupB").isHidden()             # empty group -> header hidden
    assert not _header(w, "GroupA").isHidden()         # GroupA header stays
    w.set_filter("")                                   # cleared -> all back
    assert not _header(w, "GroupB").isHidden()
