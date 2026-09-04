import numpy as np
from cryosweep_core.model import Axis, Segment, SegmentGrid

def _seg(setpoint_T, x, y, branch="up", direction=1):
    return Segment(
        swept=Axis(name="field", column="Magnetic Field (Oe)", unit="Oe"),
        direction=direction, branch=branch,
        fixed={"temperature": setpoint_T}, tol={"temperature": 0.1},
        setpoint={"temperature": setpoint_T},
        idx=np.arange(len(x)), confidence=1.0,
        x=np.asarray(x, float), data={"r": np.asarray(y, float)},
    )

def test_by_fixed_groups_isotherms():
    segs = [_seg(3.0, [0,1,2], [10,11,12]), _seg(5.0, [0,1,2], [20,21,22])]
    grid = SegmentGrid(segs)
    by_T = grid.by_fixed("temperature")
    assert set(by_T) == {3.0, 5.0}

def test_on_common_axis_interpolates_no_extrapolation():
    segs = [_seg(3.0, [0,1,2,3], [0,10,20,30]), _seg(5.0, [1,2,3,4], [5,10,15,20])]
    grid = SegmentGrid(segs)
    xgrid, cols = grid.on_common_axis("field", ["r"])
    # common overlap is [1,3]; no extrapolation beyond each segment's range
    assert xgrid.min() >= 1.0 and xgrid.max() <= 3.0
    assert np.all(np.isfinite(cols[(3.0,)]))
    assert np.all(np.isfinite(cols[(5.0,)]))

def test_on_common_axis_collapses_duplicate_x():
    segs = [_seg(3.0, [1,1,2,3], [10,12,20,30])]
    grid = SegmentGrid(segs)
    xgrid, cols = grid.on_common_axis("field", ["r"])
    # duplicate x=1 collapsed by mean -> 11 at x=1
    assert np.isclose(np.interp(1.0, xgrid, cols[(3.0,)]), 11.0, atol=1e-6)


def _mixed_seg(swept_name, setpoint):
    # mixed-mode segment: its swept axis stores setpoint[axis]=None
    return Segment(
        swept=Axis(name=swept_name, column=swept_name, unit=""),
        direction=1, branch="up", fixed=dict(setpoint), tol={},
        setpoint=dict(setpoint),
        idx=np.arange(3), confidence=1.0,
        x=np.array([0.0, 1.0, 2.0]), data={"r": np.array([1.0, 2.0, 3.0])},
    )


def test_branches_skips_none_setpoint():
    # Bug 5: a field-swept segment storing setpoint['field']=None must not crash
    # np.isclose(None, value)
    segs = [
        _mixed_seg("temperature", {"temperature": None, "field": 0.0}),
        _mixed_seg("temperature", {"temperature": None, "field": 50000.0}),
        _mixed_seg("field", {"field": None, "temperature": 2.0}),
    ]
    grid = SegmentGrid(segs)
    br = grid.branches(temperature=2.0)            # must not raise TypeError
    assert all(s.setpoint.get("temperature") is not None for s in br.values())


def test_by_fixed_skips_none_key():
    # Bug 5: segments whose swept-axis setpoint is None must not bucket under None
    segs = [
        _mixed_seg("field", {"field": None, "temperature": 2.0}),
        _mixed_seg("field", {"field": None, "temperature": 5.0}),
    ]
    grid = SegmentGrid(segs)
    by_f = grid.by_fixed("field")                  # must not raise
    assert None not in by_f
