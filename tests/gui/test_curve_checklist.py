from cryosweep_core.plotting.catalog import Series


def _many(n):                                            # n series across 3 groups
    return [Series(key=f"k{i}", label=f"L{i}", x=[0], y=[0],
                   group=f"Bridge {i % 3}", default_on=(i < 2)) for i in range(n)]


def test_checklist_height_bounded_with_many_items(qapp):
    from cryosweep_gui.plot_controls import CurveChecklist
    w = CurveChecklist(_many(200))
    assert w._list.maximumHeight() <= 200                # fixed height -> scrolls, never collapses (P1)
    assert w._list.count() >= 200


def test_default_on_drives_initial_checks(qapp):
    from cryosweep_gui.plot_controls import CurveChecklist
    w = CurveChecklist(_many(5))
    assert set(w.checked_keys()) == {"k0", "k1"}


def test_select_all_none_invert(qapp):
    from cryosweep_gui.plot_controls import CurveChecklist
    w = CurveChecklist(_many(5))
    w.select_all();  assert len(w.checked_keys()) == 5
    w.select_none(); assert w.checked_keys() == []
    w.invert();      assert len(w.checked_keys()) == 5


def test_filter_limits_select_all_scope(qapp):
    from cryosweep_gui.plot_controls import CurveChecklist
    w = CurveChecklist(_many(6))
    w.set_filter("L0")                                   # matches "L0" only
    w.select_none(); w.select_all()
    assert w.checked_keys() == ["k0"]


def test_emits_on_change(qapp):
    from cryosweep_gui.plot_controls import CurveChecklist
    seen = []
    w = CurveChecklist(_many(3))
    w.changed.connect(lambda keys: seen.append(list(keys)))
    w.select_all()
    assert seen and set(seen[-1]) == {"k0", "k1", "k2"}
