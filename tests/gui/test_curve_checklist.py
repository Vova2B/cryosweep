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


def _tdep_result(points):
    from cryosweep_core.result import Result, Provenance
    return Result(status="ok", confidence=1.0,
                  data={"probe": "hall_tdep", "points": points},
                  provenance=Provenance(file="x", sha256="", app_version="", config={}))


def test_a_declined_series_is_listed_but_unchecked(qapp):
    """The inspection series for points the decline rule withheld must be REACHABLE from the
    checklist and OFF until a user asks for it.

    Built from the real catalog builder, not from hand-made Series: the checklist is the only
    surface through which a GUI user can reach this feature at all, and a test that constructs
    its own Series would pass just as happily with the feature removed. Verified RED against a
    neutralised `_hall_withheld` before being written this way.
    """
    from cryosweep_gui.plot_controls import CurveChecklist
    from cryosweep_core.plotting.catalog import series_hall_tdep_n_t
    res = _tdep_result([
        {"temperature": 10.0, "carrier_n": 1e28, "withheld": None, "r_h_method": "antisym"},
        {"temperature": 20.0, "carrier_n": None, "r_h_method": "antisym",
         "withheld": {"carrier_n": 5e28, "carrier_type": "holes", "mobility": 2e-3}},
    ])
    w = CurveChecklist(series_hall_tdep_n_t(res))
    labels = [w._list.item(i).text() for i in range(w._list.count())]
    assert "n (declined)" in labels, "a user cannot enable what the checklist does not list"
    assert w.checked_keys() == ["n_antisym"], "declined points must not be on by default"
    w.select_all()
    assert set(w.checked_keys()) == {"n_antisym", "n_withheld"}
