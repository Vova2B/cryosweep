import pytest

def test_base_panel_defaults(qapp):
    from cryosweep_gui.inputs.base import InputPanel
    p = InputPanel("vsm")
    assert p.probe_key == "vsm"
    assert p.build_overrides() == {}
    assert p.build_header_patch() == {}

def test_generic_needs_panel_lists_needs(qapp):
    from cryosweep_gui.inputs.base import GenericNeedsPanel
    p = GenericNeedsPanel("resistivity", needs=[{"key": "width_mm"}, {"key": "length_mm"}])
    assert p.probe_key == "resistivity"
    assert p.build_overrides() == {}
    assert p.build_header_patch() == {}
    assert "width_mm" in p.needs_text() and "length_mm" in p.needs_text()

def test_build_panel_uses_registry_else_generic(qapp):
    from cryosweep_gui.inputs.base import build_panel, INPUT_PANELS
    import cryosweep_gui.inputs   # noqa: F401  (registers all bespoke panels)
    vsm = build_panel("vsm", needs=[])
    assert type(vsm).__name__ == "VSMInputPanel"
    # NO built-in probe is unregistered anymore (all 4 have bespoke panels); use a fake key.
    generic = build_panel("_unregistered_probe_", needs=[{"key": "thickness_mm"}])
    assert type(generic).__name__ == "GenericNeedsPanel"

def test_vsm_panel_header_patch(qapp):
    import cryosweep_gui.inputs.vsm   # noqa: F401
    from cryosweep_gui.inputs.base import build_panel
    p = build_panel("vsm")
    assert type(p).__name__ == "VSMInputPanel"
    assert p.build_overrides() == {}                  # VSM has no RunConfig overrides (unit is global)
    assert p.build_header_patch() == {}               # empty fields -> no patch
    p.molar_mass_edit.setText("200.5")
    p.mass_mg_edit.setText("5.0")
    assert p.build_header_patch() == {"molar_mass": 200.5, "mass_mg": 5.0}
    p.molar_mass_edit.setText("")                     # blank -> omitted
    assert p.build_header_patch() == {"mass_mg": 5.0}
    p.molar_mass_edit.setText("abc")                  # non-numeric -> omitted (no crash)
    assert "molar_mass" not in p.build_header_patch()

def test_resistivity_panel_geometry_overrides(qapp):
    import cryosweep_gui.inputs.resistivity   # noqa: F401
    from cryosweep_gui.inputs.base import build_panel
    p = build_panel("resistivity")
    assert type(p).__name__ == "ResistivityInputPanel"
    assert p.build_header_patch() == {}                 # resistivity uses no header fields
    assert p.build_overrides() == {}                    # empty fields -> no geometry (instrument fallback)
    p.width_edit.setText("2.0"); p.thickness_edit.setText("0.5"); p.length_edit.setText("3.0")
    assert p.build_overrides() == {"geometry": {"width_mm": 2.0, "thickness_mm": 0.5, "length_mm": 3.0}}
    p.width_edit.setText("")                            # partial geometry -> only present keys
    assert p.build_overrides() == {"geometry": {"thickness_mm": 0.5, "length_mm": 3.0}}
    p.thickness_edit.setText("abc")                     # non-numeric -> omitted, no crash
    assert "thickness_mm" not in p.build_overrides()["geometry"]

def test_hc_panel_n_atoms_header_patch(qapp):
    import cryosweep_gui.inputs.hc   # noqa: F401
    from cryosweep_gui.inputs.base import build_panel
    p = build_panel("heatcapacity")
    assert type(p).__name__ == "HCInputPanel"
    ov = p.build_overrides()                             # HC now returns full_init/full_fixed defaults
    assert "heatcapacity" in ov
    assert "full_init" in ov["heatcapacity"] and "full_fixed" in ov["heatcapacity"]
    assert p.build_header_patch() == {}                 # empty -> use header's n_atoms
    p.n_atoms_edit.setText("5")
    assert p.build_header_patch() == {"n_atoms": 5.0}
    p.n_atoms_edit.setText("abc")                       # non-numeric -> omitted, no crash
    assert p.build_header_patch() == {}

def test_hall_panel_build_overrides(qapp):
    import cryosweep_gui.inputs.hall   # noqa: F401
    from cryosweep_gui.inputs.base import build_panel
    p = build_panel("hall")
    assert type(p).__name__ == "HallInputPanel"
    assert p.build_header_patch() == {}                          # hall uses no header fields
    # geometry_sign is always emitted from the combo (default +1); cfg default is also 1 -> parity-safe
    assert p.build_overrides() == {"hall": {"geometry_sign": 1}}
    p.hall_channel_edit.setText("1")
    p.thickness_edit.setText("0.1"); p.thickness_unit.setCurrentText("mm")
    p.long_channel_edit.setText("2")
    assert p.build_overrides() == {"hall": {"hall_channel": 1, "thickness_mm": 0.1,
                                            "geometry_sign": 1, "longitudinal_channel": 2}}
    # thickness unit conversion: 100 um == 0.1 mm
    p.thickness_edit.setText("100"); p.thickness_unit.setCurrentText("um")
    assert p.build_overrides()["hall"]["thickness_mm"] == pytest.approx(0.1)
    # geometry sign -1
    p.geometry_sign.setCurrentText("-1")
    assert p.build_overrides()["hall"]["geometry_sign"] == -1
    # separate longitudinal file path flows through
    p.set_longitudinal_file("/tmp/long.dat")
    assert p.build_overrides()["hall"]["longitudinal_file"] == "/tmp/long.dat"
    p.set_longitudinal_file("")                                  # clearing reverts to same-file
    assert "longitudinal_file" not in p.build_overrides()["hall"]
    # non-numeric channel -> omitted, no crash
    p.hall_channel_edit.setText("abc")
    assert "hall_channel" not in p.build_overrides()["hall"]

def test_hall_panel_state_roundtrips_all_inputs(qapp):
    """Regression (owner 2026-07-09): get_state/set_state carried only hall_channel+thickness,
    so per-file flows (analyze_and_render's set_state) silently dropped geometry sign, thickness
    unit, longitudinal channel and longitudinal file."""
    import cryosweep_gui.inputs.hall   # noqa: F401
    from cryosweep_gui.inputs.base import build_panel
    p = build_panel("hall")
    p.hall_channel_edit.setText("2")
    p.thickness_edit.setText("0.5")
    p.thickness_unit.setCurrentText("um")
    p.geometry_sign.setCurrentText("-1")
    p.long_channel_edit.setText("1")
    p.set_longitudinal_file("/tmp/long.dat")
    state = p.get_state()
    q = build_panel("hall")
    q.set_state(state)
    assert q.build_overrides() == p.build_overrides()
    assert q.long_file_label.text() == "/tmp/long.dat"
    # blank state restores defaults (no stale carry-over between file entries)
    q.set_state({})
    assert q.build_overrides() == {"hall": {"geometry_sign": 1}}

def test_hall_panel_sign_change_requests_refit(qapp):
    """Regression (owner 2026-07-09): flipping +1 -> -1 changed nothing until Analyze was
    clicked; the panel now requests a re-analysis itself (probe_tab connects refit_requested)."""
    import cryosweep_gui.inputs.hall   # noqa: F401
    from cryosweep_gui.inputs.base import build_panel
    p = build_panel("hall")
    hits = []
    p.refit_requested.connect(lambda: hits.append(1))
    p.geometry_sign.setCurrentText("-1")
    assert len(hits) == 1
    # programmatic state restore must NOT re-trigger analysis (analyze_and_render calls
    # set_state per file entry -> would recurse)
    p.set_state({"geometry_sign": "+1"})
    assert len(hits) == 1
