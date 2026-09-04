from cryosweep_core.registry import build_default_registry

def test_plot_kinds_registered_and_grouped_by_probe():
    r = build_default_registry()
    assert "inverse_chi" in r.plotkind_keys()
    vsm = [k.key for k in r.plot_kinds_for("vsm")]
    assert vsm == ["inverse_chi", "vsm_moment_t", "vsm_chi_t", "vsm_chi_t_product", "vsm_mh"]   # catalog order preserved
    assert [k.key for k in r.plot_kinds_for("hall")] == [
        "hall_rh_t", "hall_mobility_t", "hall_n_t", "hall_r2_t",
        "hall_rxy_vs_B", "hall_asym_vs_B", "hall_raw_vs_asym",
        "hall_two_panel", "hall_rh_n_twin"]   # PQ-2 Task 3 composites appended (catalog order)
    assert r.plot_kinds_for("nope") == []

def test_building_registry_does_not_import_matplotlib():
    import sys
    sys.modules.pop("matplotlib", None)
    build_default_registry()
    assert "matplotlib" not in sys.modules

def test_hall_tdep_plotkinds_exact():
    from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
    keys = [k.key for k in BUILTIN_PLOTKINDS if k.probe == "hall_tdep"]
    assert keys == [
        "hall_tdep_RH_T",
        "hall_tdep_n_T",
        "hall_tdep_mobility_T",
        "hall_tdep_asym_vs_B",
        "hall_tdep_interp_RT",
        "hall_tdep_stages",
        "hall_tdep_J_T",
        "hall_tdep_summary",
        "hall_tdep_rh_n_twin",
    ]
