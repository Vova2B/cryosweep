def test_app_module_imports_and_builds_window(qapp):
    from cryosweep_gui.app import build_window
    win = build_window()
    assert win is not None
    assert win.windowTitle()            # has a non-empty title


def test_analysis_state_caches_and_patches(vsm_path):
    import dataclasses
    from cryosweep_gui.state import AnalysisState
    st = AnalysisState()
    assert st.get_raw() is None
    st.load(vsm_path)
    rt = st.get_raw()
    assert rt is not None and rt.df is not None
    # header patch produces a COPY; the cached RawTable is never mutated
    patched = st.patched_raw({"molar_mass": 200.0, "mass_mg": 5.0})
    assert patched.header.molar_mass == 200.0 and patched.header.mass_mg == 5.0
    assert st.get_raw() is rt                          # cached object identity unchanged
    # empty patch returns the cached object unchanged
    assert st.patched_raw({}) is rt
    # per-probe result cache
    st.cache_result("vsm", "RESULT_SENTINEL")
    assert st.get_result("vsm") == "RESULT_SENTINEL"
    assert st.get_result("hall") is None
