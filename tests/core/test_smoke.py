def test_import_cryosweep_core_is_clean():
    import cryosweep_core  # must not import Qt/matplotlib
    assert cryosweep_core.__version__ == "0.1.0"
