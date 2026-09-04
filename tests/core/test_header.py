from cryosweep_core.io.header import parse_header

def test_parse_hc_header(hc_path):
    h = parse_header(hc_path)
    assert h.app == "HeatCapacity"
    assert h.app_version == "3.9.6"
    assert h.molar_mass == 945.68
    assert h.n_atoms == 1.0
    assert h.mass_mg == 4.5
    # The title is the source filename and is local-only, so pin the parse, not the identity:
    # a non-empty .dat title proves the TITLE line was read (spec §2d).
    assert h.title and h.title.endswith(".dat")

def test_parse_res_header_no_molwght(res_path):
    h = parse_header(res_path)
    assert h.app == "Resistivity"
    assert h.molar_mass is None
    assert h.n_atoms is None
    # Res INFO rows have no KEY: prefix — stored by description
    assert any("Cross Section" in d for (_k, _v, d) in h.info_rows)

def test_header_is_frozen(hc_path):
    import dataclasses
    h = parse_header(hc_path)
    with __import__("pytest").raises(dataclasses.FrozenInstanceError):
        h.app = "x"
