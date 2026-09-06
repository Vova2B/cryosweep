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


# --- QD VSM writes SAMPLE_MASS / SAMPLE_MOLECULAR_WEIGHT with NO "KEY:" prefix ---
# Every shipped example uses the colon-keyed dialect (`INFO,<mg>,MASS:Sample Mass (mg)`),
# so these run on a synthetic header rather than a real file: the defect they pin was
# invisible for exactly as long as it was only reachable through real data.

_VSM_BARE_HEADER = """\
[Header]
TITLE,synthetic_vsm.dat
BYAPP,VSM,1.0
INFO,synthetic,SAMPLE_MATERIAL
INFO,2.5,SAMPLE_MASS
INFO,150,SAMPLE_MOLECULAR_WEIGHT
INFO,,SAMPLE_VOLUME
[Data]
Temperature (K),Magnetic Field (Oe),Moment (emu)
300.0,1000.0,1.0e-4
"""

def _write(tmp_path, text, name="vsm_bare.dat"):
    p = tmp_path / name
    p.write_text(text)
    return p

def test_vsm_bare_sample_mass_is_parsed(tmp_path):
    """QD VSM writes `INFO,<mg>,SAMPLE_MASS` — no colon, so the KEY: branch never fires."""
    h = parse_header(_write(tmp_path, _VSM_BARE_HEADER))
    assert h.mass_mg == 2.5

def test_vsm_bare_molecular_weight_is_parsed(tmp_path):
    """`INFO,<g/mol>,SAMPLE_MOLECULAR_WEIGHT` is the VSM spelling of MOLWGHT."""
    h = parse_header(_write(tmp_path, _VSM_BARE_HEADER))
    assert h.molar_mass == 150.0

def test_vsm_bare_non_numeric_mass_stays_none(tmp_path):
    """An anonymized example ships `INFO,anonymized,SAMPLE_MASS`; it must not raise or coerce."""
    text = _VSM_BARE_HEADER.replace("INFO,2.5,SAMPLE_MASS", "INFO,anonymized,SAMPLE_MASS")
    h = parse_header(_write(tmp_path, text))
    assert h.mass_mg is None
    assert h.molar_mass == 150.0

def test_vsm_bare_empty_value_stays_none(tmp_path):
    """`INFO,,SAMPLE_VOLUME` is the empty-INFO idiom; an empty mass must stay unset."""
    text = _VSM_BARE_HEADER.replace("INFO,2.5,SAMPLE_MASS", "INFO,,SAMPLE_MASS")
    h = parse_header(_write(tmp_path, text))
    assert h.mass_mg is None

def test_colon_keyed_dialect_still_wins(tmp_path):
    """Both dialects in one header: the explicit MASS: key must not be shadowed."""
    text = _VSM_BARE_HEADER.replace(
        "INFO,2.5,SAMPLE_MASS",
        "INFO,2.5,SAMPLE_MASS\nINFO,9.9,MASS:Sample Mass (mg)")
    h = parse_header(_write(tmp_path, text))
    assert h.mass_mg == 9.9
