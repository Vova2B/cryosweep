from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns

def test_canonicalize_hc(hc_path):
    rt = load_dat(hc_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    assert cmap.logical["temperature"] == "Sample Temp (Kelvin)"
    assert cmap.logical["field"] == "Field (Oersted)"
    assert cmap.unit["temperature"] == "K"
    assert cmap.unit["field"] == "Oe"

def test_canonicalize_res(res_path):
    rt = load_dat(res_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    assert cmap.logical["temperature"] == "Temperature (K)"
    assert cmap.logical["field"] == "Magnetic Field (Oe)"
    assert "resistivity_ch1" in cmap.logical
    assert cmap.logical["resistivity_ch1"] == "Bridge 1 Resistivity (Ohm-m)"

def test_hc_sample_maps_to_cp_not_cp_over_t(hc_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.io.columns import canonicalize_columns
    rt = load_dat(hc_path)
    _, cmap = canonicalize_columns(rt.df, rt.header)
    assert cmap.logical["hc_sample"] == "Samp HC (mJ/mole-K)"     # NOT Samp HC/Temp, NOT Samp HC Err

def test_encoding_insensitive_match(hc_path):
    # the mangled-µ Addenda column must be discoverable by a clean key
    rt = load_dat(hc_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    from cryosweep_core.io.columns import _norm
    cols = {_norm(c) for c in df.columns}
    assert _norm("Addenda HC (J/K)") in cols or any("addenda hc" in c for c in cols)

def test_res_canonicalizes_resistance_columns(res_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.io.columns import canonicalize_columns
    rt = load_dat(res_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    # raw resistance columns must be exposed for the geometry-recompute path
    assert cmap.logical["resistance_ch1"] == "Bridge 1 Resistance (Ohms)"
    assert cmap.logical["resistance_ch2"] == "Bridge 2 Resistance (Ohms)"
    assert cmap.unit["resistance_ch1"] == "Ohm"
    # resistivity mapping still works and is NOT clobbered by the resistance rule
    assert cmap.logical["resistivity_ch1"] == "Bridge 1 Resistivity (Ohm-m)"
