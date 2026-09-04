import pandas as pd
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.registry import build_default_registry


def test_real_acms_detects_as_acms_not_vsm(acms_real_path):
    rt = load_dat(acms_real_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    score, key = detect_probe(rt.header, set(df.columns), build_default_registry())
    assert key == "acms" and score >= 0.5


def test_acms_columns_canonicalize(acms_real_path):
    rt = load_dat(acms_real_path)
    _, cmap = canonicalize_columns(rt.df, rt.header)
    for k in ("m_prime", "m_dprime", "frequency", "amplitude"):
        assert k in cmap.logical, k
    assert cmap.unit["m_prime"] == "emu" and cmap.unit["amplitude"] == "Oe"


def test_existing_fixtures_unchanged(res_path, hc_path):
    reg = build_default_registry()
    for path, want in [(res_path, "resistivity"), (hc_path, "heatcapacity")]:
        rt = load_dat(path)
        df, _ = canonicalize_columns(rt.df, rt.header)
        _, key = detect_probe(rt.header, set(df.columns), reg)
        assert key == want
