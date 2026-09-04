"""Bare, TAB-separated resistivity exports (Origin 'dc rho' style) must load, auto-detect
as resistivity, and produce a rho(T) curve in Ohm-cm.

Real file: the bare dc-resistivity measurement -- tab-separated, no [Header]/[Data],
columns 'T (K)' + 'Resistivity ... (mikroOhm-cm)_H=0T_COOL', no Field column. Previously the
loader hardcoded comma -> read as one column -> every probe errored.
"""
import pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.detect.probe import detect_probe
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry

FIX = pathlib.Path(__file__).parent / "fixtures"
BARE = str(FIX / "bare_rho_synth.dat")
# oracle from make_bare_rho.py: rho = 50 + 0.5*T uOhm-cm -> Ohm-cm at 300K = 2e-4
RHO_300_OHM_CM = 2.0e-4
# RRR uses the median of the 5 physical points nearest each T-extreme (not exact endpoints),
# so it sits a little below the naive rho(300)/rho(2)=3.92; assert the sensible metallic range.


def test_load_tab_separated_bare_file_gives_two_columns():
    rt = load_dat(BARE)
    assert rt.df.shape[1] == 2, f"tab file must parse to 2 columns, got {rt.df.shape}"
    assert rt.header.bare_csv is True


def test_columns_canonicalize_temperature_and_resistivity():
    rt = load_dat(BARE)
    _, cmap = canonicalize_columns(rt.df, rt.header)
    assert "temperature" in cmap.logical          # 'T (K)' must map
    assert "resistivity_ch1" in cmap.logical       # generic 'Resistivity ... (uOhm-cm)' -> ch1


def test_autodetects_as_resistivity():
    rt = load_dat(BARE)
    _, cmap = canonicalize_columns(rt.df, rt.header)
    score, key = detect_probe(rt.header, set(rt.df.columns), build_default_registry())
    assert key == "resistivity" and score >= 0.5, (score, key)


def test_analyze_produces_rho_t_in_ohm_cm():
    res = analyze_file(load_dat(BARE), RunConfig.load(probe_override="resistivity"),
                       build_default_registry())
    assert res.status in ("ok", "low_confidence"), (res.status, res.errors)
    bridges = (res.data or {}).get("bridges", [])
    rho_t = [c for b in bridges for c in b.get("rho_t_curves", [])]
    assert rho_t, "expected a rho(T) curve"
    rho = rho_t[0]["rho"]
    hi = max(rho)
    # micro-ohm-cm (~200) must be converted to Ohm-cm (~2e-4), not left raw or scaled wrong
    assert abs(hi - RHO_300_OHM_CM) / RHO_300_OHM_CM < 0.02, f"rho_max={hi} not ~{RHO_300_OHM_CM} Ohm-cm"


def test_no_field_column_label_is_not_na_oe():
    # with no Field column the legend must not fabricate "na Oe"; it shows the curve label.
    res = analyze_file(load_dat(BARE), RunConfig.load(probe_override="resistivity"),
                       build_default_registry())
    from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
    kinds = {k.key: k for k in BUILTIN_PLOTKINDS}
    labels = [s.label for s in kinds["resistivity_rho_t"].series(res)]
    assert labels and all("na" not in l for l in labels), labels
    assert "ρ(T)" in labels[0]


def test_rrr_capability_on_zero_field_ramp():
    res = analyze_file(load_dat(BARE), RunConfig.load(probe_override="resistivity"),
                       build_default_registry())
    caps = (res.data or {}).get("bridges", [{}])[0]
    rrr = None
    for b in (res.data or {}).get("bridges", []):
        if b.get("rrr") is not None:
            rrr = b["rrr"]; break
    assert rrr is not None, "RRR should compute on a single zero-field ramp (no Field column)"
    assert 3.5 < rrr < 4.0, f"RRR={rrr} outside expected metallic range (~3.7, below naive 3.92)"
