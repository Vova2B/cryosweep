import pathlib, dataclasses, re
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.registry import build_default_registry
from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS, build_default_layout

FIX = pathlib.Path(__file__).parent / "fixtures"
KINDS = {k.key: k for k in BUILTIN_PLOTKINDS}

def _vsm():
    rt = load_dat(str(FIX / "vsm_synth.dat"))
    rt = dataclasses.replace(rt, header=dataclasses.replace(rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(probe_override="vsm"), build_default_registry())

def _res():
    return analyze_file(load_dat(str(FIX / "act_synth.dat")),
                        RunConfig.load(probe_override="resistivity"), build_default_registry())

def test_vsm_kinds_single_curve_default_on():
    res = _vsm()
    for key in ("inverse_chi", "vsm_moment_t"):
        s = KINDS[key].series(res)
        assert len(s) == 1 and s[0].default_on and len(s[0].x) == len(s[0].y) > 0
    # PQ-3 Item 2: vsm_chi_t emits χ (key "curve") + χ⁻¹ (key "inv_chi", role tag), both
    # default_on for the twin-axis default view.
    s = KINDS["vsm_chi_t"].series(res)
    assert [sr.key for sr in s] == ["curve", "inv_chi"]
    assert all(sr.default_on and len(sr.x) == len(sr.y) > 0 for sr in s)
    assert s[1].role == "inv_chi"

def test_resistivity_rho_t_one_default_on_per_bridge():
    res = _res()
    s = KINDS["resistivity_rho_t"].series(res)
    assert s, "expected rho(T) series"
    bridges = {sr.group for sr in s}
    for b in bridges:
        on = [sr for sr in s if sr.group == b and sr.default_on]
        assert len(on) == 1
    assert all(sr.key.startswith("b") and ":T:" in sr.key for sr in s)
    assert all(re.fullmatch(r"b\d+:T:(na|-?\d+):-?\d+", sr.key) for sr in s)

def test_capability_gating_returns_empty():
    res = _vsm()
    assert KINDS["resistivity_mr"].series(res) == []
    assert KINDS["hall_mobility_t"].series(res) == []

def test_build_default_layout_only_backed_kinds_in_catalog_order():
    res = _vsm()
    vsm_kinds = [k for k in BUILTIN_PLOTKINDS if k.probe == "vsm"]
    lay = build_default_layout(vsm_kinds, res)
    assert [e.kind for e in lay.plots] == ["inverse_chi", "vsm_moment_t", "vsm_chi_t", "vsm_chi_t_product"]
