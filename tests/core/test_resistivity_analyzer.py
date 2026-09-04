import numpy as np
import pytest
from cryosweep_core.analyzers.resistivity import (
    ResistivityData, BridgeResult, _physical_mask, _rrr, _classify, _mr_percent,
)

def test_physical_mask_drops_nonpositive_and_nonfinite():
    rho = np.array([1.0, -5.0, np.nan, 2.0, np.inf, 0.0])
    m = _physical_mask(rho)
    assert m.tolist() == [True, False, False, True, False, False]

def test_rrr_excludes_a_garbage_endpoint():
    T = np.linspace(2, 300, 600)
    rho = 1.0e-7 * T                          # monotonic in T
    rrr_clean, _, _ = _rrr(T, rho.copy(), k=5)
    assert rrr_clean > 1.0 and np.isfinite(rrr_clean)
    # one garbage low-T row must be filtered, not corrupt the result
    dirty = rho.copy(); dirty[0] = -1.0e7
    rrr_dirty, t_hi, t_lo = _rrr(T, dirty, k=5)
    # independent expectation: drop the non-physical row, apply the SAME k=5 median method
    phys = dirty > 0
    Tp, Rp = T[phys], dirty[phys]
    lo = float(np.median(Rp[np.argsort(Tp)[:5]]))
    hi = float(np.median(Rp[np.argsort(Tp)[-5:]]))
    assert rrr_dirty == pytest.approx(hi / lo, rel=1e-9)
    assert rrr_dirty > 1.0 and np.isfinite(rrr_dirty)   # never negative/garbage
    assert t_hi > t_lo

def test_classify_metallic_when_rho_rises_with_T():
    T = np.linspace(2, 300, 50)
    assert _classify(T, 1e-6 + 1e-8 * T) == "metallic"
    assert _classify(T, 1e-3 * np.exp(50.0 / T)) == "insulating"

def test_mr_percent_uses_interpolated_zero_field():
    H = np.linspace(-1000, 1000, 41)
    rho = 1.0e-6 * (1.0 + (H / 1000.0) ** 2)   # rho0=1e-6, rho(1000)=2e-6 -> MR=100%
    rho0, mr, hmax, low = _mr_percent(H, rho)
    assert rho0 == pytest.approx(1.0e-6, rel=1e-6)
    assert mr == pytest.approx(100.0, rel=1e-3)
    assert hmax == pytest.approx(1000.0)
    assert low is False


def _analyze_real(res_path, include_hall=False, **geom):
    """include_hall=True disables Hall-channel routing (pre-routing behavior) for tests
    whose oracle covers BOTH channels; the QD example's Ch2 is Hall-wired (odd-in-B 0.53)
    and is excluded from resistivity at defaults since the 2026-07-02 routing spec."""
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.resistivity import ResistivityAnalyzer
    rt = load_dat(res_path)
    kw = {}
    if geom:
        kw["geometry"] = geom
    if include_hall:
        kw["resistivity"] = {"exclude_hall_channel": False}
    cfg = RunConfig.load(**kw) if kw else RunConfig()
    return ResistivityAnalyzer().analyze(rt, cfg)

def test_real_file_only_populated_bridges(res_path):
    res = _analyze_real(res_path)
    assert res.status in ("ok", "low_confidence")
    chans = sorted(b["channel"] for b in res.data["bridges"])
    assert chans == [1]                                 # 3,4 empty; Ch2 routed out as Hall-wired
    assert res.data["excluded_hall_channel"] == 2
    assert res.data["excluded_hall_source"] == "detected"
    assert res.data["rho_source"] == "instrument_column"

def test_real_file_rrr_matches_oracle(res_path):
    res = _analyze_real(res_path, include_hall=True)    # oracle covers both channels
    by_ch = {b["channel"]: b for b in res.data["bridges"]}
    assert by_ch[1]["rrr"] == pytest.approx(18.52, rel=0.02)
    assert by_ch[2]["rrr"] == pytest.approx(21.80, rel=0.02)
    assert by_ch[1]["classification"] == "metallic"
    assert by_ch[2]["classification"] == "metallic"

def test_real_file_mr_on_clean_300k_loop(res_path):
    res = _analyze_real(res_path, include_hall=True)    # DQ-B MR oracle covers both channels
    by_ch = {b["channel"]: b for b in res.data["bridges"]}
    # DQ-B: field ramps are grouped into one display loop per setpoint (direction=0).
    # MR% is computed on the group's widest ramp -> byte-identical to the old per-ramp value.
    def find(b):
        for c in b["rho_h_curves"]:
            if c["held_temp_k"] is not None and round(c["held_temp_k"]) == 300 and c["direction"] == 0:
                return c
        return None
    c1 = find(by_ch[1]); c2 = find(by_ch[2])
    assert c1 is not None and c2 is not None
    assert c1["mr_percent_at_max_field"] == pytest.approx(3.51, abs=0.3)
    assert c2["mr_percent_at_max_field"] == pytest.approx(29.46, abs=0.5)

def test_real_file_capabilities_reported(res_path):
    res = _analyze_real(res_path)
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["curve_separation"]["applicable"] is True
    assert caps["RRR"]["applicable"] is True
    assert caps["magnetoresistance"]["applicable"] is True
    assert caps["power_law_fit"]["applicable"] is False      # real file: low-T window too high, rho0 unresolved
    assert "unresolved" in caps["power_law_fit"]["reason"]
    # recognized-but-unimplemented analyses are listed with applicable=False + a reason
    assert "activated_transport" in caps
    assert caps["activated_transport"]["applicable"] is False
    assert caps["activated_transport"]["reason"]            # non-empty explanation
    assert "superconducting_transition" in caps

def test_geometry_recompute_changes_source_not_rrr(res_path):
    # RRR is a ratio -> geometry-independent; source flips to "geometry"
    res = _analyze_real(res_path, width_mm=1.0, thickness_mm=0.1, length_mm=2.0)
    assert res.data["rho_source"] == "geometry"
    by_ch = {b["channel"]: b for b in res.data["bridges"]}
    assert by_ch[1]["rho_source"] == "geometry"
    assert by_ch[1]["rrr"] == pytest.approx(18.52, rel=0.05)   # ratio unchanged by geometry


def test_dispatch_routes_resistivity(res_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.analyzers.dispatch import analyze_file
    rt = load_dat(res_path)
    res = analyze_file(rt, RunConfig(), build_default_registry())
    assert res.status in ("ok", "low_confidence")
    assert res.data["probe"] == "resistivity"
    assert sorted(b["channel"] for b in res.data["bridges"]) == [1]   # Ch2 routed out (Hall-wired)

def test_discovery_lists_resistivity_analyzer():
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.discovery import discover
    d = discover(build_default_registry())
    res = [p for p in d["probes"] if p["key"] == "resistivity"][0]
    assert res["has_analyzer"] is True

def test_real_file_power_law_flagged_and_residual_withheld(res_path):
    res = _analyze_real(res_path)
    for b in res.data["bridges"]:
        pl = b["power_law"]
        assert pl is not None
        assert "rho0_unresolved" in pl["quality_flags"]      # honest: fit didn't resolve rho0
        assert b["residual_rho"] is None                     # unresolved residual not reported

def test_applicable_capabilities_have_populated_results(res_path):
    res = _analyze_real(res_path)
    caps = {c["name"]: c for c in res.data["capabilities"]}
    bridges = res.data["bridges"]
    if caps["RRR"]["applicable"]:
        assert any(b["rrr"] is not None for b in bridges)
    if caps["magnetoresistance"]["applicable"]:
        assert any(c["mr_percent_at_max_field"] is not None
                   for b in bridges for c in b["rho_h_curves"])

def test_rrr_matches_independent_recomputation(res_path):
    # independent of the analyzer's helpers: load raw, pick widest zero-field T-ramp,
    # filter rho>0, median-5 nearest-extreme endpoints, ratio. Must match analyzer rrr.
    import numpy as np, pandas as pd
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.io.columns import canonicalize_columns
    from cryosweep_core.detect.sweeps import segment_sweeps
    from cryosweep_core.config import RunConfig
    rt = load_dat(res_path); df, cmap = canonicalize_columns(rt.df, rt.header); cfg = RunConfig()
    T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    segs = [s for s in segment_sweeps(df, cmap, cfg) if s.swept.name == "temperature"]
    zf = [s for s in segs if abs(s.setpoint.get("field") or 0.0) < 50.0]
    ramp = max(zf, key=lambda s: float(np.ptp(T[s.idx])))
    idx = ramp.idx
    res = _analyze_real(res_path, include_hall=True)    # recomputation covers both channels
    by_ch = {b["channel"]: b for b in res.data["bridges"]}
    for ch in (1, 2):
        rho = pd.to_numeric(df[cmap.logical[f"resistivity_ch{ch}"]], errors="coerce").to_numpy(float) * 100.0
        m = np.isfinite(T[idx]) & np.isfinite(rho[idx]) & (rho[idx] > 0)
        Tk, Rk = T[idx][m], rho[idx][m]
        order = np.argsort(Tk)
        lo = float(np.median(Rk[order[:5]])); hi = float(np.median(Rk[order[-5:]]))
        assert by_ch[ch]["rrr"] == pytest.approx(hi / lo, rel=1e-3)

def test_rho_t2_linear_present_on_metallic_ramp():
    import pathlib
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    FIX = pathlib.Path(__file__).parent / "fixtures"
    res = analyze_file(load_dat(str(FIX / "act_synth.dat")),
                       RunConfig.load(probe_override="resistivity"), build_default_registry())
    bridges = res.data["bridges"]
    assert bridges and all(b["classification"] == "metallic" for b in bridges)
    for b in bridges:
        f = b["rho_t2_linear"]
        assert f is not None
        assert set(f["params"]) == {"rho0", "beta"}
        assert f["r2"] > 0.9
        assert f["fit_range"][1] <= 30.0 + 1e-9     # window is <=30 K

def test_rho_t2_linear_none_without_zero_field_ramp():
    import pathlib
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry
    FIX = pathlib.Path(__file__).parent / "fixtures"
    # hall_synth as resistivity has only field sweeps (no zero-field T ramp) -> no fits
    res = analyze_file(load_dat(str(FIX / "hall_synth.dat")),
                       RunConfig.load(probe_override="resistivity"), build_default_registry())
    for b in res.data["bridges"]:
        assert b["rho_t2_linear"] is None


# ---------------- §11 geometry-unset warning (2026-08-10, owner-decided) --------------

def test_geometry_unset_warning_fires_on_instrument_column(res_path):
    # QD example: header Cross Section = Length = 1 (user never set geometry in the PPMS
    # software) and no sample geometry supplied -> rho falls back to the instrument column,
    # whose absolute scale is arbitrary. Must WARN, keep reporting (spec §11).
    res = _analyze_real(res_path)
    assert res.data["rho_source"] == "instrument_column"
    w = [x for x in res.warnings if "scale is arbitrary" in x]
    assert len(w) == 1 and w[0].startswith("ch1:")
    # the warning states what is and is not affected, and names the app's own remedy
    assert "RRR" in w[0] and "MR%" in w[0]               # ratios unaffected
    assert "ρ₀" in w[0]                                   # scale-dependent: rho0 named
    assert "width / thickness / length" in w[0]           # remedy = the panel inputs
    # purely additive: nothing already reported changed in value
    assert res.data["bridges"][0]["rrr"] == pytest.approx(18.52, rel=0.02)


def test_geometry_supplied_silences_the_warning(res_path):
    res = _analyze_real(res_path, width_mm=1.0, thickness_mm=0.5, length_mm=2.0)
    assert res.data["rho_source"] == "geometry"
    assert not any("scale is arbitrary" in x for x in res.warnings)


def test_header_geometry_set_is_not_unset():
    # A header where the user DID set geometry (values != 1) must not trip the detector,
    # while unity/absent/garbage values must (unset data, not malformed data).
    import types
    from cryosweep_core.analyzers.resistivity import _header_geometry_unset
    set_hdr = types.SimpleNamespace(info={"Sample1 Cross Section": "0.24",
                                          "Sample1 Length": "1.6"})
    assert _header_geometry_unset(set_hdr, 1) is False
    for info in ({"Sample1 Cross Section": "1", "Sample1 Length": "1"},
                 {},
                 {"Sample1 Cross Section": "N/A", "Sample1 Length": ""}):
        assert _header_geometry_unset(types.SimpleNamespace(info=info), 1) is True
    # one set value is enough to count as "user touched geometry" -> not unset
    half = types.SimpleNamespace(info={"Sample1 Cross Section": "0.24",
                                       "Sample1 Length": "1"})
    assert _header_geometry_unset(half, 1) is False
