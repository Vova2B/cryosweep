import pytest
import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer


# ---- helpers used by Task 5 tests ------------------------------------------

def _curves(path):
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.hall_tempdep import _interp_fixed_field_curves
    rt = load_dat(path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    return _interp_fixed_field_curves(df, cmap, RunConfig(), 1, 1.0)


# ---- Task 5: _reconstruct_points tests -------------------------------------

def test_reconstructs_exact_known_R_H(hall_tdep_synth_path):
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    pts, _ = _reconstruct_points(_curves(hall_tdep_synth_path), thickness_m=5e-5,
                                 geometry_sign=1, min_antisym=3, want_stages=False)
    fitted = [p for p in pts if p.R_H is not None]
    assert len(fitted) >= 5
    for p in fitted:
        assert abs(p.R_H - (-3.0e-8)) < 1e-10          # exact oracle (even terms cancel)
        assert p.antisym_points == 3 and p.field_count == 7
        assert p.carrier_type == "electrons"
        assert p.low_confidence is False               # 3 antisym pts >= min 3
        assert abs(p.slope_pos_ohm_per_T - p.slope_ohm_per_T) < 1e-12   # pos==avg by construction


def test_sparsity_guard_flags_two_point_fits(hall_tdep_synth_path):
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    pts, _ = _reconstruct_points(_curves(hall_tdep_synth_path), thickness_m=5e-5,
                                 geometry_sign=1, min_antisym=4, want_stages=False)   # raise bar to 4
    assert all(p.low_confidence for p in pts if p.R_H is not None)      # 3 < 4 -> all flagged


def test_even_in_B_nulls_to_zero():
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    # synthetic even-in-B array: R(B) = R0 + a*B^2 at a single T -> antisym ~ 0 -> R_H ~ 0
    curves = {0.0: (np.array([5.0]), np.array([1e-3])),
              2e4: (np.array([5.0]), np.array([1e-3 + 5e-5*2**2])),
              -2e4: (np.array([5.0]), np.array([1e-3 + 5e-5*2**2])),
              4e4: (np.array([5.0]), np.array([1e-3 + 5e-5*4**2])),
              -4e4: (np.array([5.0]), np.array([1e-3 + 5e-5*4**2]))}
    pts, _ = _reconstruct_points(curves, 5e-5, 1, 3, False)
    assert pts and abs(pts[0].R_H) < 1e-15


def test_single_pair_yields_antisym_R_H():
    # #18 flip (2026-09-02): this test previously pinned R_H is None for one ± pair;
    # a single symmetric pair is a complete antisymmetrization and now fits.
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    curves = {0.0: (np.array([5.0]), np.array([1e-3])),
              2e4: (np.array([5.0]), np.array([2e-3])),
              -2e4: (np.array([5.0]), np.array([0.5e-3]))}   # 1 antisym magnitude
    pts, _ = _reconstruct_points(curves, 5e-5, 1, 3, False)
    assert pts and pts[0].antisym_points == 1
    assert pts[0].r_h_method == "antisym"
    assert abs(pts[0].R_H - 3.75e-4 * 5e-5) < 1e-20
    assert pts[0].low_confidence is True     # min_antisym=3 passed here: knob still bites


def test_two_positives_share_one_negative_no_double_count():
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    # two positive setpoints within 200 Oe of a single negative must NOT both pair to it
    curves = {-2e4: (np.array([5.0]), np.array([0.5e-3])),
              2e4:  (np.array([5.0]), np.array([2.0e-3])),
              20100.0: (np.array([5.0]), np.array([2.1e-3])),   # within 200 Oe of -20000
              -4e4: (np.array([5.0]), np.array([0.4e-3])),
              4e4:  (np.array([5.0]), np.array([3.0e-3]))}
    pts, _ = _reconstruct_points(curves, 5e-5, 1, 3, False)
    # only two true magnitudes (~20000, 40000) pair; -20000 is consumed once
    assert pts[0].antisym_points == 2


# ---- pre-existing Task 4 tests ---------------------------------------------

def test_fixture_loads_with_transverse_and_longitudinal(hall_tdep_synth_path):
    rt = load_dat(hall_tdep_synth_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    assert "resistance_ch1" in cmap.logical and "resistance_ch2" in cmap.logical
    assert "resistivity_ch2" in cmap.logical          # genuine rho column added in fixture
    assert "temperature" in cmap.logical and "field" in cmap.logical
    # 5 paired fields x 54-pt ramp (2..55 K) + 2 extended fields (0, +20000) x 69-pt (2..70 K)
    assert len(df) == 408          # 5*54 + 2*69 (2-point tail carried on 0 & +20000 Oe)


def test_hallcfg_has_sp7_defaults():
    from cryosweep_core.config import HallCfg
    hc = HallCfg()
    assert hc.temp_interval == 1.0
    assert hc.tdep_min_antisym_points == 1   # #18: was 3 — see test_hallcfg_min_antisym_default_is_1


def test_interp_groups_by_field_and_masks_native_range(hall_tdep_synth_path):
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.hall_tempdep import _interp_fixed_field_curves
    rt = load_dat(hall_tdep_synth_path)
    df, cmap = canonicalize_columns(rt.df, rt.header)
    cfg = RunConfig()
    curves = _interp_fixed_field_curves(df, cmap, cfg, hall_channel=1, temp_interval=1.0)
    # 7 fixed fields present (0, +/-20k, +/-40k, +/-60k), resolved via the production segmenter path
    assert set(round(k) for k in curves) == {0, 20000, -20000, 40000, -40000, 60000, -60000}
    for field, (Tg, Rg) in curves.items():
        # extended fields (0, +20000 Oe) ramp to 70 K (carry the 2-point tail); paired fields
        # ramp to 55 K. Segmentation only erodes ramp ENDS, so no curve exceeds its native max.
        t_max = 70.0 if round(field) in (0, 20000) else 55.0
        assert Tg.min() >= 2.0 and Tg.max() <= t_max     # no extrapolation past native range
        assert Tg.size == Rg.size and Tg.size >= 2


def test_point_model_defaults():
    from cryosweep_core.analyzers.hall_tempdep import HallTDepPoint
    p = HallTDepPoint(temperature=5.0)
    assert p.R_H is None and p.antisym_points == 0 and p.low_confidence is False


# ---- Task 6: HallTempDepAnalyzer.analyze() tests ---------------------------

def test_analyze_synth_exact(hall_tdep_synth_path):
    rt = load_dat(hall_tdep_synth_path)
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05, "longitudinal_channel": 2})
    res = HallTempDepAnalyzer().analyze(rt, cfg)
    assert res.status == "ok"
    # antisym points carry the trusted oracle; the 2-point tail (r_h_method=="2point",
    # R_H=-2.5e-8) is validated separately in test_full_run_extends_with_two_point_tail.
    pts = [p for p in res.data["points"]
           if p["R_H"] is not None and p.get("r_h_method") == "antisym"]
    assert all(abs(p["R_H"] - (-3.0e-8)) < 1e-10 for p in pts)
    # Grounded mobility/sigma: verify against the genuine rho2(T) = 1e-6 + 1e-8*T column
    # (not raw resistance), so dimensions are correct (S/m and m^2/(V*s)).
    for p in pts:
        rho = 1e-6 + 1e-8 * p["temperature"]
        assert p["sigma"] is not None, f"sigma is None at T={p['temperature']}"
        assert p["mobility"] is not None, f"mobility is None at T={p['temperature']}"
        assert abs(p["sigma"] - 1.0 / rho) < 1e-3 * (1.0 / rho), (
            f"sigma mismatch at T={p['temperature']}: got {p['sigma']}, expected {1.0/rho}")
        assert abs(p["mobility"] - 3.0e-8 / rho) < 1e-3 * (3.0e-8 / rho), (
            f"mobility mismatch at T={p['temperature']}: got {p['mobility']}, expected {3.0e-8/rho}")
    caps = {c["name"]: c["applicable"] for c in res.data["capabilities"]}
    assert caps["hall_coefficient"] and caps["carrier_concentration"] and caps["mobility"]


def test_analyze_missing_thickness_low_conf(hall_tdep_synth_path):
    rt = load_dat(hall_tdep_synth_path)
    res = HallTempDepAnalyzer().analyze(rt, RunConfig(hall={"hall_channel": 1}))
    assert res.status == "low_confidence"
    # the warning must name the real cause (thickness), not "no fittable T point"
    assert any("thickness" in w.lower() for w in res.warnings)


def test_analyze_real_file_sparsity_edge(hall_real_path):
    if not hall_real_path.exists():
        pytest.skip("real Hall measurement file gitignored/absent")
    rt = load_dat(hall_real_path)
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.07})
    res = HallTempDepAnalyzer().analyze(rt, cfg)
    # #18 (2026-09-02): this test previously pinned the DEFECT — status low_confidence /
    # confidence 0.0 with 121 single-pair points disowned as "2point". A single ± pair is
    # an antisymmetrization; the result was always sound and now says so.
    assert res.status == "ok"
    assert res.confidence == 1.0
    fitted = [p for p in res.data["points"] if p["R_H"] is not None]
    anti = [p for p in fitted if p["r_h_method"] == "antisym"]
    two = [p for p in fitted if p["r_h_method"] == "2point"]
    # measured 2026-09-02: 121 single-pair + 16 two-pair antisym, all trusted
    assert len(anti) == 137 and all(p["antisym_points"] >= 1 for p in anti)
    assert not any(p["low_confidence"] for p in anti)
    assert max(p["temperature"] for p in anti) > 100.0   # coverage no longer stops at ~21 K
    # exactly one T has NO ± pair at all: stays 2point + low_confidence (the knob's floor)
    assert len(two) == 1 and two[0]["antisym_points"] == 0 and two[0]["low_confidence"]
    assert all(p["carrier_type"] in ("electrons", "holes") for p in fitted)  # sign depends on wiring


def test_analyze_resistivity_example_clean(res_path):
    if not res_path.exists():
        pytest.skip("Resistivity_example gitignored/absent")
    rt = load_dat(res_path)
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05})
    res = HallTempDepAnalyzer().analyze(rt, cfg)
    assert res.status == "ok"                                   # dense antisym pts -> not flagged
    fitted = [p for p in res.data["points"] if p["R_H"] is not None]
    # all fitted are true antisym fits (clean file has full ±B pairs); grid widening
    # (zero ∩ other) admits an edge temperature with 4 pairs, still a strong fit.
    assert fitted and all(p["r_h_method"] == "antisym" for p in fitted)
    assert all(p["antisym_points"] >= 4 for p in fitted)


# ---- Task 7: registry + schema + CLI wiring --------------------------------

def test_registry_and_discovery_have_hall_tdep():
    from cryosweep_core.registry import build_default_registry
    from cryosweep_core.discovery import discover
    reg = build_default_registry()
    assert "hall_tdep" in reg.analyzer_keys()
    d = discover(reg)
    # discover() probe dicts use "key" (not "probe"); reconciled from brief which used "probe"
    assert any(p == "hall_tdep" for p in [x["key"] if isinstance(x, dict) else x for x in d["probes"]])


def test_schema_hall_tdep():
    from cryosweep_core.schema import get_schema, SCHEMA_NAMES
    assert "analyze:hall_tdep" in SCHEMA_NAMES
    s = get_schema("analyze:hall_tdep")
    assert s["title"] == "HallTempDepData"


def test_cli_hall_tdep_smoke(hall_tdep_synth_path):
    import subprocess, sys, json
    out = subprocess.run([sys.executable, "-m", "cryosweep_cli", "hall-tdep", str(hall_tdep_synth_path),
                          "--hall-channel", "1", "--thickness", "0.05", "--long-channel", "2"],
                         capture_output=True, text=True)
    env = json.loads(out.stdout)
    assert env["data"]["probe"] == "hall_tdep" and env["status"] in ("ok", "low_confidence")


# ---- Task 8: catalog series backed by hall_tdep result ---------------------

def test_hall_tdep_series_backed(hall_tdep_synth_path):
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.config import RunConfig
    from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
    from cryosweep_core.plotting.catalog import BUILTIN_PLOTKINDS
    res = HallTempDepAnalyzer().analyze(
        load_dat(hall_tdep_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05, "longitudinal_channel": 2}),
    )
    byk = {k.key: k for k in BUILTIN_PLOTKINDS}
    # RH_T: tempdep series non-empty (thickness + antisym pairs present)
    assert byk["hall_tdep_RH_T"].series(res), "RH_T series empty"
    # mobility_T: longitudinal channel supplied → sigma/mobility non-empty
    assert byk["hall_tdep_mobility_T"].series(res), "mobility_T series empty"
    # interp_RT: 7 fixed fields → 7 interpolated R(T) curves
    assert byk["hall_tdep_interp_RT"].series(res), "interp_RT series empty"
    # J_T: current_density_J always None (gated) → empty
    assert byk["hall_tdep_J_T"].series(res) == [], "J_T series should be empty"


# ---- Task 2: Sub-feature B (part 1) append-only fields ----------------------

def test_tdep_point_appends_r_h_method_and_slope2point_roundtrip():
    from cryosweep_core.analyzers.hall_tempdep import HallTDepPoint
    p = HallTDepPoint(temperature=5.0, r_h_method="2point", slope_2point_ohm_per_T=-5e-4)
    d = p.model_dump(mode="json")
    assert d["r_h_method"] == "2point"
    assert d["slope_2point_ohm_per_T"] == -5e-4
    p2 = HallTDepPoint(**d)
    assert p2.r_h_method == "2point" and p2.slope_2point_ohm_per_T == -5e-4
    # defaults on an untouched point
    q = HallTDepPoint(temperature=5.0)
    assert q.r_h_method is None and q.slope_2point_ohm_per_T is None


def test_hallcfg_two_point_fallback_default_on():
    from cryosweep_core.config import HallCfg
    assert HallCfg().tdep_two_point_fallback is True
    assert HallCfg(tdep_two_point_fallback=False).tdep_two_point_fallback is False


def _twopoint_curves():
    # zero field 0 Oe over 2..6 K, and ONE other field +20000 Oe over 2..6 K (no negative
    # counterpart -> no antisym pair). R(0)=1e-3, R(2T)=1e-3 + (-6e-4)*2 => slope=-6e-4.
    import numpy as np
    T = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    R0 = np.full(5, 1e-3)
    R20 = 1e-3 + (-6e-4) * 2.0 + 0.0 * T
    return {0.0: (T, R0), 20000.0: (T, R20)}


def test_two_point_through_origin_slope_and_flag():
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    pts, _ = _reconstruct_points(_twopoint_curves(), thickness_m=5e-5, geometry_sign=1,
                                 min_antisym=3, want_stages=False, two_point_fallback=True)
    fitted = [p for p in pts if p.R_H is not None]
    assert fitted, "2-point fallback must produce R_H points"
    for p in fitted:
        assert p.r_h_method == "2point"
        assert abs(p.slope_2point_ohm_per_T - (-6e-4)) < 1e-12   # (R(2T)-R(0))/2T
        assert abs(p.R_H - (-6e-4 * 5e-5)) < 1e-16               # slope*thickness*sign
        assert p.low_confidence is True
        assert p.carrier_type == "electrons"


def test_two_point_flag_off_is_byte_identical(hall_tdep_synth_path):
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    off, _ = _reconstruct_points(_curves(hall_tdep_synth_path), 5e-5, 1, 3, False,
                                 two_point_fallback=False)
    base, _ = _reconstruct_points(_curves(hall_tdep_synth_path), 5e-5, 1, 3, False)
    assert [p.model_dump() for p in off] == [p.model_dump() for p in base]


def test_two_point_does_not_overwrite_antisym(hall_tdep_synth_path):
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    pts, _ = _reconstruct_points(_curves(hall_tdep_synth_path), 5e-5, 1, 3, False,
                                 two_point_fallback=True)
    anti = [p for p in pts if p.antisym_points >= 2]
    assert anti
    for p in anti:
        assert p.r_h_method == "antisym"
        assert p.slope_2point_ohm_per_T is None
        assert abs(p.R_H - (-3.0e-8)) < 1e-10        # trusted antisym oracle preserved


def test_grid_widening_uses_zero_intersect_other_not_zero_alone():
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    import numpy as np
    # zero field spans 2..10; other +20000 spans 2..6; a -20000 pair spans 2..4.
    # paired intersection = 2..4; zero∩other = 2..6; zero-alone would (wrongly) give 2..10.
    curves = {
        -20000.0: (np.array([2.0, 3.0, 4.0]), np.array([1e-3, 1e-3, 1e-3]) + 6e-4 * 2),
        20000.0: (np.array([2.0, 4.0, 6.0]), np.array([1e-3, 1e-3, 1e-3]) - 6e-4 * 2),
        0.0: (np.arange(2.0, 11.0), np.full(9, 1e-3)),
    }
    pts, _ = _reconstruct_points(curves, 5e-5, 1, 3, False, two_point_fallback=True)
    temps = [p.temperature for p in pts]
    assert max(temps) == 6.0     # widened to zero∩other (6), never to zero-alone (10)
    assert min(temps) == 2.0


# ---- Task 4 (Sub-feature B part 3): 2-point tail golden + split series + cap ----

def _full_tdep(hall_tdep_synth_path):
    from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer
    return HallTempDepAnalyzer().analyze(
        load_dat(hall_tdep_synth_path),
        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.05, "longitudinal_channel": 2}))


def test_full_run_extends_with_two_point_tail(hall_tdep_synth_path):
    res = _full_tdep(hall_tdep_synth_path)
    pts = res.data["points"]
    anti = [p for p in pts if p.get("r_h_method") == "antisym"]
    twop = [p for p in pts if p.get("r_h_method") == "2point"]
    assert anti, "antisym points must remain"
    assert twop, "2-point tail must extend R_H(T)"
    # antisym oracle preserved byte-for-byte within the regenerated golden
    for p in anti:
        assert abs(p["R_H"] - (-3.0e-8)) < 1e-10
    # 2-point tail lives above the paired range (>40 K) and is low-confidence
    assert all(p["temperature"] > 40.0 for p in twop)
    assert all(p["low_confidence"] for p in twop)
    # oracle for the tail: slope2 = SLOPE_TRUE + 1e-4 = -5e-4 -> R_H = -5e-4*5e-5 = -2.5e-8
    for p in twop:
        assert abs(p["slope_2point_ohm_per_T"] - (-5e-4)) < 1e-9
        assert abs(p["R_H"] - (-2.5e-8)) < 1e-12


def test_two_point_extended_capability(hall_tdep_synth_path):
    res = _full_tdep(hall_tdep_synth_path)
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["two_point_extended"]["applicable"] is True


def test_tdep_rh_series_split_by_method(hall_tdep_synth_path):
    from cryosweep_core.plotting.catalog import series_hall_tdep_rh_t, series_hall_tdep_n_t
    res = _full_tdep(hall_tdep_synth_path)
    keys = {s.key for s in series_hall_tdep_rh_t(res)}
    assert {"R_H_antisym", "R_H_2point"} <= keys
    nkeys = {s.key for s in series_hall_tdep_n_t(res)}
    assert {"n_antisym", "n_2point"} <= nkeys


# ---- KNOWN-ISSUES #18 (2026-09-02): a single ± pair IS an antisymmetrization ----

def _single_pair_curves():
    # exactly ONE ± magnitude (±2 T), no zero-field curve. R_asym = (2e-3-0.5e-3)/2
    # = 7.5e-4 Ohm at B = 2 T -> through-origin slope 3.75e-4 Ohm/T.
    T = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    return {2e4: (T, np.full(5, 2e-3)), -2e4: (T, np.full(5, 0.5e-3))}


def test_single_pair_fits_antisym_through_origin():
    from cryosweep_core.analyzers.hall_tempdep import _reconstruct_points
    pts, _ = _reconstruct_points(_single_pair_curves(), thickness_m=5e-5,
                                 geometry_sign=1, min_antisym=1, want_stages=False)
    fitted = [p for p in pts if p.R_H is not None]
    assert fitted, "single ± pair must yield an antisym R_H"
    for p in fitted:
        assert p.antisym_points == 1
        assert p.r_h_method == "antisym"
        assert abs(p.slope_ohm_per_T - 3.75e-4) < 1e-15   # R_asym/B, anchored at (0, 0)
        assert abs(p.R_H - 3.75e-4 * 5e-5) < 1e-20
        assert p.low_confidence is False                   # min_antisym=1: pair is complete
        assert p.sigma_zero_dof is True                    # zero residual DOF, honest None
        assert p.slope_sigma_ohm_per_T is None
        assert p.r2 is None                                # 1 pt through origin: no linearity claim


def test_hallcfg_min_antisym_default_is_1():
    # #18: default 3 was calibrated when < 2 pairs meant NO antisym fit at all; with the
    # single-pair fit landed, 3 would re-flag every single-pair point and keep conf at 0.0.
    from cryosweep_core.config import HallCfg
    assert HallCfg().tdep_min_antisym_points == 1


def _write_single_pair_dat(tmp_path):
    # two fixed-field temperature ramps at ±20000 Oe only -> every common T has exactly
    # one antisym pair. R(+B) = c(T)+d, R(-B) = c(T)-d with d = 1.2e-3 -> slope = d/B.
    p = tmp_path / "single_pair_tdep.dat"
    rows = []
    # 59-point ramps: the segmenter trims ~15 rows at each block boundary, and the two
    # kept windows must still overlap in T for a common grid to exist.
    for f, sgn in ((-20000.0, -1.0), (20000.0, 1.0)):
        for i in range(59):
            T = 2.0 + i
            c = 1e-3 + 1e-5 * T
            rows.append(f"{T:.4f},{f},{c + sgn * 1.2e-3:.10e},{1e-3:.10e},{1e-6:.10e}")
    p.write_text("[Header]\nBYAPP, Resistivity\nINFO, single_pair, SAMPLE\n[Data]\n"
                 "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
                 "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n"
                 + "\n".join(rows) + "\n")
    return p


def test_single_pair_file_is_ok_with_rebased_confidence(tmp_path):
    # #18 end-to-end: a file whose every T point rests on one ± pair is a sound
    # measurement and must not report confidence 0.0 / low_confidence.
    rt = load_dat(_write_single_pair_dat(tmp_path))
    cfg = RunConfig.load(hall={"hall_channel": 1, "thickness_mm": 0.1})
    res = HallTempDepAnalyzer().analyze(rt, cfg)
    assert res.status == "ok"
    assert res.confidence == 1.0
    pts = [p for p in res.data["points"] if p["R_H"] is not None]
    assert pts and all(p["r_h_method"] == "antisym" for p in pts)
    assert all(p["antisym_points"] == 1 for p in pts)
    assert all(not p["low_confidence"] for p in pts)
    # slope d/B = 1.2e-3/2 = 6e-4 Ohm/T; R_H = slope * 1e-4 m = 6e-8 m^3/C (holes)
    assert pts[0]["R_H"] == pytest.approx(6.0e-8, rel=1e-6)
    assert pts[0]["carrier_type"] == "holes"
