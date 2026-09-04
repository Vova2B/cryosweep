"""PQ-3 Task 1 — VSM analyzer growth: per-block classifier, M(H) loops, M(T) ramp
tags, and a byte-identity oracle proving existing outputs are untouched.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib
import types

import numpy as np
import pandas as pd
import pytest

from cryosweep_core.analyzers.mag import VSMAnalyzer
from cryosweep_core.config import RunConfig
from cryosweep_core.detect.vsm_blocks import classify_vsm_blocks, ramps_from_temps
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.io.loader import load_dat

FIX = pathlib.Path(__file__).parent / "fixtures"
ORACLE = FIX / "pq3_oracle"
VSM_KINDS = ["inverse_chi", "vsm_moment_t", "vsm_chi_t", "vsm_chi_t_product"]


# ---------------- helpers -------------------------------------------------

def _cmap():
    return types.SimpleNamespace(logical={"temperature": "T", "field": "H"})


def _frame(T, H):
    return pd.DataFrame({"T": np.asarray(T, float), "H": np.asarray(H, float)})


def _ramp_T(t0, t1, n):
    return np.linspace(t0, t1, n)


def _dwell_T(t, n):
    return np.full(n, float(t))


def _analyze(path, mm=None, mass=None, unit_system="CGS"):
    rt = load_dat(str(path))
    if mm is not None:
        rt = dataclasses.replace(
            rt, header=dataclasses.replace(rt.header, molar_mass=mm, mass_mg=mass))
    return VSMAnalyzer().analyze(rt, RunConfig.load(unit_system=unit_system))


# ---------------- classifier: synthetic mixed frame -----------------------

def _mixed_frame():
    # T-sweep@100 | field-loop@30K (up) | T-sweep@100 | field-loop@30K (down)
    segs = []
    segs.append((_ramp_T(2, 60, 60), np.full(60, 100.0)))         # T up @100
    segs.append((_dwell_T(30, 50), np.linspace(0, 50000, 50)))    # H up @30K
    segs.append((_ramp_T(60, 2, 60), np.full(60, 100.0)))         # T down @100
    segs.append((_dwell_T(30, 50), np.linspace(50000, 0, 50)))    # H down @30K
    T = np.concatenate([s[0] for s in segs])
    H = np.concatenate([s[1] for s in segs])
    return _frame(T, H)


def test_classifier_block_count_and_kinds():
    blks = classify_vsm_blocks(_mixed_frame(), _cmap())
    kinds = [b.kind for b in blks]
    assert kinds.count("temperature") == 2
    assert kinds.count("field") == 2
    # 100% row coverage, contiguous
    n = 220
    covered = np.zeros(n, bool)
    for b in blks:
        covered[b.start:b.end] = True
    assert covered.all()


def test_classifier_two_separate_same_T_loops():
    blks = classify_vsm_blocks(_mixed_frame(), _cmap())
    loops = [b for b in blks if b.kind == "field"]
    assert len(loops) == 2
    assert all(abs(b.setpoint - 30.0) < 1.0 for b in loops)


# ---------------- loops on VSMData --------------------------------------

def _synth_loop_dat(tmp_path, molar=200.0, mass_mg=5.0):
    """A field-sweep-only .dat (two branches at 30 K, no T-sweep)."""
    H_up = np.linspace(-50000, 50000, 60)
    H_dn = np.linspace(50000, -50000, 60)
    H = np.concatenate([H_up, H_dn])
    T = np.full(H.size, 30.0)
    moment = 1e-3 * np.tanh(H / 20000.0)  # emu
    hdr = ("[Header]\nTITLE,loopsynth\nBYAPP,VSM,1.0,1.0\n"
           f"INFO,{mass_mg},MASS:Sample Mass (mg)\n"
           f"INFO,{molar},MOLWGHT:Formula Weight (g/mole)\n"
           "INFO,1,ATOMS:Atoms per Formula Unit\n[Data]\n")
    cols = "Temperature (K),Magnetic Field (Oe),Moment (emu),M. Std. Err. (emu)\n"
    lines = [hdr, cols]
    for t, h, m in zip(T, H, moment):
        lines.append(f"{t:.6f},{h:.4f},{m:.8e},{abs(m) * 1e-4 + 1e-12:.2e}\n")
    p = tmp_path / "loopsynth.dat"
    p.write_text("".join(lines))
    return p


def test_field_only_file_ok_loops_present_fit_none(tmp_path):
    res = _analyze(_synth_loop_dat(tmp_path))
    assert res.status == "ok"
    assert res.data.get("fit") is None
    loops = res.data.get("loops")
    assert loops and len(loops) == 2
    assert all(abs(L["temperature"] - 30.0) < 0.5 for L in loops)
    # row order preserved: first loop rises, second falls
    assert loops[0]["field_oe"][0] < loops[0]["field_oe"][-1]
    assert loops[1]["field_oe"][0] > loops[1]["field_oe"][-1]
    assert loops[0]["n_points"] == len(loops[0]["field_oe"]) == len(loops[0]["moment"])
    # capability/warning notes M(H)-only
    assert any("M(H)" in w or "field" in w.lower() for w in res.warnings)


def test_neither_sweep_low_confidence_unchanged(tmp_path):
    # Neither a temperature ramp nor a field loop. Field jitters ~30 Oe (< the 50 Oe
    # classifier tolerance -> no loop) but its normalized span still edges out the tiny
    # temperature jitter, so segment_sweeps calls the swept axis "field" -> no temperature
    # segment either. This is the genuine NEITHER case: low_confidence, unchanged message.
    rng = np.random.default_rng(0)
    H = np.full(40, 100.0) + np.linspace(-15, 15, 40)   # ~30 Oe span, monotone jitter
    T = np.full(40, 50.0) + rng.normal(0, 0.01, 40)     # ~0.05 K span
    moment = np.linspace(1e-3, 1.1e-3, 40)
    hdr = ("[Header]\nTITLE,flat\nBYAPP,VSM,1.0,1.0\n"
           "INFO,5.0,MASS:Sample Mass (mg)\n"
           "INFO,200.0,MOLWGHT:Formula Weight (g/mole)\n"
           "INFO,1,ATOMS:Atoms per Formula Unit\n[Data]\n")
    cols = "Temperature (K),Magnetic Field (Oe),Moment (emu),M. Std. Err. (emu)\n"
    lines = [hdr, cols]
    for t, h, m in zip(T, H, moment):
        lines.append(f"{t:.6f},{h:.4f},{m:.8e},1e-9\n")
    p = tmp_path / "flat.dat"
    p.write_text("".join(lines))
    res = _analyze(p)
    assert res.status == "low_confidence"
    assert any("no temperature sweep" in w for w in res.warnings)


# ---------------- ramps helper (post-filter indexing) --------------------

def test_ramps_single_monotone():
    r = ramps_from_temps(np.linspace(2, 300, 100))
    assert len(r) == 1
    assert r[0]["direction"] == "warming"
    assert (r[0]["i0"], r[0]["i1"]) == (0, 99)


def test_ramps_two_with_mask_dropped_rows():
    # ZFC/FC shape in RAW rows; a filter drops some rows -> post-filter indices differ.
    T_raw = np.concatenate([np.linspace(2, 100, 40), np.linspace(100, 2, 40)])  # up then down
    field_raw = np.full(80, 1000.0)
    field_raw[5] = 0.0        # a row the |field|>1 filter would drop
    field_raw[50] = 0.0
    keep = np.abs(field_raw) > 1.0
    T_post = T_raw[keep]      # compacted (post-filter) array
    r = ramps_from_temps(T_post)
    assert len(r) == 2
    assert r[0]["direction"] == "warming"
    assert r[1]["direction"] == "cooling"
    # indices are into the COMPACTED array, not raw rows
    assert r[0]["i0"] == 0
    assert r[1]["i1"] == T_post.size - 1
    assert r[1]["i0"] == r[0]["i1"] + 1
    # the boundary sits at the compacted peak (T_post has a duplicated 100 K peak),
    # NOT the raw turning row 40 — proving post-filter (compacted) indexing.
    assert r[0]["i1"] in (int(np.argmax(T_post)), int(np.argmax(T_post)) + 1)
    assert r[0]["i1"] < 40


# ---------------- VSMData ramps field (both-present file) -----------------

def test_ramps_field_present_on_mt_file():
    res = _analyze(FIX / "vsm_synth.dat")
    ramps = res.data.get("ramps")
    assert ramps and len(ramps) == 1
    assert ramps[0]["direction"] == "warming"
    T = res.data["temperature"]
    assert ramps[0]["i0"] == 0 and ramps[0]["i1"] == len(T) - 1


# ---------------- real fixtures ------------------------------------------

def test_vsm_n_classifier_coverage_and_loops(vsm_real_path):
    rt = load_dat(str(vsm_real_path))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    blks = classify_vsm_blocks(df, cmap)
    n = len(df)
    covered = np.zeros(n, bool)
    for b in blks:
        covered[b.start:b.end] = True
    assert covered.mean() > 0.99
    tblocks = [b for b in blks if b.kind == "temperature"]
    assert len(tblocks) >= 4                       # M(T) at 100/5000/40000/100000 Oe
    fblocks = [b for b in blks if b.kind == "field"]
    from cryosweep_core.grouping import setpoint_key
    at30 = [b for b in fblocks if setpoint_key(b.setpoint) == 30.0]
    assert len(at30) == 2                           # exactly 2 loops at 30.0 K


def test_vsm_n_loops_on_result(vsm_real_path):
    from cryosweep_core.grouping import setpoint_key
    res = _analyze(vsm_real_path, mm=200.0, mass=1.1)   # VSM_N header has no MOLWGHT/MASS
    loops = res.data.get("loops") or []
    at30 = [L for L in loops if setpoint_key(L["temperature"]) == 30.0]
    assert len(at30) == 2
    for L in loops:
        assert L["n_points"] == len(L["field_oe"]) == len(L["moment"]) > 0


def test_mpms_zfc_fc_two_ramps_in_field_group(mpms_real_path):
    rt = load_dat(str(mpms_real_path))
    rt = dataclasses.replace(
        rt, header=dataclasses.replace(rt.header, molar_mass=500.0, mass_mg=10.0))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    blks = classify_vsm_blocks(df, cmap)
    tblocks = [b for b in blks if b.kind == "temperature"]
    assert len(tblocks) >= 2
    # the 500 Oe block carries both the ZFC (warming) and FC (cooling) ramps
    b500 = min(tblocks, key=lambda b: abs(b.setpoint - 500.0))
    T = pd.to_numeric(df[cmap.logical["temperature"]], errors="coerce").to_numpy(float)
    ramps = ramps_from_temps(T[b500.start:b500.end])
    dirs = {r["direction"] for r in ramps}
    assert len(ramps) == 2 and dirs == {"warming", "cooling"}


# ---------------- byte-identity oracle -----------------------------------
# Goldens captured from base commit 22eb5fe (pristine tree) via a one-shot script;
# stored under fixtures/pq3_oracle/. The analyzer growth is additive, so:
#   - result JSON MINUS the new keys (loops, ramps) must be byte-identical,
#   - all export CSVs / sidecar must be byte-identical,
#   - vsm_synth report + the UNCHANGED existing-kind PNGs must be byte-identical.
# MPMS PNGs are excluded. NOTE (PQ-3 Task 4 reality): the analyzer exports the single
# WIDEST temperature segment, which is monotone by construction, so MPMS's ZFC/FC file
# yields exactly ONE ramp over the flat arrays -> its existing-kind renders stay single-
# series and byte-identical too. The warming/cooling ramp SPLIT (>1 ramp) is therefore
# exercised on synthetic 2-ramp results in test_render_vsm_pq3.py; growing the analyzer to
# export both ZFC+FC ramps is recognized-deferred.
# PQ-3 Task 3 amendment (sanctioned deltas, folded into the goldens):
#   - `fit_modified` is a new additive key -> added to the strip list;
#   - the CGS Curie-constant unit string was reconciled to the physically-correct
#     "emu*K/(mol*Oe)" (fit.units.C) — this legitimately changes result.json + fit_params.csv
#     + report.md, which were regenerated; it is NOT a rendering/analysis-shape change;
#   - the `vsm_chi_t` (now twin χ/χ⁻¹) and `inverse_chi` (now +modified-CW line + θ/C box)
#     renderers changed by design -> those two PNG goldens were regenerated; the two untouched
#     kinds (vsm_moment_t, vsm_chi_t_product) stay strictly byte-identical.
# Legend-fix amendment (multi-axis "best" placement): the twin `vsm_chi_t` merged legend now
#   dodges BOTH axes' data (not just the host axis'), so its golden was regenerated once more.
#   The three single-axis VSM kinds stay strictly byte-identical (the fix is gated on >1 axes).
_NEW_KEYS = ("loops", "ramps", "fit_modified", "t_blocks",
             # 2026-08-10 uncertainty-honesty additive fields (CW ladder, spec §1.2):
             "cw_ladder", "theta_spread_k", "mu_eff_spread")
_UNCHANGED_PNG_KINDS = ("vsm_moment_t", "vsm_chi_t_product")


def _strip_new(data: dict) -> dict:
    return {k: v for k, v in data.items() if k not in _NEW_KEYS}


def _export_csvs(res, tmp):
    # The `meta` sidecar embeds the invocation file path + sha (not analysis data), so it
    # is excluded from the byte oracle; the substantive CSVs (points/fit_params/derived)
    # carry no path and are compared byte-for-byte against the base-code goldens.
    from cryosweep_core.io.export import export_result
    outs = export_result(res, str(tmp / "e"))
    return {kind: pathlib.Path(p).read_bytes() for kind, p in outs.items() if kind != "meta"}


# ---------------- oracle comparison: exact on structure, tolerant on floats ----------
#
# The numeric oracles were generated on one machine's BLAS. LAPACK/BLAS results differ
# between platforms in the last one or two ULPs, so comparing the SERIALIZED TEXT pins the
# oracle to the floating-point library rather than to this code. Measured on the first Linux
# CI run (2026-09-04) from a byte-identical input file:
#     -1.0252441501357285e-07  (macOS/Accelerate)   vs
#     -1.025244150135774e-07   (Linux/OpenBLAS)      -> 4.4e-13 relative
# Every contributor not on the generating platform would have seen a red suite.
#
# So: structure, keys, strings, booleans and list lengths are still compared EXACTLY -- only
# numbers get a tolerance, and a tight one. 1e-9 sits ~4 orders above the observed platform
# spread and far below anything this project would want to miss: a real regression moves a
# fitted parameter in its leading digits, not its fifteenth. The report markdown carries only
# rounded values and stays an exact comparison; the PNG bytes remain pinned and version-skipped.
_ORACLE_RTOL = 1e-9


def _num_close(a, b) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b                       # bools are ints in Python; never tolerate
    return bool(np.isclose(a, b, rtol=_ORACLE_RTOL, atol=0.0, equal_nan=True))


def _assert_like_oracle(got, want, path="$"):
    """Recursive structural equality with float tolerance. Raises AssertionError naming the
    JSON path of the first mismatch, so a failure is as diagnosable as a text diff."""
    assert type(got) is type(want) or isinstance(got, type(want)) or isinstance(want, type(got)), \
        f"{path}: type {type(got).__name__} != {type(want).__name__}"
    if isinstance(want, dict):
        assert set(got) == set(want), f"{path}: keys differ: {set(got) ^ set(want)}"
        for k in want:
            _assert_like_oracle(got[k], want[k], f"{path}.{k}")
    elif isinstance(want, (list, tuple)):
        assert len(got) == len(want), f"{path}: length {len(got)} != {len(want)}"
        for n, (g, w) in enumerate(zip(got, want)):
            _assert_like_oracle(g, w, f"{path}[{n}]")
    elif isinstance(want, (int, float)) and not isinstance(want, bool):
        assert _num_close(got, want), f"{path}: {got!r} != {want!r} (rtol {_ORACLE_RTOL})"
    else:
        assert got == want, f"{path}: {got!r} != {want!r}"


def _assert_csv_like_oracle(blob: bytes, golden: pathlib.Path, label: str):
    """Same contract for the CSV goldens, which carry full-precision floats and are exposed
    to exactly the same platform drift. Cell-by-cell: numeric cells by tolerance, every other
    cell (headers, units, flags, blanks) byte-for-byte."""
    got_rows = blob.decode("utf-8").splitlines()
    want_rows = golden.read_bytes().decode("utf-8").splitlines()
    assert len(got_rows) == len(want_rows), f"{label}: {len(got_rows)} rows != {len(want_rows)}"
    for r, (gl, wl) in enumerate(zip(got_rows, want_rows)):
        gc, wc = gl.split(","), wl.split(",")
        assert len(gc) == len(wc), f"{label} row {r}: {len(gc)} cells != {len(wc)}"
        for c, (g, w) in enumerate(zip(gc, wc)):
            if g == w:
                continue
            try:
                gv, wv = float(g), float(w)
            except ValueError:
                raise AssertionError(f"{label} row {r} col {c}: {g!r} != {w!r}") from None
            assert _num_close(gv, wv), \
                f"{label} row {r} col {c}: {g} != {w} (rtol {_ORACLE_RTOL})"


def test_oracle_comparison_tolerates_platform_ulps_but_not_regressions():
    """The tolerance must be wide enough for the measured BLAS spread and narrow enough to
    still catch a real change. Without this, 1e-9 is an unexamined magic number."""
    base = {"a": [-1.0252441501357285e-07], "n": 300, "model": "curie_weiss", "ok": True}
    ulp = {"a": [-1.025244150135774e-07], "n": 300, "model": "curie_weiss", "ok": True}
    _assert_like_oracle(ulp, base)                       # the exact Linux-vs-macOS pair: passes
    for bad, why in [
        ({"a": [-1.0252441e-07], "n": 300, "model": "curie_weiss", "ok": True}, "7th digit"),
        ({"a": [-1.0252441501357285e-07], "n": 301, "model": "curie_weiss", "ok": True}, "count"),
        ({"a": [-1.0252441501357285e-07], "n": 300, "model": "cw", "ok": True}, "string"),
        ({"a": [-1.0252441501357285e-07], "n": 300, "model": "curie_weiss", "ok": False}, "bool"),
    ]:
        with pytest.raises(AssertionError):
            _assert_like_oracle(bad, base)


def test_oracle_vsm_synth_json_csv_report_png(tmp_path):
    res = _analyze(FIX / "vsm_synth.dat")
    # JSON minus new keys
    got = _strip_new(res.data)
    want = json.loads((ORACLE / "vsm_synth.result.json").read_text())
    _assert_like_oracle(json.loads(json.dumps(got, sort_keys=True)), want)
    # new keys ARE present (additive)
    assert set(res.data) >= set(_NEW_KEYS)
    # CSVs
    for kind, blob in _export_csvs(res, tmp_path).items():
        _assert_csv_like_oracle(blob, ORACLE / f"vsm_synth.{kind}.golden", kind)
    # report
    from cryosweep_core.reports import build_report
    assert build_report(res)["markdown"] == (ORACLE / "vsm_synth.report.md").read_text()
    # PNGs (default style) for all 4 existing kinds.
    # The PNG goldens are byte-pinned to the matplotlib THAT RENDERED THEM: 3.11.1 already
    # changes the bytes of inverse_chi (measured 2026-09-01; numpy 2.4.6->2.5.2,
    # pandas 3.0.3->3.0.5 and scipy 1.17.1->1.18.1 change nothing -- the numeric goldens
    # above hold). A fresh clone installs whatever matplotlib pip resolves, so on any other
    # version this block SKIPS rather than fails: the numeric oracle (JSON/CSV/report,
    # asserted above) is the contract; the PNG bytes are a same-version regression pin.
    import matplotlib
    _GOLDEN_MPL = "3.11.0"
    if matplotlib.__version__ != _GOLDEN_MPL:
        pytest.skip(f"PNG goldens are pinned to matplotlib {_GOLDEN_MPL} "
                    f"(running {matplotlib.__version__}); numeric oracle already asserted")
    import matplotlib.pyplot as plt
    from cryosweep_core.plotting.render import render_kind
    from cryosweep_core.plotting.export import save_figure
    from cryosweep_core.plotting.spec import GlobalStyle, PlotSpec
    for k in VSM_KINDS:
        fig = render_kind(res, k, PlotSpec(), GlobalStyle())
        p = save_figure(fig, tmp_path / f"{k}.png", GlobalStyle())
        plt.close(fig)
        assert p.read_bytes() == (ORACLE / f"vsm_synth.{k}.png").read_bytes(), k


def test_oracle_mpms_json_csv(tmp_path, mpms_real_path):
    res = _analyze(mpms_real_path, 500.0, 10.0)
    got = json.loads(json.dumps(_strip_new(res.data), sort_keys=True))
    _assert_like_oracle(got, json.loads((ORACLE / "mpms.result.json").read_text()))
    for kind, blob in _export_csvs(res, tmp_path).items():
        _assert_csv_like_oracle(blob, ORACLE / f"mpms.{kind}.golden", kind)


# ---------------- determinism --------------------------------------------

def test_determinism_two_runs_identical():
    a = _analyze(FIX / "vsm_synth.dat")
    b = _analyze(FIX / "vsm_synth.dat")
    assert json.dumps(a.data, sort_keys=True) == json.dumps(b.data, sort_keys=True)


# ---------------- t_blocks (per-temperature-block M(T) arrays) ------------
# Additive field: one entry per (temperature-sweep block x monotone ramp), carrying the
# SAME per-point math/mask as the flat exported arrays. Lets the M(T)-family renderers
# split warming/cooling on real ZFC/FC files (the flat arrays only hold the widest segment).

def _tblock_lengths_consistent(tb):
    return (len(tb["temperature"]) == len(tb["moment"])
            == len(tb["chi"]) == len(tb["inv_chi"]) > 0)


def test_t_blocks_key_appended_after_fit_modified():
    # pydantic declaration order = JSON key order; t_blocks must be the LAST vsm key
    res = _analyze(FIX / "vsm_synth.dat")
    keys = list(res.data.keys())
    assert "t_blocks" in keys
    assert keys.index("t_blocks") > keys.index("fit_modified")


def test_t_blocks_singleton_on_vsm_synth():
    # vsm_synth is a single monotone M(T) ramp -> exactly one t_block, warming
    res = _analyze(FIX / "vsm_synth.dat")
    tb = res.data["t_blocks"]
    assert len(tb) == 1
    assert tb[0]["direction"] == "warming"
    assert _tblock_lengths_consistent(tb[0])


def test_t_blocks_mask_drops_field_zero():
    # Direct unit test of the per-block mask (matches the flat CW path: |field|>1 AND finite).
    # A single warming block with two field~0 rows + one NaN-inv_chi row -> all three dropped.
    from cryosweep_core.analyzers.mag import _compute_t_blocks
    from cryosweep_core.detect.vsm_blocks import VSMBlock
    n = 30
    temp = np.linspace(2.0, 100.0, n)
    field = np.full(n, 1000.0)
    field[7] = 0.0
    field[19] = 0.0
    moment = 1.0 / (temp + 5.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi = moment / field
        inv_chi = 1.0 / chi                                   # field~0 -> inf chi -> inv_chi 0 (finite)
    inv_chi[25] = np.nan                                      # a genuinely non-finite inv_chi row
    blocks = [VSMBlock(start=0, end=n, kind="temperature", setpoint=1000.0)]
    tb = _compute_t_blocks(blocks, temp, field, moment, chi, inv_chi)
    tvals = [t for b in tb for t in b.temperature]
    assert len(tvals) == n - 3                                # rows 7, 19 (|field|<1) + 25 (NaN)
    for bad in (temp[7], temp[19], temp[25]):
        assert float(bad) not in tvals
    for b in tb:
        assert (len(b.temperature) == len(b.moment) == len(b.chi) == len(b.inv_chi))
        assert setpoint_key(b.field_oe) == 1000.0


from cryosweep_core.grouping import setpoint_key  # noqa: E402  (used by mask + real-fixture tests)


def test_t_blocks_mpms_zfc_fc_same_field_both_directions(mpms_real_path):
    from cryosweep_core.grouping import setpoint_key
    res = _analyze(mpms_real_path, mm=683.22, mass=12.0)
    tb = res.data["t_blocks"]
    # the 500 Oe field group carries BOTH the ZFC (warming) and FC (cooling) ramps as
    # separate t_blocks at the SAME rounded field setpoint
    at500 = [b for b in tb if setpoint_key(b["field_oe"]) == 500.0]
    assert len(at500) >= 2
    assert {b["direction"] for b in at500} == {"warming", "cooling"}
    assert all(_tblock_lengths_consistent(b) for b in tb)
    # field groups must not bleed together: the 40000 Oe ramp's moment (∝ field) is far larger
    # than the 500 Oe ramps'; if a boundary row leaked in, max(500-Oe moment) would spike.
    at40k = [b for b in tb if setpoint_key(b["field_oe"]) == 40000.0]
    if at40k:
        mx500 = max(m for b in at500 for m in b["moment"])
        mn40k = min(m for b in at40k for m in b["moment"])
        assert mx500 < mn40k


def test_t_blocks_vsm_n_multiple_field_setpoints(vsm_real_path):
    from cryosweep_core.grouping import setpoint_key
    res = _analyze(vsm_real_path, mm=200.0, mass=1.1)
    tb = res.data["t_blocks"]
    fields = {setpoint_key(b["field_oe"]) for b in tb}
    assert len(fields) >= 2                                   # 100/5000/40000/100000 Oe ramps
    assert all(_tblock_lengths_consistent(b) for b in tb)
