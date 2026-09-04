"""Multi-field M(T): recover temperature blocks the span-dominance guard discards.

A multi-field M(T) measurement is several temperature ramps at different held fields,
run back to back. The rolling-activity labeller gives them ONE label (temperature wins
throughout), and the span-dominance guard then rejects the merged block because the
field varies across it almost as much as the temperature does:

    temperature span 1180  vs  field span 790  ->  ratio 1.49  <  required 5.0

The guard is right that the merged block is ambiguous; the gap is that nothing tried
splitting it.

Recovery only ever EXAMINES blocks the guard has already rejected, so a file that
already produced blocks produces byte-identical blocks (pinned below over every
fixture). It is NOT, however, behaviour-neutral overall: a file whose find_blocks was
previously EMPTY now gets blocks, and that suppresses segment_sweeps' whole-frame
fallback. Both real VSM files take that path, which is why the CW ramp-choice rule
below had to be made explicit rather than left to "widest segment wins".
"""
from __future__ import annotations

import pathlib

from cryosweep_core.config import RunConfig
from cryosweep_core.detect.sweeps import find_blocks, segment_sweeps
from cryosweep_core.io.columns import canonicalize_columns
from cryosweep_core.io.loader import load_dat

FIX = pathlib.Path(__file__).parent / "fixtures"
MULTI = FIX / "vsm_multifield_mt_synth.dat"

# Block output for every pre-existing fixture, captured BEFORE the recovery pass existed.
# Any change here is a regression: the recovery must never alter a file that already
# produced blocks.
BASELINE = {
    "acms_real_subset.dat": [(0, 234, "temperature")],
    "acms_sc_synth.dat": [(0, 440, "temperature")],
    "act_synth.dat": [(0, 150, "temperature")],
    "bare_rho_synth.dat": [(0, 120, "temperature")],
    "hall_long_synth.dat": [],
    "hall_onesided_synth.dat": [(0, 21, "field")],
    "hall_synth.dat": [(0, 322, "field"), (327, 486, "field")],
    "hall_tdep_std_synth.dat": [
        (0, 39, "temperature"), (39, 70, "field"), (70, 93, "temperature"),
        (93, 124, "field"), (124, 147, "temperature"), (147, 178, "field"),
        (178, 201, "temperature"), (201, 232, "field"), (232, 255, "temperature"),
        (255, 286, "field"), (286, 324, "temperature"), (355, 408, "temperature")],
    "hall_tdep_synth.dat": [
        (0, 39, "temperature"), (39, 70, "field"), (70, 93, "temperature"),
        (93, 124, "field"), (124, 147, "temperature"), (147, 178, "field"),
        (178, 201, "temperature"), (201, 232, "field"), (232, 255, "temperature"),
        (255, 286, "field"), (286, 324, "temperature"), (355, 408, "temperature")],
    "hc_schottky_synth.dat": [
        (0, 25, "temperature"), (25, 56, "field"), (56, 65, "temperature"),
        (65, 96, "field"), (96, 105, "temperature"), (105, 136, "field"),
        (136, 160, "temperature")],
    "hc_multifield_synth.dat": [(0, 51, "field")],
    "hc_synth.dat": [(0, 27, "temperature")],
    "mpms_synth.dat": [(0, 150, "temperature")],
    "rho_sc_synth.dat": [(0, 134, "temperature")],
    "tto_deltat_synth.dat": [(0, 40, "temperature")],
    "tto_gap_synth.dat": [(0, 150, "temperature")],
    "tto_norho_synth.dat": [(0, 150, "temperature")],
    "tto_powerlaw_synth.dat": [(0, 150, "temperature")],
    "tto_real_subset.dat": [(0, 244, "temperature")],
    "tto_synth.dat": [(0, 135, "temperature"), (166, 180, "temperature")],
    "vsm_synth.dat": [(0, 300, "temperature")],
}


def _blocks(path):
    rt = load_dat(str(path))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    return [(b.start, b.end, b.swept_axis) for b in find_blocks(df, cmap, RunConfig.load())]


def test_existing_fixtures_segment_identically():
    """The whole risk of this change lives in this test."""
    for name, expect in BASELINE.items():
        got = _blocks(FIX / name)
        assert got == expect, f"{name}: {got} != {expect}"


def test_multifield_mt_yields_one_block_per_field():
    blocks = _blocks(MULTI)
    temps = [b for b in blocks if b[2] == "temperature"]
    assert len(temps) >= 3, blocks
    # each recovered block must hold ONE field setpoint
    rt = load_dat(str(MULTI))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    import numpy as np
    import pandas as pd

    H = pd.to_numeric(df[cmap.logical["field"]], errors="coerce").to_numpy(float)
    for s, e, _ in temps:
        seg = H[s:e]
        seg = seg[np.isfinite(seg)]
        assert np.ptp(seg) <= 50.0, f"block {s}-{e} spans fields {np.ptp(seg)}"


def test_multifield_mt_produces_temperature_segments():
    rt = load_dat(str(MULTI))
    df, cmap = canonicalize_columns(rt.df, rt.header)
    segs = [s for s in segment_sweeps(df, cmap, RunConfig.load()) if s.swept.name == "temperature"]
    assert len(segs) >= 3, [s.swept.name for s in segs]


def test_analyzer_recovers_the_multifield_sweep():
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry

    r = analyze_file(load_dat(str(MULTI)), RunConfig.load(), build_default_registry())
    assert r.status in ("ok", "low_confidence"), (r.status, r.gate, r.errors)
    tb = r.data.get("t_blocks") or []
    assert tb, "no t_blocks recovered from a 3-field M(T)"
    fields = sorted({round(b["field_oe"]) for b in tb})
    assert len(fields) >= 3, fields


def test_plateau_runs_edge_cases():
    """_plateau_runs underpins the recovery split; its degenerate inputs are pinned here.

    A ramp must NOT read as a plateau (every row steps, so all runs fall under min_len),
    and an all-non-finite axis must NOT be treated as changing -- absence of evidence for
    a setpoint change is not evidence of one.
    """
    import numpy as np

    from cryosweep_core.detect.sweeps import _plateau_runs

    assert _plateau_runs(np.array([]), 1.0, 3) == []
    assert _plateau_runs(np.full(10, np.nan), 1.0, 3) == [(0, 10)]
    assert _plateau_runs(np.zeros(10), 1.0, 3) == [(0, 10)]
    assert _plateau_runs(np.arange(10.0), 0.5, 3) == []
    assert _plateau_runs(np.r_[np.zeros(6), np.full(6, 100.0)], 1.0, 3) == [(0, 6), (6, 12)]
    # boundary: runs of EXACTLY min_len are kept (`>=`), not dropped (`>`)
    assert _plateau_runs(np.r_[np.zeros(3), np.full(3, 100.0)], 1.0, 3) == [(0, 3), (3, 6)]


def test_cw_ramp_choice_prefers_low_field_over_a_truncated_one():
    """The CW ramp rule, on synthetic data with the real failure shape.

    A short low-field ramp must lose to a full-range higher-field ramp (a CW fit over a
    truncated window is not a CW fit), while among FULL-range ramps the lowest field wins.
    """
    import numpy as np
    import pandas as pd

    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.io.loader import load_dat
    from cryosweep_core.registry import build_default_registry

    r = analyze_file(load_dat(str(MULTI)), RunConfig.load(), build_default_registry())
    fr = (r.data.get("fit") or {}).get("fit_range")
    assert fr is not None, r.data.get("fit")
    assert fr[1] - fr[0] > 250.0, fr          # full-range ramp, not a truncated one
    fields = [b["field_oe"] for b in (r.data.get("t_blocks") or [])]
    assert min(fields) == 500.0, fields        # lowest held field is present as a candidate


def _write_variant(tmp_path, *, extra_field=None, extra_short=True):
    """A 3-field M(T) plus an optional extra ramp, written as a VSM file.

    `extra_short` controls whether the extra ramp is a TRUNCATED window (excluded by the
    span gate) or a FULL-RANGE one (a real candidate that reaches the field ranking).
    """
    rows = []
    fields = [500.0, 20000.0, 40000.0] + ([extra_field] if extra_field is not None else [])
    for H in fields:
        short = extra_field is not None and H == extra_field and extra_short
        n = 20 if short else 60
        for i in range(n):
            T = 2.0 + i * (1.0 if short else 5.0)
            m = 1.0e-3 * max(H, 1.0) / 500.0 / (T + 5.0)
            rows.append(f"{T:.6f},{H:.4f},{m:.8e},1.0e-07")
    p = tmp_path / "variant.dat"
    p.write_text(
        "[Header]\nTITLE,variant\nBYAPP,VSM,1.0,1.0\n"
        "INFO,5.0,MASS:Sample Mass (mg)\nINFO,200.0,MOLWGHT:Formula Weight (g/mole)\n"
        "INFO,1,ATOMS:Atoms per Formula Unit\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Moment (emu),M. Std. Err. (emu)\n"
        + "\n".join(rows) + "\n")
    return p


def _fit_of(path):
    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry

    r = analyze_file(load_dat(str(path)), RunConfig.load(), build_default_registry())
    return r, (r.data.get("fit") or {})


def test_truncated_ramp_is_excluded_by_the_span_gate(tmp_path):
    """The 50% span threshold must actually exclude a short low-field ramp.

    Without the gate, "lowest |field|" alone picks the 100 Oe / 2-21 K ramp -- a window far
    too narrow to carry a Curie-Weiss fit. Both `0.5 ->0.0` (gate deleted) and `0.5 ->0.9`
    mutants previously survived the suite; this is the test that kills them.
    """
    p = _write_variant(tmp_path, extra_field=100.0)
    r, fit = _fit_of(p)
    assert fit, (r.status, r.data.get("reason"))
    lo, hi = fit["fit_range"]
    assert hi - lo > 200.0, f"fit ran on a truncated window {lo}-{hi}"


def test_span_gate_keeps_all_comparable_ramps(tmp_path):
    """Guards the other side: a 0.9 threshold would wrongly drop comparable full-range
    ramps and hand the fit to the widest one regardless of field."""
    p = _write_variant(tmp_path)
    r, fit = _fit_of(p)
    assert fit, r.status
    # all three ramps span 2-297 K, so the LOWEST field must win
    assert abs(fit["fit_range"][0] - 2.0) < 1e-6, fit["fit_range"]


def test_zero_field_ramp_does_not_cost_the_fit(tmp_path):
    """A nominal-zero / remanent-field ZFC ramp is the lowest |field| candidate, but every
    one of its points fails the |field| > 1 Oe physical mask. Selection must fall through
    to the next ranked ramp instead of losing the fit."""
    p = _write_variant(tmp_path, extra_field=0.3, extra_short=False)
    r, fit = _fit_of(p)
    assert fit, f"CW fit lost to an unusable zero-field ramp: {r.status} {r.data.get('reason')}"
    assert fit["n_points"] >= 3


def test_plateau_runs_tolerance_is_strict(tmp_path):
    """`> tol` vs `>= tol` at the boundary: a step EQUAL to the tolerance is not a split."""
    import numpy as np

    from cryosweep_core.detect.sweeps import _plateau_runs

    v = np.r_[np.zeros(5), np.full(5, 1.0)]          # step of exactly 1.0
    assert _plateau_runs(v, 1.0, 3) == [(0, 10)], "a step equal to tol must NOT split"
    assert _plateau_runs(v, 0.99, 3) == [(0, 5), (5, 10)]


def test_dead_end_requires_every_gate_to_be_unfixable():
    """`any` vs `all`: a result carrying even ONE user-fixable gate is not a dead end, so it
    must never be rerouted to another probe."""
    from cryosweep_core.analyzers.dispatch import _is_dead_end
    from cryosweep_core.result import Gate, Provenance, Result

    prov = Provenance(file='x', sha256='0', app_version='1.0', config={})

    fixable = Gate(need="molar_mass", reason="no MOLWGHT", remedy={"flag": "--molar-mass"})
    unfixable = Gate(need="ac_data", reason="no usable AC data")
    assert _is_dead_end(Result(status="gated", gate=[unfixable], provenance=prov))
    assert not _is_dead_end(Result(status="gated", gate=[fixable], provenance=prov))
    assert not _is_dead_end(Result(status="gated", gate=[unfixable, fixable], provenance=prov))
    assert not _is_dead_end(Result(status="gated", gate=[], provenance=prov))
    assert not _is_dead_end(Result(status="ok", gate=[unfixable], provenance=prov))


def test_span_gate_threshold_value_matters(tmp_path):
    """Kills the `0.5 -> 0.9` mutant, which the identical-span fixtures cannot.

    Ramps of EQUAL span cannot distinguish the two thresholds -- that is why the earlier
    test missed it. Here the 500 Oe ramp segments to ~135 K against the 40 kOe ramp's
    ~192 K: a ratio of 0.70, so it is a candidate at 0.5 and is dropped at 0.9. Note the
    spans are measured AFTER segmentation, which trims each ramp at the activity-window
    blur around the field change -- the raw row counts do not predict them.
    """
    rows = []
    for H, n in ((500.0, 61), (40000.0, 81)):
        for i in range(n):
            T = 2.0 + i * 3.0
            rows.append(f"{T:.6f},{H:.4f},{1.0e-3 * H / 500.0 / (T + 5.0):.8e},1.0e-07")
    p = tmp_path / "spans.dat"
    p.write_text(
        "[Header]\nTITLE,spans\nBYAPP,VSM,1.0,1.0\n"
        "INFO,5.0,MASS:Sample Mass (mg)\nINFO,200.0,MOLWGHT:Formula Weight (g/mole)\n"
        "INFO,1,ATOMS:Atoms per Formula Unit\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Moment (emu),M. Std. Err. (emu)\n"
        + "\n".join(rows) + "\n")
    r, fit = _fit_of(p)
    assert fit, r.status
    span = fit["fit_range"][1] - fit["fit_range"][0]
    assert span < 160.0, (
        f"expected the ~135 K / 500 Oe ramp, got span {span:.1f} -- the span threshold "
        f"dropped a low-field candidate it should have kept")


def test_one_stray_finite_row_does_not_steal_the_moment_column(tmp_path):
    """Kills the `most-finite -> first-any` mutant.

    DC-mode files carry `Moment (emu)` present and empty. A single stray finite value in
    it must not win the column election: with `.any()` it did, and the result collapsed
    from ok + CW fit to "insufficient physical points".
    """
    src = (FIX / "acms_dconly_synth.dat").read_text().splitlines()
    h = next(i for i, l in enumerate(src) if l.startswith("[Data]"))
    mi = src[h + 1].split(",").index("Moment (emu)")
    row = src[h + 3].split(",")
    row[mi] = "1.0e-06"
    src[h + 3] = ",".join(row)
    p = tmp_path / "stray.dat"
    p.write_text("\n".join(src) + "\n")

    import dataclasses

    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry

    rt = load_dat(str(p))
    rt = dataclasses.replace(rt, header=dataclasses.replace(
        rt.header, molar_mass=200.0, mass_mg=5.0))
    r = analyze_file(rt, RunConfig.load(), build_default_registry())
    assert r.data.get("moment_source") == "m_dc", r.data.get("moment_source")
    assert r.status == "ok", (r.status, r.data.get("reason"))


def _vsm_mt_result():
    import dataclasses

    from cryosweep_core.analyzers.dispatch import analyze_file
    from cryosweep_core.registry import build_default_registry

    from tests.core.conftest import require_real

    rt = load_dat(str(require_real("vsm_mt")))
    rt = dataclasses.replace(rt, header=dataclasses.replace(
        rt.header, molar_mass=200.0, mass_mg=5.0))
    return analyze_file(rt, RunConfig.load(), build_default_registry())


def test_one_physical_field_yields_one_label():
    """Two blocks of the SAME 40 kOe ramp must not be labelled 40000 and 40001.

    Their masked medians are 40000.8870 and 39999.5860 -- one physical field, 3 parts in
    1e5 apart, split by setpoint_key's magnitude-blind integer rounding.
    """
    r = _vsm_mt_result()
    fields = sorted({b["field_oe"] for b in r.data["t_blocks"]})
    assert fields == [500.0, 20000.0, 40000.0], fields   # not four, and not 40000.2


def test_vsm_mt_real_file_block_inventory():
    """The corrected shape on the file that exposed the defect: three fields, five blocks,
    no fragment. Measured: 150 / 150 / 150 / 139 / 142 points."""
    r = _vsm_mt_result()
    inv = sorted((round(b["field_oe"]), b["direction"], len(b["temperature"]))
                 for b in r.data["t_blocks"])
    assert {f for f, _d, _n in inv} == {500, 20000, 40000}, inv
    assert all(n >= 10 for _f, _d, n in inv), inv
    assert len(inv) == 5, inv


def test_high_field_branch_crosses_zero_susceptibility():
    """NOT a defect this slice fixes -- pinned so it is not misattributed again.

    The 1/chi panel on this file shows vertical stripes at high T. They were initially
    blamed on the block fragments; they are not. One 40 kOe branch has chi crossing zero
    18 times above 200 K (chi_min = -1.08e-05), so 1/chi swings between -1.0e6 and +1.4e6.
    That is a reciprocal-of-near-zero rendering problem, independent of segmentation.
    """
    import numpy as np

    r = _vsm_mt_result()
    crossing = []
    for b in r.data["t_blocks"]:
        T = np.asarray(b["temperature"], float)
        chi = np.asarray([v if v is not None else np.nan for v in b["chi"]], float)
        hi = T > 200.0
        if hi.sum() < 3:
            continue
        c = chi[hi][np.isfinite(chi[hi])]
        if c.size and (np.diff(np.sign(c)) != 0).any():
            crossing.append((b["field_oe"], int((np.diff(np.sign(c)) != 0).sum())))
    assert crossing, "expected a zero-crossing chi branch on this file"
    assert all(f > 30000.0 for f, _n in crossing), crossing
