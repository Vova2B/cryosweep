import numpy as np
import pandas as pd
from cryosweep_core.config import RunConfig
from cryosweep_core.io.columns import ColumnMap
from cryosweep_core.detect.sweeps import find_blocks, split_by_direction, segment_sweeps, select_swept_axis


def test_select_swept_axis_no_candidates():
    # Bug 6: a frame with zero whitelisted axes must not IndexError
    cmap = ColumnMap(logical={"resistance": "R"}, unit={"resistance": "ohm"})
    df = pd.DataFrame({"R": [1.0, 2.0, 3.0]})
    swept, conf, warn = select_swept_axis(df, cmap, RunConfig.load())
    assert swept is None
    assert conf == 0.0
    assert warn == ["no candidate axes"]


def test_segment_sweeps_no_candidates_returns_empty():
    # Bug 6: no whitelisted axes -> [] (don't fabricate an empty segment)
    cmap = ColumnMap(logical={"resistance": "R"}, unit={"resistance": "ohm"})
    df = pd.DataFrame({"R": [1.0, 2.0, 3.0]})
    assert segment_sweeps(df, cmap, RunConfig.load()) == []

CMAP = ColumnMap(logical={"temperature": "T", "field": "H"}, unit={"temperature": "K", "field": "Oe"})
CFG = RunConfig.load()
RNG = np.random.default_rng(0)

def _df(T, H):
    return pd.DataFrame({"T": np.asarray(T, float), "H": np.asarray(H, float)})

def test_ramp_not_shattered_picks_temperature():
    # single temperature ramp, field held ~0.5 Oe -> ONE temperature block
    T = np.linspace(2.2, 252.0, 300)
    H = 0.5 + 0.03 * RNG.standard_normal(300)
    blocks = find_blocks(_df(T, H), CMAP, CFG)
    assert len(blocks) == 1
    assert blocks[0].swept_axis == "temperature"
    segs = segment_sweeps(_df(T, H), CMAP, CFG)
    assert len(segs) == 1 and segs[0].direction == 1

def test_field_loop_splits_up_down():
    # isotherm at fixed T=300, field 0->+9T->-9T->0 (Oe)
    up = np.linspace(0, 90000, 100); down = np.linspace(90000, -90000, 200); back = np.linspace(-90000, 0, 100)
    H = np.concatenate([up, down, back]); T = 300 + 0.01 * RNG.standard_normal(H.size)
    blocks = find_blocks(_df(T, H), CMAP, CFG)
    assert all(b.swept_axis == "field" for b in blocks)   # huge field span dominates
    runs = split_by_direction(H, CFG)
    assert len(runs) >= 2
    segs = segment_sweeps(_df(T, H), CMAP, CFG)
    assert {s.branch for s in segs} >= {"up", "down"}

def test_transition_block_rejected_and_split_ramps_merged():
    # Bug 3a: a field-stepping "transition" while temperature is resetting must NOT become a
    #          swept=field segment with a bogus held temperature.
    # Bug 3b: two consecutive temperature ramps at the SAME held field (settle gap between them)
    #          must merge into one segment.
    rng = np.random.default_rng(0)
    # ramp A (held field 0), then a transition (field steps 0->50000 while T resets 30->2),
    # then ramp B part 1 and ramp B part 2 at held field 50000 (a settle gap between them).
    Ta = np.linspace(2.0, 30.0, 60); Ha = 0.5 + 0.02 * rng.standard_normal(60)
    Ht = np.linspace(0, 50000, 30); Tt = np.linspace(30.0, 2.0, 30)             # diagonal transition
    Tb1 = np.linspace(2.0, 16.0, 30); Hb1 = 50000 + 0.5 * rng.standard_normal(30)
    Tb2 = np.linspace(16.1, 30.0, 30); Hb2 = 50000 + 0.5 * rng.standard_normal(30)
    df = _df(np.concatenate([Ta, Tt, Tb1, Tb2]), np.concatenate([Ha, Ht, Hb1, Hb2]))
    segs = segment_sweeps(df, CMAP, CFG)
    # no field segment with a spurious held temperature in the transition range
    assert all(s.swept.name == "temperature" for s in segs), [s.swept.name for s in segs]
    # held fields recovered: one ramp at ~0, one merged ramp at ~50000 (not two)
    held = sorted(round(s.setpoint["field"], 0) for s in segs)
    at_50k = [h for h in held if abs(h - 50000) <= 3000]
    assert len(at_50k) == 1, held         # split ramp merged into ONE segment

def test_mixed_mode_two_blocks():
    # THE key case: a temperature ramp THEN a field loop in one file -> 2 blocks, different swept axes
    T1 = np.linspace(2.0, 30.0, 150); H1 = 0.5 + 0.03 * RNG.standard_normal(150)
    H2 = np.concatenate([np.linspace(0, 90000, 120), np.linspace(90000, -90000, 240), np.linspace(-90000, 0, 120)])
    T2 = 2.0 + 0.01 * RNG.standard_normal(H2.size)
    df = _df(np.concatenate([T1, T2]), np.concatenate([H1, H2]))
    blocks = find_blocks(df, CMAP, CFG)
    swept = [b.swept_axis for b in blocks]
    assert swept.count("temperature") == 1
    assert swept.count("field") >= 1            # the loop may be 1 field block (split into branches downstream)
    assert swept[0] == "temperature"            # order preserved


import json, pathlib
from cryosweep_core.io.loader import load_dat
from cryosweep_core.io.columns import canonicalize_columns

EXPECTED = json.loads((pathlib.Path(__file__).parent / "fixtures" / "expected_segments.json").read_text())

def _blocks(path):
    rt = load_dat(path); df, cmap = canonicalize_columns(rt.df, rt.header)
    return find_blocks(df, cmap, CFG), df, cmap

def test_hc_is_four_temperature_ramps(hc_path):
    # Bug 3 + Bug 7: EXACTLY 4 temperature ramps, 0 field segments, all +1, held fields recovered.
    blocks, df, cmap = _blocks(hc_path)
    exp = EXPECTED["heat_capacity"]
    segs = segment_sweeps(df, cmap, CFG)
    temp_segs = [s for s in segs if s.swept.name == "temperature"]
    field_segs = [s for s in segs if s.swept.name == "field"]
    assert len(segs) == exp["n_temperature_blocks"]            # exactly 4 segments total
    assert len(temp_segs) == exp["n_temperature_blocks"]       # all of them temperature ramps
    assert len(field_segs) == 0                                # NO spurious field transition blocks
    assert all(s.direction == 1 for s in temp_segs)            # all ramps go up
    # the 4 held fields are recovered, one segment per held field (no split ramps)
    held = sorted(s.setpoint.get("field") for s in temp_segs)
    assert len(held) == len(exp["held_fields"])
    for f in exp["held_fields"]:
        assert sum(abs(f - h) <= exp["field_band"] for h in held) == 1, (f, held)

def test_res_is_mixed_mode(res_path):
    blocks, df, cmap = _blocks(res_path)
    exp = EXPECTED["resistivity"]
    field_blocks = [b for b in blocks if b.swept_axis == "field"]
    temp_blocks = [b for b in blocks if b.swept_axis == "temperature"]
    assert len(field_blocks) >= exp["n_field_blocks_min"]       # the MR loops
    assert len(temp_blocks) >= exp["n_temperature_blocks_min"]  # the R(T) isotherms + ramps
    segs = segment_sweeps(df, cmap, CFG)
    # field-loops are held at the expected temperatures
    loop_T = [s.setpoint.get("temperature") for s in segs if s.swept.name == "field"]
    for t in exp["field_loop_temperatures"]:
        assert any(abs(t - g) <= exp["temp_band"] for g in loop_T), t
    # Bug 7: NO spurious held-temperature setpoint outside the expected set for field segments
    # (the buggy detector produced 4.92 / 12.27 / 17.12 / 21.94 K transition blocks).
    allowed_T = exp["field_loop_temperatures"]
    for g in loop_T:
        assert any(abs(t - g) <= exp["temp_band"] for t in allowed_T), \
            f"spurious field-segment held temperature {g} (not in {allowed_T})"
    # R(T) isotherms are held at the expected fields
    iso_H = [s.setpoint.get("field") for s in segs if s.swept.name == "temperature"]
    for h in exp["isofield_isotherm_fields"]:
        assert any(abs(h - g) <= exp["field_band"] for g in iso_H), h

def test_synthetic_regressions():
    import tests.core.fixtures.make_synthetic as mk
    from cryosweep_core.io.columns import ColumnMap
    cmap = ColumnMap(logical={"temperature": "temperature", "field": "field"},
                     unit={"temperature": "K", "field": "Oe"})
    assert len(find_blocks(mk.ramp(), cmap, CFG)) == 1                    # ramp not shattered
    mb = find_blocks(mk.mixed_two_block(), cmap, CFG)
    assert [b.swept_axis for b in mb].count("temperature") == 1          # exactly one T block
    assert any(b.swept_axis == "field" for b in mb)                       # and the loop


import pytest

@pytest.mark.skip(reason="Informational (spec layer b/c), non-gating: documents v1 vs v2 divergence.")
def test_v1_crosscheck_documented():
    """v1's utils.separate_data picks a single global swept axis and therefore
    mislabels the MIXED-MODE Resistivity file (field-swept MR loops AND
    temperature-swept R(T) isotherms in one .dat). v2's per-block find_blocks
    resolves it into 7 field-swept blocks + 15 temperature-swept blocks
    (confirmed against the raw canonicalized T/H columns). Running v1 here would
    import Qt, so this divergence is recorded as documentation rather than
    executed in-process; the golden oracle (test_res_is_mixed_mode) is the gate.
    """
