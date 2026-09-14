# tests/core/test_hall_fit_quality_basis.py
"""What the `fit` ceiling averages, and what it does when there is nothing to average.

Two defects with one root -- absent evidence certifying a result:

(a) `fit_quality` averaged r2 over ALL points, declined ones included. A point whose carrier
    density was withheld contributes nothing to the published result, so its r2 must not
    move the published result's fit ceiling. r2 itself STAYS on declined points -- sigma and
    r2 are different claims and are never conflated -- only the aggregate's basis changes,
    and `fit_n` says how many r2 went into the mean.

(b) `fit_quality = mean(r2s) if r2s else 1.0`: no fit evidence scored as a PERFECT fit while
    the envelope reported `fit: None` -- the computation and the JSON disagreed about the
    same quantity. r2 is absent BY CONSTRUCTION for the one-pair-per-temperature protocol
    (measured: 138/138 points on the real Hall file, 130/130 on the shipped subset), so this
    was not an edge case. The fix is NOT to score it 0.0 either: `min()` makes 0.0 absorbing
    and would pin confidence at exactly 0.0 on every such file, erasing the informative
    `resolved` term -- the same conflation inverted, and against this repository's own
    convention (`power_law_n_spread` is None, never 0.0, plus a flag that annotates without
    revoking). Instead the fit term is DROPPED from the min, the status is capped at
    low_confidence, and a machine-readable `fit_quality_unavailable` flag plus a warning say
    why. A non-finite r2 (a constant-y channel) is likewise no evidence, not a number.
"""
import math
import pathlib
import numpy as np
import pytest

from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.registry import build_default_registry
from cryosweep_core.analyzers.dispatch import analyze_file
from cryosweep_core.analyzers.hall import HallAnalyzer, hall_confidence
from cryosweep_core.analyzers.hall_tempdep import HallTempDepAnalyzer

EXAMPLES = pathlib.Path(__file__).resolve().parents[2] / "examples"
_REG = build_default_registry()
_RATIO = 1000.0


# ---- the shared rule ---------------------------------------------------------------------

def test_hall_confidence_with_fit_evidence_is_unchanged():
    status, conf, flags = hall_confidence(0.9, 0.8, 0.5)
    assert (status, conf) == ("ok", 0.8)
    assert flags == []


def test_hall_confidence_without_fit_evidence_annotates_and_caps_but_does_not_revoke():
    status, conf, flags = hall_confidence(None, 0.8, 0.5)
    assert conf == 0.8, "the remaining ceiling carries the confidence -- not 1.0, not 0.0"
    assert status == "low_confidence", "no fit evidence can never certify `ok`"
    assert "fit_quality_unavailable" in flags


def test_hall_confidence_treats_non_finite_fit_as_absent():
    status, conf, flags = hall_confidence(float("nan"), 0.6, 0.5)
    assert conf == 0.6 and status == "low_confidence"
    assert "fit_quality_unavailable" in flags


# ---- (a) the aggregate's basis: published points only -----------------------------------

_TDEP_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, fitbasis_synth, SAMPLE\n[Data]\n"
             "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
             "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m)\n")


def _write_tdep_three_pairs(tmp_path, name="fitbasis.dat", sd_split=30.0):
    """Three held-field pairs (so every T has antisym_points == 3 and a real r2), each its
    own T ramp. An odd-in-B scatter whose amplitude grows with T makes r2 fall with T, and
    the instrument sigma is tiny below `sd_split` (resolved) and huge above (withheld) --
    so the declined points carry a DIFFERENT r2 population than the published ones."""
    rows = []
    for b, sign in ((10000.0, 1.0), (-10000.0, -1.0), (20000.0, 1.0), (-20000.0, -1.0),
                    (30000.0, 1.0), (-30000.0, -1.0)):
        for i in range(70):
            T = 2.0 + i
            amp = 4e-4 * T / 70.0
            rxy = 1e-3 + sign * 5e-4 * (abs(b) / 1e4) + amp * math.sin(b / 1300.0)
            sd_rho = 1e-12 if T < sd_split else 1e-4
            rows.append(f"{T:.4f},{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e}")
    p = tmp_path / name
    p.write_text(_TDEP_HDR + "\n".join(rows) + "\n")
    return p


def test_tempdep_fit_averages_r2_over_published_points_only(tmp_path):
    res = HallTempDepAnalyzer().analyze(load_dat(_write_tdep_three_pairs(tmp_path)),
                                        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1}))
    pts = res.data["points"]
    published = [p for p in pts if p["carrier_n"] is not None]
    declined = [p for p in pts if p["carrier_n"] is None]
    assert published and declined, "fixture must mix published and withheld points"
    assert all(p["r2"] is not None for p in declined), "r2 stays on declined points"
    pub_r2 = [p["r2"] for p in published]
    all_r2 = [p["r2"] for p in pts if p["r2"] is not None]
    assert np.mean(pub_r2) != pytest.approx(np.mean(all_r2)), "fixture must discriminate"
    parts = res.confidence_parts
    assert parts["fit"] == pytest.approx(float(np.mean(pub_r2)))
    assert parts["fit_n"] == len(published)


_HALL_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, fitbasis_fs_synth, SAMPLE\n[Data]\n"
             "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
             "Bridge 1 Resistivity (Ohm-m),Bridge 1 Std. Dev. (Ohm-m)\n")


def _write_hall_two_loops(tmp_path):
    """Two held-T field loops with different scatter (so different r2): 10 K resolved
    (tiny instrument sigma), 20 K withheld (huge)."""
    rows = []
    for T, sd_rho, amp in ((10.0, 1e-12, 1e-6), (20.0, 1e-4, 8e-6)):
        for b in np.arange(-20000.0, 20000.1, 1000.0):
            rxy = 1e-3 + 5e-4 * (b / 1e4) + amp * math.sin(b / 1300.0)
            rows.append(f"{T:.4f},{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e}")
    p = tmp_path / "fitbasis_fs.dat"
    p.write_text(_HALL_HDR + "\n".join(rows) + "\n")
    return p


def test_field_sweep_fit_averages_r2_over_published_points_only(tmp_path):
    res = HallAnalyzer().analyze(load_dat(_write_hall_two_loops(tmp_path)),
                                 RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1}))
    pts = {p["temperature"]: p for p in res.data["points"]}
    assert pts[10.0]["carrier_n"] is not None and pts[20.0]["carrier_n"] is None
    assert pts[20.0]["r2"] is not None
    assert pts[10.0]["r2"] != pytest.approx(pts[20.0]["r2"])
    assert res.confidence_parts["fit"] == pytest.approx(pts[10.0]["r2"])
    assert res.confidence_parts["fit_n"] == 1


# ---- (b) no fit evidence: annotate and cap, never certify, never revoke -----------------

def _write_tdep_two_pairs_majority_resolved(tmp_path):
    """Two pairs only -> antisym_points == 2 everywhere -> r2 None by the zero-DOF rule.
    Resolved on instrument sigma below 50 K (measured fraction 0.8205)."""
    rows = []
    for b, sign in ((10000.0, 1.0), (-10000.0, -1.0), (20000.0, 1.0), (-20000.0, -1.0)):
        rxy = 1e-3 + sign * 5e-4 * (abs(b) / 1e4)
        for i in range(70):
            T = 2.0 + i
            sd_rho = 1e-12 if T < 50.0 else 1e-4
            rows.append(f"{T:.4f},{b:.1f},{rxy:.10e},{rxy / _RATIO:.10e},{sd_rho:.10e}")
    p = tmp_path / "nofit.dat"
    p.write_text(_TDEP_HDR + "\n".join(rows) + "\n")
    return p


def test_tempdep_without_fit_evidence_is_capped_not_scored(tmp_path):
    res = HallTempDepAnalyzer().analyze(load_dat(_write_tdep_two_pairs_majority_resolved(tmp_path)),
                                        RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1}))
    parts = res.confidence_parts
    assert parts["fit"] is None and parts["fit_n"] == 0
    assert 0.5 <= parts["resolved"] < 1.0, "fixture must be majority-resolved so the cap is what bites"
    assert res.confidence == parts["resolved"], "confidence is the remaining ceiling, not 0.0"
    assert res.status == "low_confidence"
    assert "fit_quality_unavailable" in res.data["flags"]
    assert any("fit quality" in w.lower() for w in res.warnings)


def test_field_sweep_non_finite_r2_is_no_evidence():
    # channel 2 of the shipped field-sweep example is a constant: r2 is nan on every loop.
    # Before: confidence nan (nan >= 0.5 is False, so the status happened to read
    # low_confidence by accident). Now the nan is not evidence, and the envelope says so.
    cfg = RunConfig.load(probe_override="hall")
    cfg.hall.hall_channel = 2
    cfg.hall.thickness_mm = 0.07
    res = analyze_file(load_dat(str(EXAMPLES / "hall_field_sweeps.dat")), cfg, _REG)
    assert not math.isnan(res.confidence)
    assert res.confidence_parts["fit"] is None
    assert res.confidence == res.confidence_parts["resolved"]
    assert res.status == "low_confidence"
    assert "fit_quality_unavailable" in res.data["flags"]


def test_shipped_subset_is_capped_by_the_absent_fit_evidence():
    # Accepted consequence: examples/hall_mixed_sweeps.dat channel 1 moves ok -> low_confidence
    # (exit 0 -> 11). r2 is absent on all 130 of its points by construction (one +-pair per
    # temperature), so the old 1.0 default was certifying a fit that was never measured.
    # Confidence itself is unchanged: 73/130 resolved.
    cfg = RunConfig.load(probe_override="hall_tdep")
    cfg.hall.hall_channel = 1
    cfg.hall.thickness_mm = 0.07
    res = analyze_file(load_dat(str(EXAMPLES / "hall_mixed_sweeps.dat")), cfg, _REG)
    assert res.confidence == pytest.approx(73 / 130)
    assert res.confidence_parts["fit"] is None
    assert res.status == "low_confidence"
    assert "fit_quality_unavailable" in res.data["flags"]
