# tests/core/test_hall_rho_xx_zero_field.py
"""rho_xx for mobility must be a ZERO-FIELD quantity.

_long_rho_xx used to mean over every row at a temperature with no field mask. On a
magnetoresistive longitudinal channel that averages the whole field loop into the
denominator of mu = |R_H|/rho_xx. Measured on the real Hall file: the mean sat 60.8%
above the |H| < 50 Oe value at 2 K, so mu was 38% low, and the reported rho_xx was
non-monotonic (2 K above 5 K).
"""
import types

import numpy as np
from cryosweep_core.io.loader import load_dat
from cryosweep_core.config import RunConfig
from cryosweep_core.analyzers.hall import HallAnalyzer, _mobility_gap_reason

_HDR = ("[Header]\nBYAPP, Resistivity\nINFO, mr_synth, SAMPLE\n[Data]\n"
        "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
        "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")

# rho_xx(H) = rho0 * (1 + 1.0 * (H/1e4)^2): a strongly magnetoresistive longitudinal
# channel. At |H| <= 50 Oe it is rho0 to 2.5e-5 relative; averaged over a +-2 T loop it
# is ~1.33 * rho0. The two answers differ by a third, which is the point of the test.
_RHO0 = 1e-6

def _row(T, B_oe):
    rxy = 1e-3 + 5e-4 * (B_oe / 1e4)
    rho = _RHO0 * (1.0 + (B_oe / 1e4) ** 2)
    return f"{T:.4f},{B_oe:.1f},{rxy:.10e},{rho * 1e3:.10e},{rho:.10e}"

def _write(tmp_path, include_zero=True):
    rows = []
    for T in (10.0, 50.0):
        fields = np.arange(-20000.0, 20000.1, 250.0)
        if not include_zero:                       # drop everything inside +-50 Oe
            fields = fields[np.abs(fields) > 50.0]
        rows += [_row(T, b) for b in fields]
        # 20-row separator. It MUST exceed StabilityCfg.window (16, config.py:41): the
        # rolling-activity classifier uses a centred window of that size, so a shorter
        # separator lets the neighbouring loop's field activity bleed across the gap.
        # The two loops then merge into one block, the fixed-axis drift guard rejects
        # the merged stretch, and the T=10 point silently disappears.
        rows += [_row(t, 20000.0) for t in np.arange(T + 1, T + 40, 2.0)]  # ramp separator
    p = tmp_path / ("mr.dat" if include_zero else "mr_nozero.dat")
    p.write_text(_HDR + "\n".join(rows) + "\n")
    return p

def _analyze(path):
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1,
                          "longitudinal_channel": 2})
    return HallAnalyzer().analyze(load_dat(path), cfg)

def test_rho_xx_is_the_zero_field_value_not_the_loop_average(tmp_path):
    res = _analyze(_write(tmp_path))
    pts = {p["temperature"]: p for p in res.data["points"]}
    assert set(pts) >= {10.0, 50.0}
    for T, p in pts.items():
        # zero-field value, not the ~1.33x loop average
        assert abs(p["rho_xx"] - _RHO0) / _RHO0 < 0.01, (T, p["rho_xx"])

def test_point_records_the_field_its_rho_xx_came_from(tmp_path):
    res = _analyze(_write(tmp_path))
    for p in res.data["points"]:
        assert p["rho_xx_field_oe"] is not None
        assert p["rho_xx_field_oe"] <= 50.0            # already a median of |H|, never negative

def test_mobility_declines_when_no_near_zero_field_row_exists(tmp_path):
    res = _analyze(_write(tmp_path, include_zero=False))
    for p in res.data["points"]:
        assert p["mobility"] is None
        assert p["rho_xx"] is None
        assert "rho_xx_no_zero_field" in p["derived_flags"]


def _write_two_temps_one_missing_zero_field(tmp_path):
    """Review round 1, Important #1: a 10 K loop with every |H| > 50 Oe (no zero-field
    row at all) beside a 50 K loop that has one. True zero-field rho_xx would be 2e-6 at
    10 K and 6e-6 at 50 K, but 10 K's true value is never measurable from this file --
    the point is that the analyzer must say so, not hand 10 K the 50 K value."""
    hdr = ("[Header]\nBYAPP, Resistivity\nINFO, two_t_synth, SAMPLE\n[Data]\n"
           "Temperature (K),Magnetic Field (Oe),Bridge 1 Resistance (Ohms),"
           "Bridge 2 Resistance (Ohms),Bridge 2 Resistivity (Ohm-m)\n")
    def row(T, B_oe, rho):
        rxy = 1e-3 + 5e-4 * (B_oe / 1e4)
        return f"{T:.4f},{B_oe:.1f},{rxy:.10e},{rho * 1e3:.10e},{rho:.10e}"
    rows = []
    fields_10 = np.arange(-20000.0, 20000.1, 250.0)
    fields_10 = fields_10[np.abs(fields_10) > 50.0]     # skip the +-50 Oe crossing entirely
    rows += [row(10.0, b, 2e-6) for b in fields_10]
    # 20-row separator -- see the window-size comment on the +-40 range above.
    rows += [row(t, 20000.0, 2e-6) for t in np.arange(11.0, 50.0, 2.0)]
    fields_50 = np.arange(-20000.0, 20000.1, 250.0)     # includes H=0
    rows += [row(50.0, b, 6e-6) for b in fields_50]
    p = tmp_path / "two_t.dat"
    p.write_text(hdr + "\n".join(rows) + "\n")
    return p

def test_setpoint_without_its_own_zero_field_row_declines_not_borrows(tmp_path):
    res = _analyze(_write_two_temps_one_missing_zero_field(tmp_path))
    pts = {p["temperature"]: p for p in res.data["points"]}
    assert set(pts) >= {10.0, 50.0}
    # 10 K has no zero-field row of its own: must decline, never borrow 50 K's 6e-6
    assert pts[10.0]["rho_xx"] is None
    assert pts[10.0]["mobility"] is None
    assert pts[10.0]["rho_xx_field_oe"] is None
    assert "rho_xx_no_zero_field" in pts[10.0]["derived_flags"]
    # 50 K genuinely has one: must still resolve correctly
    assert abs(pts[50.0]["rho_xx"] - 6e-6) / 6e-6 < 1e-6
    assert pts[50.0]["rho_xx_field_oe"] is not None and pts[50.0]["rho_xx_field_oe"] <= 50.0


def test_capability_reason_names_the_actual_cause_not_a_contradiction(tmp_path):
    """Review round 1, Important #2: the mobility capability's decline reason must not
    contradict longitudinal_source. It previously always said "no longitudinal
    channel/file supplied" even when one clearly was -- self-contradicting in the same
    JSON object whenever a channel was supplied but produced no usable rho_xx."""
    res = _analyze(_write(tmp_path, include_zero=False))
    assert res.data["longitudinal_source"] == "same_file:ch2"
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["mobility"]["applicable"] is False
    reason = caps["mobility"]["reason"]
    assert "same_file:ch2" in reason
    assert "no longitudinal channel/file supplied" not in reason
    assert "no |H| <" in reason


def test_missing_longitudinal_channel_is_not_diagnosed_as_no_zero_field(tmp_path):
    """Review round 1, Important #3: four different causes used to collapse into one
    "no zero field row" message. Realistic case cited by the reviewer: --long-channel
    pointing at a bridge with no data in the file at all (field_sweep_points reads that
    channel's RESISTANCE column, so its RESISTIVITY column can be legitimately absent
    even though the Hall channel itself is fine) must not send the user hunting for a
    zero-field row that was never the issue."""
    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1, "longitudinal_channel": 3})
    res = HallAnalyzer().analyze(load_dat(_write(tmp_path)), cfg)
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["mobility"]["applicable"] is False
    reason = caps["mobility"]["reason"]
    assert "column not found" in reason
    assert "no |H| <" not in reason
    for p in res.data["points"]:
        assert "rho_xx_channel_missing" in p["derived_flags"]
        assert "rho_xx_no_zero_field" not in p["derived_flags"]


def test_capability_reason_names_temperature_misalignment_not_missing_zero_field_data(tmp_path):
    """Review round 2, Important #1: a regression round 1's per-point decline created,
    not a pre-existing gap. A --long-file that genuinely HAS zero-field rows -- just at
    temperatures nowhere near any Hall setpoint -- must not be diagnosed as "carries no
    |H| < 50 Oe row", which is false: the source does carry such rows, they simply don't
    align (within temp_interval) with any Hall setpoint. The old file-wide-only decline
    could never produce this combination: its unconditional np.interp clamp always
    resolved once ANY zero-field row existed anywhere. Round 1's per-point coverage check
    made "_long_rho_xx succeeds at the file level, but every setpoint still declines"
    possible, and _mobility_gap_reason only ever looked at the file-level signal."""
    hall_path = _write(tmp_path)          # Hall setpoints at 10 K and 50 K
    # A separate longitudinal file whose zero-field rows sit at 200/250 K -- far outside
    # the default HallCfg.temp_interval (1.0 K) of either Hall setpoint.
    long_hdr = ("[Header]\nBYAPP, Resistivity\nINFO, long_only, SAMPLE\n[Data]\n"
                "Temperature (K),Magnetic Field (Oe),Bridge 2 Resistivity (Ohm-m)\n")
    long_rows = [f"{T:.4f},{b:.1f},{1e-6:.10e}"
                 for T in (200.0, 250.0) for b in (-100.0, 0.0, 100.0)]
    long_path = tmp_path / "long_only.dat"
    long_path.write_text(long_hdr + "\n".join(long_rows) + "\n")

    cfg = RunConfig(hall={"hall_channel": 1, "thickness_mm": 0.1,
                          "longitudinal_channel": 2, "longitudinal_file": str(long_path)})
    res = HallAnalyzer().analyze(load_dat(hall_path), cfg)
    assert res.data["longitudinal_source"].startswith("file:")
    assert res.data["points"]                          # Hall fits still resolve fine
    for p in res.data["points"]:
        assert p["mobility"] is None and p["rho_xx"] is None
    caps = {c["name"]: c for c in res.data["capabilities"]}
    assert caps["mobility"]["applicable"] is False
    reason = caps["mobility"]["reason"].lower()
    assert "no |h| <" not in reason        # must NOT claim missing zero-field data (false)
    assert "temp" in reason                # must name the real cause: temperature misalignment


def _pt(mobility, flags):
    """Minimal stand-in for a HallTempPoint: _mobility_gap_reason only ever reads
    .mobility and .derived_flags off each point, so a real point is unnecessary here."""
    return types.SimpleNamespace(mobility=mobility, derived_flags=flags)


def test_mixed_decline_causes_are_not_generalised_to_one(tmp_path):
    """Review round 3, Finding B: round 2's fix reported the temperature-misalignment
    cause whenever ANY declining point carried the rho_xx flag -- an existential claim
    presented as universal. A point whose own rho_xx succeeded (no rho_xx flag at all)
    but whose mobility is still None because its R_H fit failed is a real, reachable
    case (field_sweep_points sets mobility from R_H and rho_xx independently), and the
    old wording asserted something false of it: that its zero-field row was missing too."""
    points = [_pt(mobility=None, flags=["rho_xx_no_zero_field"]),   # genuinely misaligned
              _pt(mobility=None, flags=[])]                          # failed for another reason
    reason = _mobility_gap_reason("file:x.dat:ch2", None, points).lower()
    assert "none fall within" not in reason         # not the universal misalignment claim
    assert "no |h| <" not in reason                 # not the universal no-zero-field claim
    assert "some" in reason                         # names the mix, not a single cause


def test_reason_without_per_point_evidence_names_no_specific_cause():
    """Review round 3, Finding A: hall_tempdep.py's _capabilities calls
    _mobility_gap_reason(long_source, rho_reason) with only two arguments -- HallTDepPoint
    has no derived_flags until Task 5, so `points` defaults to None there and can never
    be supplied. With no per-point evidence at all, the reason must not assert either of
    the per-point-specific claims (round 2's misalignment text, or the plain no-zero-field
    text): both would be diagnoses this call site has no evidence for."""
    reason = _mobility_gap_reason("file:x.dat:ch2", None, points=None).lower()
    assert "no |h| <" not in reason
    assert "none fall within" not in reason
